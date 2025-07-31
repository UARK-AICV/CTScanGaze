"""
.. Part of GazeParser package.
.. Copyright (C) 2012-2015 Hiroyuki Sogo.
.. Distributed under the terms of the GNU General Public License (GPL).

Evaluating simirality between two fixation sequences with ScanMatch algorithm,
proposed by Cristino, Mathot, Theeuwes and Gilchrist (2010).

Example
------------
Following script compares fixation sequence of two participants.::

    import GazeParser
    (data1, additionalData1) = GazeParser.load('participant1.db')
    (data2, additionalData2) = GazeParser.load('participant2.db')

    #create a ScanMatch object.
    matchObject = ScanMatch(Xres=720, Yres=720, Xbin=4, Ybin=4, offset=(152, 24), Threshold=1.5)

    #convert fixations to a sequence of symbols.
    sequence1 = sObj.fixationToSequence(data1[0].getFixationCenter())
    sequence2 = sObj.fixationToSequence(data2[0].getFixationCenter())

    #perform ScanMatch
    (score, align, f) = matchObject.match(sequence1, sequence2)

REFERENCE:
 Cristino, F., Mathot, S., Theeuwes, J., & Gilchrist, I. D. (2010).
 ScanMatch: a novel method for comparing fixation sequences.
 Behav Res Methods, 42(3), 692-700.
"""

from __future__ import absolute_import, division, print_function

import numpy
import numpy as np
from numba import jit, njit


@njit
def cal_distance(mat, Xbin, Ybin, Zbin):
    indI = 0
    indJ = 0
    for i in range(Ybin):
        for j in range(Xbin):
            for k in range(Zbin):
                for ii in range(Ybin):
                    for jj in range(Xbin):
                        for kk in range(Zbin):
                            mat[indI, indJ] = numpy.sqrt(
                                (j - jj) ** 2 + (i - ii) ** 2 + (k - kk) ** 2
                            )
                            indI += 1
                indI = 0
                indJ += 1
    return mat


@jit(nopython=True)
def fill_alignment_matrix(A, B, sub_matrix, gap_value):
    """
    Fill the sequence alignment matrix using JIT-compiled code
    Needleman-Wunsch algorithm => try to align two sequence, then return all the alignment scores, we will get the best possible alignment score for comparison.
    Args:
        A: First sequence (as integer indices)
        B: Second sequence (as integer indices)
        sub_matrix: Substitution matrix for scoring matches/mismatches
        gap_value: Penalty for gaps

    Returns:
        F: The filled dynamic programming matrix
    """
    n, m = len(A), len(B)
    F = np.zeros((n + 1, m + 1), dtype=np.float32)
    F = numpy.zeros((n + 1, m + 1))
    for i in range(n + 1):
        F[i, 0] = gap_value * (i + 1)
    for j in range(m + 1):
        F[0, j] = gap_value * (j + 1)

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            match = F[i - 1, j - 1] + sub_matrix[A[i - 1], B[j - 1]]
            delete = F[i - 1, j] + gap_value
            insert = F[i, j - 1] + gap_value
            F[i, j] = max(match, delete, insert)
    return F


class ScanMatch(object):
    """
    ScanMatch Object.
    """

    def __init__(self, **kw):
        """
        :param int Xres:
        :param int Yres:
        :param int Xbin:
        :param int Ybin:
        :param float Threshold:
        :param float GapValue:
        :param float TempBin:
        :param (int, int) Offset:
        """
        self.Xres = 1024
        self.Yres = 768
        self.Zres = 512
        self.Xbin = 8
        self.Ybin = 6
        self.Zbin = 6
        self.Threshold = 3.5
        self.GapValue = 0.0
        self.TempBin = 0.0
        self.Offset = (0, 0, 0)

        for k in kw.keys():
            if k == "Xres":
                self.Xres = kw[k]
            elif k == "Yres":
                self.Yres = kw[k]
            elif k == "Zres":
                self.Zres = kw[k]
            elif k == "Xbin":
                self.Xbin = kw[k]
            elif k == "Ybin":
                self.Ybin = kw[k]
            elif k == "Zbin":
                self.Zbin = kw[k]
            elif k == "Threshold":
                self.Threshold = kw[k]
            elif k == "GapValue":
                self.GapValue = kw[k]
            elif k == "TempBin":
                self.TempBin = kw[k]
            elif k == "Offset":
                self.Offset = kw[k]
            else:
                raise ValueError("Unknown parameter: %s." % k)

        self.int_vectorize = numpy.vectorize(int)

        self.CreateSubMatrix()
        self.GridMask()

    def CreateSubMatrix(self, Threshold=None):
        if Threshold is not None:
            self.Threshold = Threshold
        mat = numpy.zeros(
            (self.Xbin * self.Ybin * self.Zbin, self.Xbin * self.Ybin * self.Zbin)
        )

        mat = cal_distance(mat, self.Xbin, self.Ybin, self.Zbin)
        max_sub = numpy.max(mat)

        self.SubMatrix = numpy.abs(mat - max_sub) - (max_sub - self.Threshold)

    def GridMask(self):
        # not entirely masking, this will just put the coordinate into a particular positional bin.
        a = numpy.reshape(
            numpy.arange(self.Xbin * self.Ybin * self.Zbin),
            (self.Ybin, self.Xbin, self.Zbin),
        )
        m = float(self.Xbin) / self.Xres
        n = float(self.Ybin) / self.Yres
        l = float(self.Zbin) / self.Zres
        xi = numpy.int32(numpy.arange(0, self.Xbin, m))  # m and n are >0 and <= 1
        yi = numpy.int32(numpy.arange(0, self.Ybin, n))
        zi = numpy.int32(numpy.arange(0, self.Zbin, l))
        # print(xi.shape) # (512,)

        self.mask = numpy.zeros((self.Yres, self.Xres, self.Zres))
        self.mask = a[np.ix_(yi, xi, zi)]

    def fixationToSequence(self, data):
        """
        Converts fixation data to a sequence of region indices.
        This function processes the input fixation data to ensure all values are within bounds,
        then maps the fixation points to a sequence of region indices based on a predefined mask.
        If a temporal bin size is specified, the function also takes the duration of fixations into account,
        duplicating region indices according to the duration.
        Parameters:
        data (numpy.ndarray): A 2D array of shape (length, 4) where each row represents a fixation point
                              with x, y, z coordinates and a timestamp.
        Returns:
        numpy.ndarray: A 1D array of region indices corresponding to the fixation points.
        """

        d = data.copy()
        # print(d.shape) # (length, 4) for xyzt
        d[:, :3] -= self.Offset
        d[d < 0] = 0
        d[d[:, 0] >= self.Xres, 0] = self.Xres - 1
        d[d[:, 1] >= self.Yres, 1] = self.Yres - 1
        d[d[:, 2] >= self.Zres, 2] = self.Zres - 1
        d = self.int_vectorize(d)
        # the above steps are just for making sure all values is in bound
        seq_num = self.mask[d[:, 1], d[:, 0], d[:, 2]]

        # print(seq_num.shape) # (length, ) = [idx_1, idx2, ...]
        # temp bin means if we take duration into account.
        if self.TempBin != 0:
            fix_time = numpy.round(d[:, 3] / float(self.TempBin))
            # print(fix_time.shape, fix_time) # (length,) for example 10 length can have [ 7. 10. 10. 11. 12. 13. 14. 14. 17.  0.]
            tmp = []
            for f in range(d.shape[0]):
                tmp.extend([seq_num[f] for _ in range(int(fix_time[f]))])
            # an extra step where you further duplicate the position bin based on the occurance (duration) of a bin.
            # if a bin 1 occurs twice, then the new list is 1 1
            seq_num = numpy.array(tmp)
        return seq_num

    def match(self, A, B):
        """
        Align two sequences A and B using dynamic programming and calculate a normalized match score.
        Parameters
        ----------
        A : array-like
            The first sequence to align.
        B : array-like
            The second sequence to align.
        Returns
        -------
        matchScore : float
            The normalized match score between 0 and 1.
        align : ndarray
            The optimal alignment between sequences A and B. Each row represents an aligned pair.
        F : ndarray
            The alignment score matrix.
        Notes
        -----
        This method uses a substitution matrix (`self.SubMatrix`) and a gap penalty (`self.GapValue`)
        to perform global alignment between the sequences A and B. It fills the alignment matrix,
        performs traceback to determine the optimal alignment, and computes a normalized match score.
        """

        n = len(A)
        m = len(B)

        # print("3.3.0")
        F = fill_alignment_matrix(A, B, self.SubMatrix, self.GapValue)

        # print("3.3.1")
        AlignmentA = numpy.zeros(n + m) - 1
        AlignmentB = numpy.zeros(n + m) - 1
        i = n
        j = m
        step = 0

        while i > 0 and j > 0:
            score = F[i, j]
            scoreDiag = F[i - 1, j - 1]
            # scoreUp = F[i, j-1]
            scoreLeft = F[i - 1, j]

            if score == scoreDiag + self.SubMatrix[A[i - 1], B[j - 1]]:
                AlignmentA[step] = A[i - 1]
                AlignmentB[step] = B[j - 1]
                i -= 1
                j -= 1
            elif score == scoreLeft + self.GapValue:
                AlignmentA[step] = A[i - 1]
                i -= 1
            else:
                AlignmentB[step] = B[j - 1]
                j -= 1

            step += 1
        # print("3.3.2")

        while i > 0:
            AlignmentA[step] = A[i - 1]
            i -= 1
            step += 1

        while j > 0:
            AlignmentB[step] = B[j - 1]
            j -= 1
            step += 1
        # print("3.3.3")
        F = F.transpose()

        maxF = numpy.max(F)
        maxSub = numpy.max(self.SubMatrix)
        scale = maxSub * max((m, n))
        matchScore = maxF / scale

        align = numpy.vstack(
            [AlignmentA[step - 1 :: -1], AlignmentB[step - 1 :: -1]]
        ).transpose()

        return matchScore, align, F

    def maskFromArray(self, array):
        self.mask = array

    def subMatrixFromArray(self, array):
        self.SubMatrix = array


def generateMaskFromArray(data, threshold, margeColor):
    dataArray = data.copy()
    uniqueData = numpy.unique(dataArray)
    for i in range(len(uniqueData)):
        index = numpy.where(dataArray == uniqueData[i])
        if len(index[0]) <= threshold:
            dataArray[index] = margeColor

    uniqueData2 = numpy.unique(dataArray)
    for i in range(len(uniqueData2)):
        index = numpy.where(dataArray == uniqueData2[i])
        dataArray[index] = i

    return dataArray, uniqueData2


if __name__ == "__main__":
    import numpy as np
    import scipy.io as sio

    mat_fname = "ScanMatch_DataExample.mat"
    mat_contents = sio.loadmat(mat_fname)

    data1 = mat_contents["data1"]
    data2 = mat_contents["data2"]
    data3 = mat_contents["data3"]
    # create a ScanMatch object.
    ScanMatchwithDuration = ScanMatch(
        Xres=1024, Yres=768, Xbin=12, Ybin=8, Offset=(0, 0), TempBin=100, Threshold=3.5
    )
    ScanMatchwithoutDuration = ScanMatch(
        Xres=1024, Yres=768, Xbin=12, Ybin=8, Offset=(0, 0), Threshold=3.5
    )

    sequence1 = ScanMatchwithDuration.fixationToSequence(data1).astype(np.int32)
    sequence2 = ScanMatchwithDuration.fixationToSequence(data2).astype(np.int32)
    sequence3 = ScanMatchwithDuration.fixationToSequence(data3).astype(np.int32)

    # perform ScanMatch
    (score1, align1, f1) = ScanMatchwithDuration.match(sequence1, sequence2)

    (score2, align2, f2) = ScanMatchwithDuration.match(sequence1, sequence3)

    (score3, align3, f3) = ScanMatchwithDuration.match(sequence2, sequence3)

    # without
    sequence1_ = ScanMatchwithoutDuration.fixationToSequence(data1[:, :2]).astype(
        np.int32
    )
    sequence2_ = ScanMatchwithoutDuration.fixationToSequence(data2[:, :2]).astype(
        np.int32
    )
    sequence3_ = ScanMatchwithoutDuration.fixationToSequence(data3[:, :2]).astype(
        np.int32
    )

    # perform ScanMatch
    (score1_, align1_, f1_) = ScanMatchwithoutDuration.match(sequence1_, sequence2_)

    (score2_, align2_, f2_) = ScanMatchwithoutDuration.match(sequence1_, sequence3_)

    (score3_, align3_, f3_) = ScanMatchwithoutDuration.match(sequence3_, sequence3_)
