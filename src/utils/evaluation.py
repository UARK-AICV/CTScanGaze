import copy

import numpy as np
import utils.evaltools.multimatch_3dgaze as multimatch
from tqdm import tqdm
from utils.evaltools.scanmatch3d import ScanMatch
from utils.evaltools.visual_attention_metrics import string_edit_distance


def comprehensive_evaluation_by_subject(
    gt_fix_vectors, predict_fix_vectors, args, is_eliminating_nan=True
):
    # Determine which metrics to compute
    eval_metrics = getattr(args, 'eval_metrics', 'all')
    compute_multimatch = eval_metrics in ['all', 'multimatch']
    compute_scanmatch = eval_metrics in ['all', 'scanmatch']
    compute_sed = eval_metrics in ['all', 'sed']

    # row: user prediction
    # col: gt prediction
    collect_scanmatch_with_duration_rlts = (
        np.zeros((len(predict_fix_vectors), args.subject_num, args.subject_num)) - 1
    )
    collect_scanmatch_without_duration_rlts = (
        np.zeros((len(predict_fix_vectors), args.subject_num, args.subject_num)) - 1
    )
    collect_SED_rlts = (
        np.zeros((len(predict_fix_vectors), args.subject_num, args.subject_num)) - 1
    )

    collect_multimatch_diag_rlts = (
        np.zeros((len(predict_fix_vectors), args.subject_num, 5)) - 1
    )
    collect_scanmatch_with_duration_diag_rlts = (
        np.zeros((len(predict_fix_vectors), args.subject_num)) - 1
    )
    collect_scanmatch_without_duration_diag_rlts = (
        np.zeros((len(predict_fix_vectors), args.subject_num)) - 1
    )
    collect_SED_diag_rlts = np.zeros((len(predict_fix_vectors), args.subject_num)) - 1

    scores_of_each_images = (
        np.zeros((len(predict_fix_vectors), args.subject_num, args.subject_num, 9)) - 1
    )
    args.depth = 512
    # create a ScanMatch object (only if needed)
    if compute_scanmatch:
        ScanMatchwithDuration = ScanMatch(
            Xres=args.width,
            Yres=args.height,
            Zres=args.depth,
            Xbin=6,
            Ybin=4,
            Zbin=8,
            Offset=(0, 0, 0),
            TempBin=25,
            Threshold=3.5,
        )
        ScanMatchwithoutDuration = ScanMatch(
            Xres=args.width,
            Yres=args.height,
            Zres=args.depth,
            Xbin=6,
            Ybin=4,
            Zbin=8,
            Offset=(0, 0, 0),
            Threshold=3.5,
        )

    stimulus = np.zeros((args.height, args.width, args.depth), dtype=np.float32)

    with tqdm(total=len(gt_fix_vectors)) as pbar:
        for index in range(len(gt_fix_vectors)):
            gt_fix_vector = gt_fix_vectors[index]
            predict_fix_vector = predict_fix_vectors[index]
            for row_idx in range(len(predict_fix_vector)):
                for col_idx in range(len(gt_fix_vector)):
                    # print("1")
                    inner_gt_fix_vector = gt_fix_vector[col_idx]
                    inner_predict_fix_vector = predict_fix_vector[row_idx]

                    # Initialize scores list
                    scores_of_given_image_with_gt = []

                    # calculate multimatch (only if requested)
                    if compute_multimatch:
                        mm_inner_gt_fix_vector = inner_gt_fix_vector.copy()
                        mm_inner_predict_fix_vector = inner_predict_fix_vector.copy()

                        if len(mm_inner_gt_fix_vector) < 3:
                            padding_vector = []
                            for _ in range(3 - len(mm_inner_gt_fix_vector)):
                                padding_vector.append((1.0, 1.0, 1.0, 1e-3))
                            padding_vector = np.array(
                                padding_vector,
                                dtype={
                                    "names": ("start_x", "start_y", "start_z", "duration"),
                                    "formats": ("f8", "f8", "f8", "f8"),
                                },
                            )
                            mm_inner_gt_fix_vector = np.concatenate(
                                [mm_inner_gt_fix_vector, padding_vector]
                            )
                        if len(mm_inner_predict_fix_vector) < 3:
                            padding_vector = []
                            for _ in range(3 - len(mm_inner_predict_fix_vector)):
                                padding_vector.append((1.0, 1.0, 1.0, 1e-3))
                            padding_vector = np.array(
                                padding_vector,
                                dtype={
                                    "names": ("start_x", "start_y", "start_z", "duration"),
                                    "formats": ("f8", "f8", "f8", "f8"),
                                },
                            )
                            mm_inner_predict_fix_vector = np.concatenate(
                                [mm_inner_predict_fix_vector, padding_vector]
                            )
                        # penalty if the length is shorter than GT
                        if len(mm_inner_predict_fix_vector) < len(mm_inner_gt_fix_vector):
                            padding_vector = []
                            for _ in range(
                                len(mm_inner_gt_fix_vector)
                                - len(mm_inner_predict_fix_vector)
                            ):
                                padding_vector.append((1.0, 1.0, 1.0, 1e-3))
                            padding_vector = np.array(
                                padding_vector,
                                dtype={
                                    "names": ("start_x", "start_y", "start_z", "duration"),
                                    "formats": ("f8", "f8", "f8", "f8"),
                                },
                            )
                            mm_inner_predict_fix_vector = np.concatenate(
                                [mm_inner_predict_fix_vector, padding_vector]
                            )

                        # # print(mm_inner_gt_fix_vector.shape)  # (length,)
                        # # print(mm_inner_predict_fix_vector.shape)  # (length,)

                        # print("2")
                        rlt = multimatch.docomparison(
                            mm_inner_gt_fix_vector,
                            mm_inner_predict_fix_vector,
                            screensize=[args.width, args.height, args.depth],
                        )
                        # # print(np.array(rlt).shape)  # (5,) = five scores of mm
                        # # print(rlt)
                        # print("3")
                        if row_idx == col_idx:
                            collect_multimatch_diag_rlts[index, row_idx] = np.array(rlt)
                        scores_of_given_image_with_gt = list(copy.deepcopy(rlt))
                    else:
                        # Add dummy values if not computing multimatch
                        scores_of_given_image_with_gt = [0.0] * 5
                    # print("3.1")
                    # perform scanmatch (only if requested)
                    if compute_scanmatch:
                        # we need to transform the scale of time from s to ms
                        # with duration
                        np_fix_vector_1 = np.array(
                            [list(_) for _ in list(inner_gt_fix_vector)]
                        )
                        np_fix_vector_2 = np.array(
                            [list(_) for _ in list(inner_predict_fix_vector)]
                        )
                        np_fix_vector_1[:, -1] *= 1000
                        np_fix_vector_2[:, -1] *= 1000
                        # # print(np_fix_vector_1.shape) # (length, 4) if xyzt, else (length, 3) if xyt
                        # # print(np_fix_vector_2.shape) # (length, 4) if xyzt, else (length, 3) if xyt
                        # print("3.2")
                        sequence1_wd = ScanMatchwithDuration.fixationToSequence(
                            np_fix_vector_1
                        ).astype(np.int32)
                        sequence2_wd = ScanMatchwithDuration.fixationToSequence(
                            np_fix_vector_2
                        ).astype(np.int32)
                        # # print(sequence1_wd) # dynamic based on occurance as well.
                        # print("3.3")
                        (score, align, f) = ScanMatchwithDuration.match(
                            sequence1_wd, sequence2_wd
                        )
                        # print("3.4")
                        collect_scanmatch_with_duration_rlts[index, row_idx, col_idx] = (
                            score
                        )
                        if row_idx == col_idx:
                            collect_scanmatch_with_duration_diag_rlts[index, row_idx] = (
                                score
                            )
                        scores_of_given_image_with_gt.append(score)
                        # print("4")
                        # without duration
                        sequence1_wod = ScanMatchwithoutDuration.fixationToSequence(
                            np_fix_vector_1
                        ).astype(np.int32)
                        sequence2_wod = ScanMatchwithoutDuration.fixationToSequence(
                            np_fix_vector_2
                        ).astype(np.int32)
                        (score, align, f) = ScanMatchwithoutDuration.match(
                            sequence1_wod, sequence2_wod
                        )
                        collect_scanmatch_without_duration_rlts[index, row_idx, col_idx] = (
                            score
                        )
                        if row_idx == col_idx:
                            collect_scanmatch_without_duration_diag_rlts[index, row_idx] = (
                                score
                            )
                        scores_of_given_image_with_gt.append(score)
                    else:
                        # Add dummy values if not computing scanmatch
                        scores_of_given_image_with_gt.extend([0.0, 0.0])
                        # Prepare vectors for potential SED calculation
                        np_fix_vector_1 = np.array(
                            [list(_) for _ in list(inner_gt_fix_vector)]
                        )
                        np_fix_vector_2 = np.array(
                            [list(_) for _ in list(inner_predict_fix_vector)]
                        )
                        np_fix_vector_1[:, -1] *= 1000
                        np_fix_vector_2[:, -1] *= 1000
                    # print("5")
                    # perfrom SED (only if requested)
                    if compute_sed:
                        sed = string_edit_distance(
                            stimulus, np_fix_vector_1, np_fix_vector_2
                        )
                        collect_SED_rlts[index, row_idx, col_idx] = sed
                        if row_idx == col_idx:
                            collect_SED_diag_rlts[index, row_idx] = sed
                        scores_of_given_image_with_gt.append(sed)
                    else:
                        scores_of_given_image_with_gt.append(0.0)

                    # print("6")
                    # perfrom STDE
                    stde = 0.0
                    # stde = scaled_time_delay_embedding_similarity(
                    #     np_fix_vector_1, np_fix_vector_2, stimulus
                    # )
                    # collect_STDE_rlts[index, row_idx, col_idx] = stde
                    # if row_idx == col_idx:
                    #     collect_STDE_diag_rlts[index, row_idx] = stde
                    scores_of_given_image_with_gt.append(stde)

                    scores_of_each_images[index, row_idx, col_idx] = (
                        scores_of_given_image_with_gt
                    )
                    # print("7")

            pbar.update(1)

    # the diag is for easy compute mean and std of multimatch for report.
    cur_metrics = dict()
    cur_metrics_std = dict()

    if compute_multimatch:
        collect_multimatch_diag_rlts = np.array(collect_multimatch_diag_rlts)
        collect_multimatch_diag_rlts = collect_multimatch_diag_rlts.reshape(-1, 5)
        collect_multimatch_diag_rlts = collect_multimatch_diag_rlts[
            (collect_multimatch_diag_rlts == -1).sum(-1) == 0
        ]
        if is_eliminating_nan:
            collect_multimatch_diag_rlts = collect_multimatch_diag_rlts[
                np.isnan(collect_multimatch_diag_rlts.sum(axis=1)) == False
            ]
        multimatch_metric_mean = np.mean(collect_multimatch_diag_rlts, axis=0)
        multimatch_metric_std = np.std(collect_multimatch_diag_rlts, axis=0)

        multimatch_cur_metrics = dict()
        multimatch_cur_metrics["vector"] = multimatch_metric_mean[0]
        multimatch_cur_metrics["direction"] = multimatch_metric_mean[1]
        multimatch_cur_metrics["length"] = multimatch_metric_mean[2]
        multimatch_cur_metrics["position"] = multimatch_metric_mean[3]
        multimatch_cur_metrics["duration"] = multimatch_metric_mean[4]
        multimatch_cur_metrics["mean"] = np.mean(multimatch_metric_mean)
        cur_metrics["MultiMatch"] = multimatch_cur_metrics

        multimatch_cur_metrics_std = dict()
        multimatch_cur_metrics_std["vector"] = multimatch_metric_std[0]
        multimatch_cur_metrics_std["direction"] = multimatch_metric_std[1]
        multimatch_cur_metrics_std["length"] = multimatch_metric_std[2]
        multimatch_cur_metrics_std["position"] = multimatch_metric_std[3]
        multimatch_cur_metrics_std["duration"] = multimatch_metric_std[4]
        cur_metrics_std["MultiMatch"] = multimatch_cur_metrics_std

    if compute_scanmatch:
        collect_scanmatch_with_duration_diag_rlts = (
            collect_scanmatch_with_duration_diag_rlts[
                collect_scanmatch_with_duration_diag_rlts != -1
            ]
        )
        collect_scanmatch_without_duration_diag_rlts = (
            collect_scanmatch_without_duration_diag_rlts[
                collect_scanmatch_without_duration_diag_rlts != -1
            ]
        )
        scanmatch_with_duration_metric_mean = np.mean(
            collect_scanmatch_with_duration_diag_rlts
        )
        scanmatch_with_duration_metric_std = np.std(
            collect_scanmatch_with_duration_diag_rlts
        )
        scanmatch_without_duration_metric_mean = np.mean(
            collect_scanmatch_without_duration_diag_rlts
        )
        scanmatch_without_duration_metric_std = np.std(
            collect_scanmatch_without_duration_diag_rlts
        )

        scanmatch_cur_metrics = dict()
        scanmatch_cur_metrics["w/o duration"] = scanmatch_without_duration_metric_mean
        scanmatch_cur_metrics["with duration"] = scanmatch_with_duration_metric_mean
        cur_metrics["ScanMatch"] = scanmatch_cur_metrics

        scanmatch_cur_metrics_std = dict()
        scanmatch_cur_metrics_std["w/o duration"] = scanmatch_without_duration_metric_std
        scanmatch_cur_metrics_std["with duration"] = scanmatch_with_duration_metric_std
        cur_metrics_std["ScanMatch"] = scanmatch_cur_metrics_std

    if compute_sed:
        SED_metrics_rlts = np.array(collect_SED_diag_rlts)
        SED_metrics_rlts = SED_metrics_rlts.reshape(-1, len(gt_fix_vector))

        SED_metrics_rlts = SED_metrics_rlts[SED_metrics_rlts != -1]
        SED_metrics_mean = SED_metrics_rlts.mean()
        SED_metrics_std = SED_metrics_rlts.std()

        VAME_cur_metrics = dict()
        VAME_cur_metrics["SED"] = SED_metrics_mean
        cur_metrics["VAME"] = VAME_cur_metrics

        VAME_cur_metrics_std = dict()
        VAME_cur_metrics_std["SED"] = SED_metrics_std
        cur_metrics_std["VAME"] = VAME_cur_metrics_std

    scores_of_each_images = scores_of_each_images.tolist()

    return cur_metrics, cur_metrics_std, scores_of_each_images
