import json
import warnings
from os.path import join

import einops
import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage as filters
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

warnings.filterwarnings("ignore", category=DeprecationWarning)

class CTScanGaze(Dataset):
    """
    CT-ScanGaze dataset for 3D volumetric scanpath prediction training
    """

    def __init__(
        self,
        stimuli_dir,
        feature_dir,
        fixations_dir,
        action_map=(16, 16, 16),
        origin_size=(512, 512, 512),
        resize=(8, 8, 8),
        max_length=16,
        blur_sigma=1,
        type="train",
        transform=None,
    ):
        self.stimuli_dir = stimuli_dir
        self.feature_dir = feature_dir
        self.action_map = action_map
        self.origin_size = origin_size
        self.resize = resize
        self.max_length = max_length
        self.blur_sigma = blur_sigma
        self.type = type
        self.transform = transform
        self.PAD = [-3, -3, -3]

        self.resizescale_x = origin_size[1] / resize[1]
        self.resizescale_y = origin_size[0] / resize[0]
        self.resizescale_z = origin_size[2] / resize[2]

        self.downscale_x = resize[1] / action_map[1]
        self.downscale_y = resize[0] / action_map[0]
        self.downscale_z = resize[2] / action_map[2]

        self.fixations_file = join(
            self.stimuli_dir, "data_simplified_split_matched_pt.json"
        )
        with open(self.fixations_file) as json_file:
            fixations = json.load(json_file)
        fixations = [_ for _ in fixations if _["split"] == type]
        self.fixations = fixations

        self.imgid_to_sub = {}
        for index, fixation in enumerate(self.fixations):
            self.imgid_to_sub.setdefault(
                "{}/{}".format(fixation["task"], fixation["name"]), []
            ).append(index)
        self.imgid = list(self.imgid_to_sub.keys())

    def __len__(self):
        return len(self.imgid)

    def show_image(self, img):
        plt.figure()
        plt.imshow(img)
        plt.show()

    def __getitem__(self, idx):
        img_name = self.imgid[idx]
        img_path = join(self.feature_dir, img_name.replace("jpg", "pth"))
        image_ftrs = torch.load(img_path).unsqueeze(0)
        image_ftrs = F.interpolate(image_ftrs, size=(8, 8, 8))
        image_ftrs = einops.rearrange(image_ftrs, "b d h w c -> b (h w c) d")
        images = []
        subjects = []
        target_scanpaths = []
        durations = []
        action_masks = []
        duration_masks = []
        for ids in self.imgid_to_sub[img_name]:
            fixation = self.fixations[ids]

            scanpath = np.zeros(
                (
                    self.max_length,
                    self.action_map[0],
                    self.action_map[1],
                    self.action_map[2],
                ),
                dtype=np.float32,
            )
            # the first element denotes the termination action
            target_scanpath = np.zeros(
                (
                    self.max_length,
                    self.action_map[0] * self.action_map[1] * self.action_map[2] + 1,
                ),
                dtype=np.float32,
            )
            duration = np.zeros(self.max_length, dtype=np.float32)
            action_mask = np.zeros(self.max_length, dtype=np.float32)
            duration_mask = np.zeros(self.max_length, dtype=np.float32)
            
            # if use cvpr paper, we do not need to normalize
            pos_x = np.array(fixation["X"]).astype(np.float32)
            pos_y = np.array(fixation["Y"]).astype(np.float32)
            pos_z = np.array(fixation["Z"]).astype(np.float32)
            duration_raw = np.array(fixation["T"]).astype(np.float32)

            pos_x_discrete = np.zeros(self.max_length, dtype=np.int32) - 1
            pos_y_discrete = np.zeros(self.max_length, dtype=np.int32) - 1
            pos_z_discrete = np.zeros(self.max_length, dtype=np.int32) - 1
            for index in range(len(pos_x)):
                # only preserve the max_length ground-truth
                if index == self.max_length:
                    break
                # since pixel is start from 1 ~ max based on matlab
                pos_x_discrete[index] = ((pos_x[index] - 1) / self.downscale_x).astype(
                    np.int32
                )
                pos_y_discrete[index] = ((pos_y[index] - 1) / self.downscale_y).astype(
                    np.int32
                )
                pos_z_discrete[index] = ((pos_z[index] - 1) / self.downscale_z).astype(
                    np.int32
                )
                duration[index] = duration_raw[index]
                action_mask[index] = 1
                duration_mask[index] = 1
            if action_mask.sum() <= self.max_length - 1:
                action_mask[int(action_mask.sum())] = 1

            for index in range(self.max_length):
                if (
                    pos_x_discrete[index] == -1
                    or pos_y_discrete[index] == -1
                    or pos_z_discrete[index] == -1
                ):
                    target_scanpath[index, 0] = 1
                else:
                    scanpath[
                        index,
                        pos_y_discrete[index],
                        pos_x_discrete[index],
                        pos_z_discrete[index],
                    ] = 1
                    if self.blur_sigma:
                        scanpath[index] = filters.gaussian_filter(
                            scanpath[index], self.blur_sigma
                        )
                        scanpath[index] /= scanpath[index].sum()
                    target_scanpath[index, 1:] = scanpath[index].reshape(-1)

            images.append(image_ftrs)
            durations.append(duration)
            action_masks.append(action_mask)
            duration_masks.append(duration_mask)
            subjects.append(int(fixation["subject"]) - 1)
            target_scanpaths.append(target_scanpath)

        images = torch.cat(images)
        subjects = np.array(subjects)
        target_scanpaths = np.array(target_scanpaths)
        durations = np.array(durations)
        action_masks = np.array(action_masks)
        duration_masks = np.array(duration_masks)

        return {
            "image": images,
            "subject": subjects,
            "img_name": img_name,
            "duration": durations,
            "action_mask": action_masks,
            "duration_mask": duration_masks,
            "target_scanpath": target_scanpaths,
        }

    def collate_func(self, batch):
        img_batch = []
        subject_batch = []
        img_name_batch = []
        duration_batch = []
        action_mask_batch = []
        duration_mask_batch = []
        target_scanpath_batch = []

        for sample in batch:
            (
                tmp_img,
                tmp_subject,
                tmp_img_name,
                tmp_duration,
                tmp_action_mask,
                tmp_duration_mask,
                tmp_target_scanpath,
            ) = (
                sample["image"],
                sample["subject"],
                sample["img_name"],
                sample["duration"],
                sample["action_mask"],
                sample["duration_mask"],
                sample["target_scanpath"],
            )
            img_batch.append(tmp_img)
            subject_batch.append(tmp_subject)
            img_name_batch.append(tmp_img_name)
            duration_batch.append(tmp_duration)
            action_mask_batch.append(tmp_action_mask)
            duration_mask_batch.append(tmp_duration_mask)
            target_scanpath_batch.append(tmp_target_scanpath)

        data = dict()
        data["images"] = torch.cat(img_batch)
        data["subjects"] = np.concatenate(subject_batch)
        data["img_names"] = img_name_batch
        data["durations"] = np.concatenate(duration_batch)
        data["action_masks"] = np.concatenate(action_mask_batch)
        data["duration_masks"] = np.concatenate(duration_mask_batch)
        data["target_scanpaths"] = np.concatenate(target_scanpath_batch)

        data = {
            k: torch.from_numpy(v) if type(v) is np.ndarray else v
            for k, v in data.items()
        }  # Turn all ndarray to torch tensor
        data = {
            k: v.unsqueeze(0) if type(v) is torch.Tensor else v for k, v in data.items()
        }
        return data


class CTScanGaze_evaluation(Dataset):
    """
    CT-ScanGaze dataset for evaluation and testing
    """

    def __init__(
        self,
        stimuli_dir,
        feature_dir,
        fixations_dir,
        action_map=(30, 40),
        origin_size=(600, 800),
        resize=(240, 320),
        type="validation",
        transform=None,
    ):
        self.stimuli_dir = stimuli_dir
        self.feature_dir = feature_dir
        self.action_map = action_map
        self.origin_size = origin_size
        self.resize = resize
        self.type = type
        self.transform = transform

        self.resizescale_x = origin_size[1] / resize[1]
        self.resizescale_y = origin_size[0] / resize[0]
        self.resizescale_z = origin_size[2] / resize[2]

        self.downscale_x = resize[1] / action_map[1]
        self.downscale_y = resize[0] / action_map[0]
        self.downscale_z = resize[2] / action_map[2]

        self.fixations_file = join(
            self.stimuli_dir, "data_simplified_split_matched_pt.json"
        )
        with open(self.fixations_file) as json_file:
            fixations = json.load(json_file)
        fixations = [_ for _ in fixations if _["split"] == type]
        self.fixations = fixations

        self.imgid_to_sub = {}
        for index, fixation in enumerate(self.fixations):
            self.imgid_to_sub.setdefault(
                "{}/{}".format(fixation["task"], fixation["name"]), []
            ).append(index)
        self.imgid = list(self.imgid_to_sub.keys())

        objects = set([_.split("/")[0] for _ in self.imgid])

    def __len__(self):
        return len(self.imgid)

    def show_image(self, img):
        plt.figure()
        plt.imshow(img)
        plt.show()

    def __getitem__(self, idx):
        img_name = self.imgid[idx]
        img_path = join(self.feature_dir, img_name.replace("jpg", "pth"))
        image_ftrs = torch.load(img_path).unsqueeze(0)
        image_ftrs = F.interpolate(image_ftrs, size=(8, 8, 8))
        image_ftrs = einops.rearrange(image_ftrs, "b d h w c -> b (h w c) d")

        images = []
        fix_vectors = []
        subjects = []
        firstfixs = []
        for ids in self.imgid_to_sub[img_name]:
            fixation = self.fixations[ids]

            # if use cvpr paper, we do not need to normalize
            x_start = np.array(fixation["X"]).astype(np.float32)
            y_start = np.array(fixation["Y"]).astype(np.float32)
            z_start = np.array(fixation["Z"]).astype(np.float32)
            duration = np.array(fixation["T"]).astype(np.float32)

            length = fixation["length"]

            # in the middle of the image
            firstfix = np.array([self.resize[0] / 2, self.resize[1] / 2, 0], np.int64)

            fix_vector = []
            for order in range(length):
                fix_vector.append(
                    (x_start[order], y_start[order], z_start[order], duration[order])
                )
            fix_vector = np.array(
                fix_vector,
                dtype={
                    "names": ("start_x", "start_y", "start_z", "duration"),
                    "formats": ("f8", "f8", "f8", "f8"),
                },
            )

            fix_vectors.append(fix_vector)
            subjects.append(int(fixation["subject"]) - 1)
            firstfixs.append(firstfix)
            images.append(image_ftrs)

        images = torch.cat(images)
        return {
            "image": images,
            "fix_vectors": fix_vectors,
            "firstfix": firstfixs,
            "img_name": img_name,
            "subject": subjects,
        }

    def collate_func(self, batch):
        img_batch = []
        fix_vectors_batch = []
        firstfix_batch = []
        subject_batch = []
        img_name_batch = []

        for sample in batch:
            (
                tmp_img,
                tmp_fix_vectors,
                tmp_firstfix,
                tmp_subject,
                tmp_img_name,
            ) = (
                sample["image"],
                sample["fix_vectors"],
                sample["firstfix"],
                sample["subject"],
                sample["img_name"],
            )
            img_batch.append(tmp_img)
            fix_vectors_batch.append(tmp_fix_vectors)
            firstfix_batch.append(tmp_firstfix)
            subject_batch.append(tmp_subject)
            img_name_batch.append(tmp_img_name)

        data = {}
        data["images"] = torch.stack(img_batch)
        data["fix_vectors"] = fix_vectors_batch
        data["firstfixs"] = np.stack(firstfix_batch)
        data["subjects"] = np.array(subject_batch)
        data["img_names"] = img_name_batch

        data = {
            k: torch.from_numpy(v) if type(v) is np.ndarray else v
            for k, v in data.items()
        }  # Turn all ndarray to torch tensor

        return data
