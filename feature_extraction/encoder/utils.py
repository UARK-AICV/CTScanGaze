# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any, Iterable

import torch
import torch.nn.functional as F
from monai.data.meta_tensor import MetaTensor
from monai.data.utils import (
    compute_importance_map,
    dense_patch_slices,
    get_valid_patch_size,
)
from monai.utils import (
    BlendMode,
    PytorchPadMode,
    convert_data_type,
    convert_to_dst_type,
    ensure_tuple,
    ensure_tuple_rep,
    fall_back_tuple,
    look_up_option,
    optional_import,
    pytorch_after,
)

tqdm, _ = optional_import("tqdm", name="tqdm")
_nearest_mode = "nearest-exact" if pytorch_after(1, 11) else "nearest"

__all__ = [
    "sliding_window_encode_swin_only",
    "sliding_window_encode",
    "sliding_window_encode_pyramid",
    "sliding_window_encode_read_only_for_document",
]


def sliding_window_encode_pyramid(
    inputs: torch.Tensor | MetaTensor,
    roi_size: Sequence[int] | int,
    sw_batch_size: int,
    predictor: Callable[
        ..., torch.Tensor | Sequence[torch.Tensor] | dict[Any, torch.Tensor]
    ],
    overlap: Sequence[float] | float = 0.25,
    mode: BlendMode | str = BlendMode.CONSTANT,
    sigma_scale: Sequence[float] | float = 0.125,
    padding_mode: PytorchPadMode | str = PytorchPadMode.CONSTANT,
    cval: float = 0.0,
    sw_device: torch.device | str | None = None,
    device: torch.device | str | None = None,
) -> torch.Tensor | tuple[torch.Tensor, ...] | dict[Any, torch.Tensor]:
    buffered = False
    process_fn = None
    num_spatial_dims = len(inputs.shape) - 2
    overlap = ensure_tuple_rep(overlap, num_spatial_dims)
    for o in overlap:
        if o < 0 or o >= 1:
            raise ValueError(f"overlap must be >= 0 and < 1, got {overlap}.")
    compute_dtype = inputs.dtype

    # determine image spatial size and batch size
    # Note: all input images must have the same image size and batch size
    batch_size, _, *image_size_ = inputs.shape
    device = device or inputs.device
    sw_device = sw_device or inputs.device

    temp_meta = None
    if isinstance(inputs, MetaTensor):
        temp_meta = MetaTensor([]).copy_meta_from(inputs, copy_attr=False)
    inputs = convert_data_type(inputs, torch.Tensor, wrap_sequence=True)[0]
    # inputs.shape after torch.Size([1, 1, 333, 229, 224])
    roi_size = fall_back_tuple(roi_size, image_size_)
    # in case that image size is smaller than roi size
    image_size = tuple(
        max(image_size_[i], roi_size[i]) for i in range(num_spatial_dims)
    )
    pad_size = []
    for k in range(len(inputs.shape) - 1, 1, -1):
        diff = max(roi_size[k - 2] - inputs.shape[k], 0)
        half = diff // 2
        pad_size.extend([half, diff - half])
    if any(pad_size):
        inputs = F.pad(
            inputs,
            pad=pad_size,
            mode=look_up_option(padding_mode, PytorchPadMode),
            value=cval,
        )

    # Store all slices
    scan_interval = _get_scan_interval(image_size, roi_size, num_spatial_dims, overlap)
    slices = dense_patch_slices(
        image_size, roi_size, scan_interval, return_slice=not buffered
    )

    num_win = len(slices)  # number of windows per image
    total_slices = num_win * batch_size  # total number of windows
    windows_range: Iterable
    non_blocking = False
    windows_range = range(0, total_slices, sw_batch_size)

    # Create window-level importance map

    # stores output and count map
    output_image_list, count_map_list, sw_device_buffer = [], [], []  # type: ignore
    # for each patch
    saved_embeddings = []
    for slice_g in tqdm(windows_range):
        slice_range = range(
            slice_g,
            min(slice_g + sw_batch_size, total_slices),
        )
        unravel_slice = [
            [slice(idx // num_win, idx // num_win + 1), slice(None)]
            + list(slices[idx % num_win])
            for idx in slice_range
        ]
        # print(unravel_slice) # [[slice(0, 1, None), slice(None, None, None), slice(0, 96, None), slice(0, 96, None), slice(0, 96, None)], [slice(0, 1, None), slice(None, None, None), slice(0, 96, None), slice(0, 96, None), slice(72, 168, None)], [slice(0, 1, None), slice(None, None, None), slice(0, 96, None), slice(0, 96, None), slice(128, 224, None)], [slice(0, 1, None), slice(None, None, None), slice(0, 96, None), slice(72, 168, None), slice(0, 96, None)]] and increase
        # print(sw_batch_size) # 4
        if sw_batch_size > 1:
            win_data = torch.cat([inputs[win_slice] for win_slice in unravel_slice]).to(
                sw_device
            )
        else:
            win_data = inputs[unravel_slice[0]].to(sw_device)
            # print(unravel_slice[0])
        # print(win_data.shape) # torch.Size([4, 1, 96, 96, 96]) and torch.Size([1, 1, 96, 96, 96])
        dec1, dec4 = predictor.encode_pyramid(win_data)
        # for gaze former we only need the last layer of encoder or swin vit
        saved_embeddings.append(
            {
                "dec1": dec1.detach().cpu(),
                "dec4": dec4.detach().cpu(),
                "unravel_slice": unravel_slice,
            }
        )

    return saved_embeddings


def sliding_window_encode(
    inputs: torch.Tensor | MetaTensor,
    roi_size: Sequence[int] | int,
    sw_batch_size: int,
    predictor: Callable[
        ..., torch.Tensor | Sequence[torch.Tensor] | dict[Any, torch.Tensor]
    ],
    overlap: Sequence[float] | float = 0.25,
    mode: BlendMode | str = BlendMode.CONSTANT,
    sigma_scale: Sequence[float] | float = 0.125,
    padding_mode: PytorchPadMode | str = PytorchPadMode.CONSTANT,
    cval: float = 0.0,
    sw_device: torch.device | str | None = None,
    device: torch.device | str | None = None,
) -> torch.Tensor | tuple[torch.Tensor, ...] | dict[Any, torch.Tensor]:
    buffered = False
    process_fn = None
    num_spatial_dims = len(inputs.shape) - 2
    overlap = ensure_tuple_rep(overlap, num_spatial_dims)
    for o in overlap:
        if o < 0 or o >= 1:
            raise ValueError(f"overlap must be >= 0 and < 1, got {overlap}.")
    compute_dtype = inputs.dtype

    # determine image spatial size and batch size
    # Note: all input images must have the same image size and batch size
    batch_size, _, *image_size_ = inputs.shape
    device = device or inputs.device
    sw_device = sw_device or inputs.device

    temp_meta = None
    if isinstance(inputs, MetaTensor):
        temp_meta = MetaTensor([]).copy_meta_from(inputs, copy_attr=False)
    inputs = convert_data_type(inputs, torch.Tensor, wrap_sequence=True)[0]
    # inputs.shape after torch.Size([1, 1, 333, 229, 224])
    roi_size = fall_back_tuple(roi_size, image_size_)
    # in case that image size is smaller than roi size
    image_size = tuple(
        max(image_size_[i], roi_size[i]) for i in range(num_spatial_dims)
    )
    pad_size = []
    for k in range(len(inputs.shape) - 1, 1, -1):
        diff = max(roi_size[k - 2] - inputs.shape[k], 0)
        half = diff // 2
        pad_size.extend([half, diff - half])
    if any(pad_size):
        inputs = F.pad(
            inputs,
            pad=pad_size,
            mode=look_up_option(padding_mode, PytorchPadMode),
            value=cval,
        )

    # Store all slices
    scan_interval = _get_scan_interval(image_size, roi_size, num_spatial_dims, overlap)
    slices = dense_patch_slices(
        image_size, roi_size, scan_interval, return_slice=not buffered
    )

    num_win = len(slices)  # number of windows per image
    total_slices = num_win * batch_size  # total number of windows
    windows_range: Iterable
    non_blocking = False
    windows_range = range(0, total_slices, sw_batch_size)

    # Create window-level importance map

    # stores output and count map
    output_image_list, count_map_list, sw_device_buffer = [], [], []  # type: ignore
    # for each patch
    saved_embeddings = []
    for slice_g in tqdm(windows_range):
        slice_range = range(
            slice_g,
            min(slice_g + sw_batch_size, total_slices),
        )
        unravel_slice = [
            [slice(idx // num_win, idx // num_win + 1), slice(None)]
            + list(slices[idx % num_win])
            for idx in slice_range
        ]
        # print(unravel_slice) # [[slice(0, 1, None), slice(None, None, None), slice(0, 96, None), slice(0, 96, None), slice(0, 96, None)], [slice(0, 1, None), slice(None, None, None), slice(0, 96, None), slice(0, 96, None), slice(72, 168, None)], [slice(0, 1, None), slice(None, None, None), slice(0, 96, None), slice(0, 96, None), slice(128, 224, None)], [slice(0, 1, None), slice(None, None, None), slice(0, 96, None), slice(72, 168, None), slice(0, 96, None)]] and increase
        # print(sw_batch_size) # 4
        if sw_batch_size > 1:
            win_data = torch.cat([inputs[win_slice] for win_slice in unravel_slice]).to(
                sw_device
            )
        else:
            win_data = inputs[unravel_slice[0]].to(sw_device)
            # print(unravel_slice[0])
        # print(win_data.shape) # torch.Size([4, 1, 96, 96, 96]) and torch.Size([1, 1, 96, 96, 96])
        hidden_states_out, enc0, enc1, enc2, enc3, dec4 = predictor.encode(win_data)
        # saved_embeddings.append(
        #     {
        #         "win_data": win_data.detach().cpu(),
        #         "hidden_states_out": [x.detach().cpu() for x in hidden_states_out],
        #         "enc0": enc0.detach().cpu(),
        #         "enc1": enc1.detach().cpu(),
        #         "enc2": enc2.detach().cpu(),
        #         "enc3": enc3.detach().cpu(),
        #         "dec4": dec4.detach().cpu(),
        #         "unravel_slice": unravel_slice,
        #     }
        # )
        # for gaze former we only need the last layer of encoder or swin vit
        saved_embeddings.append(
            {
                "hidden_states_out": hidden_states_out[4].detach().cpu(),
                "dec4": dec4.detach().cpu(),
                "unravel_slice": unravel_slice,
            }
        )
        # print(hidden_states_out[0].shape)  # torch.Size([4, 48, 48, 48, 48])
        # print(hidden_states_out[1].shape)  # torch.Size([4, 96, 24, 24, 24])
        # print(hidden_states_out[2].shape)  # torch.Size([4, 192, 12, 12, 12])
        # print(hidden_states_out[3].shape)  # torch.Size([4, 384, 6, 6, 6])
        # print(hidden_states_out[4].shape)  # torch.Size([4, 768, 3, 3, 3])
        # print(enc0.shape)  # torch.Size([4, 48, 96, 96, 96])
        # print(enc1.shape)  # torch.Size([4, 48, 48, 48, 48])
        # print(enc2.shape)  # torch.Size([4, 96, 24, 24, 24])
        # print(enc3.shape)  # torch.Size([4, 192, 12, 12, 12])
        # print(dec4.shape)  # torch.Size([4, 768, 3, 3, 3])
        # print(seg_prob_out.shape) # torch.Size([4, 14, 96, 96, 96]) or torch.Size([1, 14, 96, 96, 96])
    return saved_embeddings


def sliding_window_encode_swin_only(
    inputs: torch.Tensor | MetaTensor,
    roi_size: Sequence[int] | int,
    sw_batch_size: int,
    predictor: Callable[
        ..., torch.Tensor | Sequence[torch.Tensor] | dict[Any, torch.Tensor]
    ],
    overlap: Sequence[float] | float = 0.25,
    mode: BlendMode | str = BlendMode.CONSTANT,
    sigma_scale: Sequence[float] | float = 0.125,
    padding_mode: PytorchPadMode | str = PytorchPadMode.CONSTANT,
    cval: float = 0.0,
    sw_device: torch.device | str | None = None,
    device: torch.device | str | None = None,
) -> torch.Tensor | tuple[torch.Tensor, ...] | dict[Any, torch.Tensor]:
    buffered = False
    process_fn = None
    num_spatial_dims = len(inputs.shape) - 2
    overlap = ensure_tuple_rep(overlap, num_spatial_dims)
    for o in overlap:
        if o < 0 or o >= 1:
            raise ValueError(f"overlap must be >= 0 and < 1, got {overlap}.")
    compute_dtype = inputs.dtype

    # determine image spatial size and batch size
    # Note: all input images must have the same image size and batch size
    batch_size, _, *image_size_ = inputs.shape
    device = device or inputs.device
    sw_device = sw_device or inputs.device

    temp_meta = None
    if isinstance(inputs, MetaTensor):
        temp_meta = MetaTensor([]).copy_meta_from(inputs, copy_attr=False)
    inputs = convert_data_type(inputs, torch.Tensor, wrap_sequence=True)[0]
    # inputs.shape after torch.Size([1, 1, 333, 229, 224])
    roi_size = fall_back_tuple(roi_size, image_size_)
    # in case that image size is smaller than roi size
    image_size = tuple(
        max(image_size_[i], roi_size[i]) for i in range(num_spatial_dims)
    )
    pad_size = []
    for k in range(len(inputs.shape) - 1, 1, -1):
        diff = max(roi_size[k - 2] - inputs.shape[k], 0)
        half = diff // 2
        pad_size.extend([half, diff - half])
    if any(pad_size):
        inputs = F.pad(
            inputs,
            pad=pad_size,
            mode=look_up_option(padding_mode, PytorchPadMode),
            value=cval,
        )

    # Store all slices
    scan_interval = _get_scan_interval(image_size, roi_size, num_spatial_dims, overlap)
    slices = dense_patch_slices(
        image_size, roi_size, scan_interval, return_slice=not buffered
    )

    num_win = len(slices)  # number of windows per image
    total_slices = num_win * batch_size  # total number of windows
    windows_range: Iterable
    non_blocking = False
    windows_range = range(0, total_slices, sw_batch_size)

    # Create window-level importance map

    # stores output and count map
    output_image_list, count_map_list, sw_device_buffer = [], [], []  # type: ignore
    # for each patch
    saved_embeddings = []
    for slice_g in tqdm(windows_range):
        slice_range = range(
            slice_g,
            min(slice_g + sw_batch_size, total_slices),
        )
        unravel_slice = [
            [slice(idx // num_win, idx // num_win + 1), slice(None)]
            + list(slices[idx % num_win])
            for idx in slice_range
        ]
        # print(unravel_slice) # [[slice(0, 1, None), slice(None, None, None), slice(0, 96, None), slice(0, 96, None), slice(0, 96, None)], [slice(0, 1, None), slice(None, None, None), slice(0, 96, None), slice(0, 96, None), slice(72, 168, None)], [slice(0, 1, None), slice(None, None, None), slice(0, 96, None), slice(0, 96, None), slice(128, 224, None)], [slice(0, 1, None), slice(None, None, None), slice(0, 96, None), slice(72, 168, None), slice(0, 96, None)]] and increase
        # print(sw_batch_size) # 4
        if sw_batch_size > 1:
            win_data = torch.cat([inputs[win_slice] for win_slice in unravel_slice]).to(
                sw_device
            )
        else:
            win_data = inputs[unravel_slice[0]].to(sw_device)
            # print(unravel_slice[0])
        # print(win_data.shape) # torch.Size([4, 1, 96, 96, 96]) and torch.Size([1, 1, 96, 96, 96])
        hidden_states_out = predictor.swin_only_encode(win_data)
        saved_embeddings.append(
            {
                "win_data": win_data.detach().cpu(),
                "hidden_states_out": [x.detach().cpu() for x in hidden_states_out],
                "unravel_slice": unravel_slice,
            }
        )
        # print(hidden_states_out[0].shape)  # torch.Size([4, 48, 48, 48, 48])
        # print(hidden_states_out[1].shape)  # torch.Size([4, 96, 24, 24, 24])
        # print(hidden_states_out[2].shape)  # torch.Size([4, 192, 12, 12, 12])
        # print(hidden_states_out[3].shape)  # torch.Size([4, 384, 6, 6, 6])
        # print(hidden_states_out[4].shape)  # torch.Size([4, 768, 3, 3, 3])
        # print(enc0.shape)  # torch.Size([4, 48, 96, 96, 96])
        # print(enc1.shape)  # torch.Size([4, 48, 48, 48, 48])
        # print(enc2.shape)  # torch.Size([4, 96, 24, 24, 24])
        # print(enc3.shape)  # torch.Size([4, 192, 12, 12, 12])
        # print(dec4.shape)  # torch.Size([4, 768, 3, 3, 3])
        # print(seg_prob_out.shape) # torch.Size([4, 14, 96, 96, 96]) or torch.Size([1, 14, 96, 96, 96])
    return saved_embeddings


def sliding_window_encode_read_only_for_document(
    inputs: torch.Tensor | MetaTensor,
    roi_size: Sequence[int] | int,
    sw_batch_size: int,
    predictor: Callable[
        ..., torch.Tensor | Sequence[torch.Tensor] | dict[Any, torch.Tensor]
    ],
    overlap: Sequence[float] | float = 0.25,
    mode: BlendMode | str = BlendMode.CONSTANT,
    sigma_scale: Sequence[float] | float = 0.125,
    padding_mode: PytorchPadMode | str = PytorchPadMode.CONSTANT,
    cval: float = 0.0,
    sw_device: torch.device | str | None = None,
    device: torch.device | str | None = None,
    progress: bool = False,
    roi_weight_map: torch.Tensor | None = None,
    *args: Any,
    **kwargs: Any,
) -> torch.Tensor | tuple[torch.Tensor, ...] | dict[Any, torch.Tensor]:
    buffered = False
    process_fn = None
    num_spatial_dims = len(inputs.shape) - 2
    overlap = ensure_tuple_rep(overlap, num_spatial_dims)
    for o in overlap:
        if o < 0 or o >= 1:
            raise ValueError(f"overlap must be >= 0 and < 1, got {overlap}.")
    compute_dtype = inputs.dtype

    # determine image spatial size and batch size
    # Note: all input images must have the same image size and batch size
    batch_size, _, *image_size_ = inputs.shape
    device = device or inputs.device
    sw_device = sw_device or inputs.device

    temp_meta = None
    if isinstance(inputs, MetaTensor):
        temp_meta = MetaTensor([]).copy_meta_from(inputs, copy_attr=False)
    inputs = convert_data_type(inputs, torch.Tensor, wrap_sequence=True)[0]
    # inputs.shape after torch.Size([1, 1, 333, 229, 224])
    roi_size = fall_back_tuple(roi_size, image_size_)
    # in case that image size is smaller than roi size
    image_size = tuple(
        max(image_size_[i], roi_size[i]) for i in range(num_spatial_dims)
    )
    pad_size = []
    for k in range(len(inputs.shape) - 1, 1, -1):
        diff = max(roi_size[k - 2] - inputs.shape[k], 0)
        half = diff // 2
        pad_size.extend([half, diff - half])
    if any(pad_size):
        inputs = F.pad(
            inputs,
            pad=pad_size,
            mode=look_up_option(padding_mode, PytorchPadMode),
            value=cval,
        )

    # Store all slices
    scan_interval = _get_scan_interval(image_size, roi_size, num_spatial_dims, overlap)
    slices = dense_patch_slices(
        image_size, roi_size, scan_interval, return_slice=not buffered
    )

    num_win = len(slices)  # number of windows per image
    total_slices = num_win * batch_size  # total number of windows
    windows_range: Iterable
    non_blocking = False
    windows_range = range(0, total_slices, sw_batch_size)

    # Create window-level importance map
    valid_patch_size = get_valid_patch_size(image_size, roi_size)

    try:
        valid_p_size = ensure_tuple(valid_patch_size)
        importance_map_ = compute_importance_map(
            valid_p_size,
            mode=mode,
            sigma_scale=sigma_scale,
            device=sw_device,
            dtype=compute_dtype,
        )
        if len(importance_map_.shape) == num_spatial_dims and not process_fn:
            importance_map_ = importance_map_[
                None, None
            ]  # adds batch, channel dimensions
    except Exception as e:
        raise RuntimeError(
            f"patch size {valid_p_size}, mode={mode}, sigma_scale={sigma_scale}, device={device}\n"
            "Seems to be OOM. Please try smaller patch size or mode='constant' instead of mode='gaussian'."
        ) from e
    importance_map_ = convert_data_type(
        importance_map_, torch.Tensor, device=sw_device, dtype=compute_dtype
    )[0]

    # stores output and count map
    output_image_list, count_map_list, sw_device_buffer = [], [], []  # type: ignore
    # for each patch
    # because they calculate each patch for a whole batch for parallel, so their only option is to put the for loop of batch inside the for loop of patch for convenience.
    for slice_g in tqdm(windows_range):
        slice_range = range(
            slice_g,
            min(slice_g + sw_batch_size, total_slices),
        )
        unravel_slice = [
            [slice(idx // num_win, idx // num_win + 1), slice(None)]
            + list(slices[idx % num_win])
            for idx in slice_range
        ]
        if sw_batch_size > 1:
            win_data = torch.cat([inputs[win_slice] for win_slice in unravel_slice]).to(
                sw_device
            )
        else:
            win_data = inputs[unravel_slice[0]].to(sw_device)
        seg_prob_out = predictor(win_data)  # batched patch
        # print(seg_prob_out.shape) # torch.Size([4, 14, 96, 96, 96]) or torch.Size([1, 14, 96, 96, 96])
        # convert seg_prob_out to tuple seg_tuple, this does not allocate new memory.
        dict_keys, seg_tuple = _flatten_struct(seg_prob_out)
        w_t = importance_map_
        if len(w_t.shape) == num_spatial_dims:
            w_t = w_t[None, None]
        w_t = w_t.to(dtype=compute_dtype, device=sw_device)
        sw_device_buffer = list(seg_tuple)
        # print(len(sw_device_buffer)) # 1 even for 4,14,...
        for ss in range(len(sw_device_buffer)):
            b_shape = sw_device_buffer[ss].shape
            # print(b_shape) # torch.Size([4, 14, 96, 96, 96]) or torch.Size([1, 14, 96, 96, 96])
            seg_chns, seg_shape = b_shape[1], b_shape[2:]
            # print(seg_chns, seg_shape) # 14 torch.Size([96, 96, 96]). 14 is for background + 13 classes
            z_scale = None
            # print(z_scale) # None
            # print(w_t.shape) # torch.Size([1, 1, 96, 96, 96])
            # print(len(output_image_list))  # 0, to 1, 1,1, ... no increase
            if len(output_image_list) <= ss:
                # print("in here how many")  # this prints only once
                output_shape = [batch_size, seg_chns]
                # print(output_shape) # [1,14]
                output_shape += list(image_size)
                # print(list(image_size))  # [333, 229, 224]
                # print(output_shape) # [1, 14, 333, 229, 224]
                # allocate memory to store the full output and the count for overlapping parts
                output_image_list.append(
                    torch.zeros(output_shape, dtype=compute_dtype, device=device)
                )
                # print(output_image_list[0].shape) # torch.Size([1, 14, 333, 229, 224])
                # print([1, 1] + output_shape[2:]) # [1, 1, 333, 229, 224]
                count_map_list.append(
                    torch.zeros(
                        [1, 1] + output_shape[2:], dtype=compute_dtype, device=device
                    )
                )
                w_t_ = w_t.to(device)
                # print(slices) # [(slice(0, 96, None), slice(0, 96, None), slice(0, 96, None)), (slice(0, 96, None), slice(0, 96, None), slice(72, 168, None)), (slice(0, 96, None), slice(0, 96, None), slice(128, 224, None)), (slice(0, 96, None), slice(72, 168, None), slice(0, 96, None)), (slice(0, 96, None), slice(72, 168, None), slice(72, 168, None)), (slice(0, 96, None), slice(72, 168, None), slice(128, 224, None)), (slice(0, 96, None), slice(133, 229, None), slice(0, 96, None)), (slice(0, 96, None), slice(133, 229, None), slice(72, 168, None)), (slice(0, 96, None), slice(133, 229, None), slice(128, 224, None)), (slice(72, 168, None), slice(0, 96, None), slice(0, 96, None)), (slice(72, 168, None), slice(0, 96, None), slice(72, 168, None)), (slice(72, 168, None), slice(0, 96, None), slice(128, 224, None)), (slice(72, 168, None), slice(72, 168, None), slice(0, 96, None)), (slice(72, 168, None), slice(72, 168, None), slice(72, 168, None)), (slice(72, 168, None), slice(72, 168, None), slice(128, 224, None)), (slice(72, 168, None), slice(133, 229, None), slice(0, 96, None)), (slice(72, 168, None), slice(133, 229, None), slice(72, 168, None)), (slice(72, 168, None), slice(133, 229, None), slice(128, 224, None)), (slice(144, 240, None), slice(0, 96, None), slice(0, 96, None)), (slice(144, 240, None), slice(0, 96, None), slice(72, 168, None)), (slice(144, 240, None), slice(0, 96, None), slice(128, 224, None)), (slice(144, 240, None), slice(72, 168, None), slice(0, 96, None)), (slice(144, 240, None), slice(72, 168, None), slice(72, 168, None)), (slice(144, 240, None), slice(72, 168, None), slice(128, 224, None)), (slice(144, 240, None), slice(133, 229, None), slice(0, 96, None)), (slice(144, 240, None), slice(133, 229, None), slice(72, 168, None)), (slice(144, 240, None), slice(133, 229, None), slice(128, 224, None)), (slice(216, 312, None), slice(0, 96, None), slice(0, 96, None)), (slice(216, 312, None), slice(0, 96, None), slice(72, 168, None)), (slice(216, 312, None), slice(0, 96, None), slice(128, 224, None)), (slice(216, 312, None), slice(72, 168, None), slice(0, 96, None)), (slice(216, 312, None), slice(72, 168, None), slice(72, 168, None)), (slice(216, 312, None), slice(72, 168, None), slice(128, 224, None)), (slice(216, 312, None), slice(133, 229, None), slice(0, 96, None)), (slice(216, 312, None), slice(133, 229, None), slice(72, 168, None)), (slice(216, 312, None), slice(133, 229, None), slice(128, 224, None)), (slice(237, 333, None), slice(0, 96, None), slice(0, 96, None)), (slice(237, 333, None), slice(0, 96, None), slice(72, 168, None)), (slice(237, 333, None), slice(0, 96, None), slice(128, 224, None)), (slice(237, 333, None), slice(72, 168, None), slice(0, 96, None)), (slice(237, 333, None), slice(72, 168, None), slice(72, 168, None)), (slice(237, 333, None), slice(72, 168, None), slice(128, 224, None)), (slice(237, 333, None), slice(133, 229, None), slice(0, 96, None)), (slice(237, 333, None), slice(133, 229, None), slice(72, 168, None)), (slice(237, 333, None), slice(133, 229, None), slice(128, 224, None))]

                for __s in slices:
                    count_map_list[-1][(slice(None), slice(None), *__s)] += w_t_

            sw_device_buffer[ss] *= w_t
            sw_device_buffer[ss] = sw_device_buffer[ss].to(device)
            _compute_coords(
                unravel_slice, z_scale, output_image_list[ss], sw_device_buffer[ss]
            )
            # idea is add the patch results to the final output tensor until we have all the patches. The list output_image_list has one item because our batch only has 1 item (so ss = 0 to 1). You still need to for loop all slices to add to coresponding patch.
            # so to save it, what we need to keep is unravel_slice and sw_device_buffer
        sw_device_buffer = []

    # account for any overlapping sections
    for ss in range(len(output_image_list)):
        output_image_list[ss] /= count_map_list.pop(0)
    # print(output_image_list[0].shape) # torch.Size([1, 14, 333, 229, 224])

    final_output = _pack_struct(output_image_list, dict_keys)
    if temp_meta is not None:
        final_output = convert_to_dst_type(final_output, temp_meta, device=device)[0]
    else:
        final_output = convert_to_dst_type(final_output, inputs, device=device)[0]

    return final_output  # type: ignore


def _compute_coords(coords, z_scale, out, patch):
    """sliding window batch spatial scaling indexing for multi-resolution outputs."""
    for original_idx, p in zip(coords, patch):
        idx_zm = list(original_idx)  # 4D for 2D image, 5D for 3D image
        # print(idx_zm) # example [slice(0, 1, None), slice(None, None, None), slice(237, 333, None), slice(133, 229, None), slice(128, 224, None)]
        # print(p.shape) # torch.Size([14, 96, 96, 96])
        out[idx_zm] += p


def _get_scan_interval(
    image_size: Sequence[int],
    roi_size: Sequence[int],
    num_spatial_dims: int,
    overlap: Sequence[float],
) -> tuple[int, ...]:
    """
    Compute scan interval according to the image size, roi size and overlap.
    Scan interval will be `int((1 - overlap) * roi_size)`, if interval is 0,
    use 1 instead to make sure sliding window works.

    """
    if len(image_size) != num_spatial_dims:
        raise ValueError(
            f"len(image_size) {len(image_size)} different from spatial dims {num_spatial_dims}."
        )
    if len(roi_size) != num_spatial_dims:
        raise ValueError(
            f"len(roi_size) {len(roi_size)} different from spatial dims {num_spatial_dims}."
        )

    scan_interval = []
    for i, o in zip(range(num_spatial_dims), overlap):
        if roi_size[i] == image_size[i]:
            scan_interval.append(int(roi_size[i]))
        else:
            interval = int(roi_size[i] * (1 - o))
            scan_interval.append(interval if interval > 0 else 1)
    return tuple(scan_interval)


def _flatten_struct(seg_out):
    dict_keys = None
    seg_probs: tuple[torch.Tensor, ...]
    if isinstance(seg_out, torch.Tensor):
        seg_probs = (seg_out,)
    elif isinstance(seg_out, Mapping):
        dict_keys = sorted(seg_out.keys())  # track predictor's output keys
        seg_probs = tuple(seg_out[k] for k in dict_keys)
    else:
        seg_probs = ensure_tuple(seg_out)
    return dict_keys, seg_probs


def _pack_struct(seg_out, dict_keys=None):
    if dict_keys is not None:
        return dict(zip(dict_keys, seg_out))
    if isinstance(seg_out, (list, tuple)) and len(seg_out) == 1:
        return seg_out[0]
    return ensure_tuple(seg_out)
