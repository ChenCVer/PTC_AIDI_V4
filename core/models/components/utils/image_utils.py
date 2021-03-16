import torch
import numpy as np


def slide_window_crop_tensor(input_tensor, crop_shape, overlap_hw=(0, 0)):
    if isinstance(crop_shape, int):
        crop_shape = (crop_shape, crop_shape)
    if isinstance(overlap_hw, int):
        overlap_hw = (overlap_hw, overlap_hw)

    row, col = input_tensor.shape[2:]
    maps = np.zeros(input_tensor.shape[2:], dtype=np.float32)
    if row <= crop_shape[0] and col <= crop_shape[1]:
        image = input_tensor
        return [image], [[0, 0]], np.ones(image.shape[2:], dtype=np.float32)

    piece_list = []
    pts_list = []
    overlap_h, overlap_w = overlap_hw
    stride_row = crop_shape[0] - overlap_h
    stride_col = crop_shape[1] - overlap_w

    for row_n in range(0, row - crop_shape[0] + stride_row, stride_row):
        for col_n in range(0, col - crop_shape[1] + stride_col, stride_col):

            row_start = row_n
            row_end = row_n + crop_shape[0]
            col_start = col_n
            col_end = col_n + crop_shape[1]
            if row_n + crop_shape[0] > row:
                row_start = row - crop_shape[0]
                row_end = row
            if col_n + crop_shape[1] > col:
                col_start = col - crop_shape[1]
                col_end = col

            piece = input_tensor[:, :, row_start:row_end, col_start:col_end]

            maps[row_start:row_end, col_start:col_end] += 1
            pts = [row_start, col_start]
            piece_list.append(piece)
            pts_list.append(pts)

    return piece_list, pts_list, maps


def overlap_crop_inv_mean(piece_list, pts_list, maps, src_shape):
    if len(piece_list) == 1:
        return piece_list[0]
    image = np.zeros(src_shape, dtype=np.float32)
    piece_size = piece_list[0].shape[0]

    for i in range(len(piece_list)):
        piece = piece_list[i]

        image[pts_list[i][0]:pts_list[i][0] + piece_size,
        pts_list[i][1]:pts_list[i][1] + piece_size] = \
            image[pts_list[i][0]:pts_list[i][0] + piece_size,
            pts_list[i][1]:pts_list[i][1] + piece_size] + piece

    if len(image.shape) == 2:
        return image / maps
    else:
        for i in range(image.shape[2]):
            image[:, :, i] = image[:, :, i] / maps
        return image


def overlap_crop_inv_max(piece_list, pts_list, maps, src_shape):
    if len(piece_list) == 1:
        return piece_list[0]
    cuda_device = piece_list[0].device
    whole_pred = torch.full(src_shape, -1e5).to(device=cuda_device)
    piece_size_x, piece_size_y = piece_list[0].shape[2:]

    for i, piece_tensor in enumerate(piece_list):
        whole_piece = whole_pred[:, :, pts_list[i][0]:pts_list[i][0] + piece_size_x,
                                       pts_list[i][1]:pts_list[i][1] + piece_size_y]

        whole_pred[:, :, pts_list[i][0]:pts_list[i][0] + piece_size_x,
                         pts_list[i][1]:pts_list[i][1] + piece_size_y] = torch.where(whole_piece > piece_tensor,
                                                                                     whole_piece, piece_tensor)

    return whole_pred