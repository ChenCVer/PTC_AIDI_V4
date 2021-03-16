import os
import cv2
import numpy as np


class SlideWindowCrop(object):

    def __init__(self, crop_shape, resize_shape,
                 overlap_h=0, overlap_w=0, normlize=False):

        assert isinstance(crop_shape, tuple), "crop_shape must be tuple"
        assert isinstance(resize_shape, tuple), "resize_shape must be tuple"
        assert overlap_h < crop_shape[0]
        assert overlap_w < crop_shape[1]
        self.crop_shape = crop_shape
        self.resize_shape = resize_shape
        self.overlap_h = overlap_h
        self.overlap_w = overlap_w
        self.normlize = normlize

    def __call__(self, inputs):

        row, col = inputs.shape[:2]
        map = np.zeros(inputs.shape[:2], dtype=np.float32)

        if row <= self.crop_shape[0] and col <= self.crop_shape[1]:
            if self.normlize:
                image = np.array(inputs, np.float32) / 255.0
            else:
                image = inputs
            return [image], [[0, 0]], np.ones(image.shape[:2], dtype=np.float32)

        piece_list = []
        pts_list = []
        stride_row = self.crop_shape[0] - self.overlap_h
        stride_col = self.crop_shape[1] - self.overlap_w

        for row_n in range(0, row - self.crop_shape[0] + stride_row, stride_row):
            for col_n in range(0, col - self.crop_shape[1] + stride_col, stride_col):

                row_start = row_n
                row_end = row_n + self.crop_shape[0]
                col_start = col_n
                col_end = col_n + self.crop_shape[1]
                if row_n + self.crop_shape[0] > row:
                    row_start = row - self.crop_shape[0]
                    row_end = row
                if col_n + self.crop_shape[1] > col:
                    col_start = col - self.crop_shape[1]
                    col_end = col

                piece = inputs[row_start:row_end, col_start:col_end]
                # resize
                piece = cv2.resize(piece, self.resize_shape, interpolation=1)

                map[row_start:row_end, col_start:col_end] += 1
                pts = [row_start, col_start]
                if self.normlize:
                    piece = np.array(piece, np.float32) / 255.0
                piece_list.append(piece)
                pts_list.append(pts)

        return piece_list, pts_list, map


if __name__ == '__main__':
    data_root = "/home/cxj/Desktop/data/electronic_datasets/第三次标注数据/mask_pow"
    save_root = "/home/cxj/Desktop/data/electronic_datasets/第三次标注数据/slide_512_no_resize"
    img_names = os.listdir(data_root)
    crop_img = SlideWindowCrop((512, 512), (512, 512), 64, 64)
    for idx, img_name in enumerate(img_names):
        if img_name.endswith("_mask.png"):
            continue
        print("img_name = ", img_name)
        img_endswith = img_name.split(".")[1]
        img_path = os.path.join(data_root, img_name)
        mask_path = os.path.join(data_root, img_name[:-4] + "_mask.png")
        img = cv2.imread(img_path)
        try:
            mask = cv2.imread(mask_path)
        except:
            mask = np.zeros_like(img)
        # rotate
        rows, cols = img.shape[:2]
        angle = np.random.randint(0, 360)
        M = cv2.getRotationMatrix2D((rows / 2, cols / 2), angle, 1.)
        img = cv2.warpAffine(img, M, (rows, cols), borderValue=(114, 114, 114))
        mask = cv2.warpAffine(mask, M, (rows, cols), borderValue=(0, 0, 0))
        # crop img
        img_list, _, _ = crop_img(img)
        mask_list, _, _ = crop_img(mask)
        for jdx, img_piece in enumerate(img_list):
            piece_img_name = img_name[:-4] + "_" + str(idx) + "_" + str(jdx) + "." + img_endswith
            piece_mask_name = img_name[:-4] + "_" + str(idx) + "_" + str(jdx) + "_mask.png"
            save_path = os.path.join(save_root, piece_img_name)
            save_mask_path = os.path.join(save_root, piece_mask_name)
            if np.sum(mask_list[jdx]) != 0:
                cv2.imwrite(save_path, img_piece)
                cv2.imwrite(save_mask_path, mask_list[jdx])