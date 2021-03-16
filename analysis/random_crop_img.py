import os
import random
import cv2
import numpy as np

# 每张图片被裁剪的次数s
crop_index = 100
input_size = 512

data_root = "/home/cxj/Desktop/data/electronic_datasets/0222数据标注/data/point"
save_root = "/home/cxj/Desktop/data/electronic_datasets/0222数据标注/piece_data"

img_names = [x for x in os.listdir(data_root) if
             x.endswith(".png") and not x.endswith("_mask.png")
             and not x.endswith(".json")]

cv2.namedWindow("crop_img", 0)
cv2.namedWindow("crop_label", 0)

for idx, img_name in enumerate(img_names):
    img_path = os.path.join(data_root, img_name)
    mask_path = os.path.join(data_root, img_name[:-4] + "_mask.png")
    img = cv2.imread(img_path)
    mask = cv2.imread(mask_path)
    for jdx in range(crop_index):
        # 随机生成一个点.
        img_h, img_w = img.shape[:2]
        img_all = img.copy()
        mask_all = mask.copy()
        rand_x = random.randint(input_size // 2, img_w - input_size // 2)
        rand_y = random.randint(input_size // 2, img_h - input_size // 2)
        crop_img = img_all[rand_y - input_size // 2: rand_y + input_size // 2,
                           rand_x - input_size // 2: rand_x + input_size // 2]
        crop_label = mask_all[rand_y - input_size // 2: rand_y + input_size // 2,
                            rand_x - input_size // 2: rand_x + input_size // 2]
        if np.max(crop_label) != 255:
            continue
        # 1. rotate
        if random.random() < 0.5:
            rows, cols = crop_img.shape[:2]
            angle = random.randint(-360, 360)
            M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
            crop_img = cv2.warpAffine(crop_img, M, (rows, cols), borderValue=(114, 114, 114))
            crop_label = cv2.warpAffine(crop_label, M, (rows, cols), borderValue=(0, 0, 0))
        # 2. sflip
        if random.random() < 0.5:
            crop_img = cv2.flip(crop_img, 1)
            crop_label = cv2.flip(crop_label, 1)
        else:
            crop_img = cv2.flip(crop_img, 0)
            crop_label = cv2.flip(crop_label, 0)
        cv2.imshow("crop_img", crop_img)
        cv2.imshow("crop_label", crop_label)
        new_img_name = img_name[:-4] + "_" + str(idx) + "_" + str(jdx) + ".png"
        new_mask_name = new_img_name[:-4] + "_mask.png"
        save_img_path = os.path.join(save_root, new_img_name)
        save_mask_path = os.path.join(save_root, new_mask_name)
        cv2.imwrite(save_img_path, crop_img)
        cv2.imwrite(save_mask_path, crop_label)
        cv2.waitKey(1)
