import os
import cv2
import numpy as np
txt_path_root = '/home/cxj/Desktop/data/electronic_datasets/0201数据标注/点_512'
save_path_root = "/home/cxj/Desktop/data/electronic_datasets/0201数据标注/点_mask"

orig_lsit = [x for x in os.listdir(txt_path_root) if x.endswith(".txt")]


def gaussian2D(shape, sigma=1):
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]
    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    # 限制最小的值
    return h


def draw_umich_gaussian(heatmap, center, radius, k=1):
    diameter = 2 * radius + 1
    gaussian = gaussian2D((diameter, diameter), sigma=diameter / 5)
    # 一个圆对应内切正方形的高斯分布
    x, y = int(center[0]), int(center[1])
    width, height = heatmap.shape
    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)
    masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
    masked_gaussian = gaussian[radius - top:radius +
                               bottom, radius - left:radius + right]
    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:
        np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
        # 将高斯分布覆盖到heatmap上，取最大，而不是叠加
    return heatmap


for txt_file in orig_lsit:
    print("txt_file: ", txt_file)
    txt_path = os.path.join(txt_path_root, txt_file)
    img_path = txt_path[:-4] + ".jpg"
    img = cv2.imread(img_path)
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_h, img_w = img.shape[:2]
    heatmap = np.zeros((img_h, img_w))
    with open(txt_path, "r") as f:
        for i, line in enumerate(f):
            if i == 0:
                # 第一行
                continue
            else:
                x1, y1 = [(t.strip()) for t in line.split()]
                cx, cy = int(x1), int(y1)
                draw_umich_gaussian(heatmap, (cx, cy), 5)

    heatmap = np.uint8(heatmap * 255.0)

    img_weight = cv2.addWeighted(img, 0.5, heatmap, 0.5, 0)
    cv2.namedWindow("img", 0)
    cv2.imshow("img", img_weight)
    cv2.waitKey(0)
    mask_path = os.path.join(save_path_root, txt_file[:-4] + "_mask.jpg")
    cv2.imwrite(mask_path, heatmap)
