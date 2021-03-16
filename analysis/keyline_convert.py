import os
import cv2
import json
import numpy as np

"""
标注软件采用的是labelme的line, 是json文件.
"""

data_root = "/home/cxj/Desktop/data/electronic_datasets/0222数据标注/orig/point"
save_root = "/home/cxj/Desktop/data/electronic_datasets/0222数据标注/mask"


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


img_names = [x for x in os.listdir(data_root) if not x.endswith(".json") and not x.endswith("_mask.png")]

cv2.namedWindow("mask", 0)
cv2.namedWindow("gauss", 0)
for idx, img_name in enumerate(img_names):
    print("img_name: ", img_name)
    img_path = os.path.join(data_root, img_name)
    json_path = os.path.join(data_root, img_name[:-4] + ".json")
    img = cv2.imread(img_path)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mask = np.zeros_like(img_gray)
    with open(json_path, 'r', encoding='utf8')as fp:
        json_data = json.load(fp)
        labels = json_data["shapes"]
        for jdx, label in enumerate(labels):
            points = label["points"]
            try:
                cv2.line(mask, (int(points[0][1]), int(points[0][0])),
                         (int(points[1][1]), int(points[1][0])), 255, 1,
                         lineType=cv2.LINE_4)
            except:
                cv2.circle(mask, (int(points[0][1]), int(points[0][0])), 1, 255, -1)

    cv2.imshow("mask", mask)
    # guass filter
    points = np.argwhere(mask == 255)
    heatmap = np.zeros_like(mask).astype('float64')
    for kdx, point in enumerate(points):
        center = (point[0], point[1])
        draw_umich_gaussian(heatmap, center, 20)

    heatmap = np.uint8(heatmap * 255)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_GRAY2BGR)
    img_me = cv2.addWeighted(img, 0.5, heatmap, 0.5, 0)

    cv2.imshow("gauss", img_me)

    # save
    save_path = os.path.join(save_root, img_name[:-4] + "_mask.png")
    cv2.imwrite(save_path, heatmap)
    cv2.waitKey(0)
