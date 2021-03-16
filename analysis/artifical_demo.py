import os
import cv2
import numpy as np

data_root = "/home/cxj/Desktop/data/electronic_datasets/0201数据标注/线_mask"
save_root = "/home/cxj/Desktop/data/electronic_datasets/0201数据标注/results"
img_names = [x for x in os.listdir(data_root)if not x.endswith("_mask.png")]

for idx, img_name in enumerate(img_names):
    img_path = os.path.join(data_root, img_name)
    mask_path = os.path.join(data_root, img_name[:-4] + "_mask.png")
    img = cv2.imread(img_path)
    mask_img = cv2.imread(mask_path)
    gray = cv2.cvtColor(mask_img, cv2.COLOR_BGR2GRAY)  # 把输入图像灰度化
    # 直接阈值化是对输入的单通道矩阵逐像素进行阈值分割。
    cnt_count = 0
    mask = np.zeros_like(img)
    ret, binary = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)
    cv2.namedWindow("img", 0)
    cv2.imshow("img", binary)
    cv2.waitKey(1)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for idx, cnt in enumerate(contours):
        # solve the min area rect for cnt
        min_rect = cv2.minAreaRect(cnt)
        center = min_rect[0]
        area = min_rect[1][0] * min_rect[1][1]
        box = np.int0(cv2.boxPoints(min_rect))
        if area < 100:
            cv2.drawContours(mask, [box], 0, (0, 0, 255), 2)
            # cv2.circle(mask, (int(center[0]), int(center[1])), 2, (0, 255, 0), -1)
        else:
            cv2.drawContours(mask, [box], 0, (0, 255, 0), 2)
            # cv2.circle(mask, (int(center[0]), int(center[1])), 2, (0, 255, 0), -1)
        cnt_count += 1

    cv2.putText(mask, str(cnt_count), (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 2)

    final_result = cv2.addWeighted(img, 0.7, mask, 0.3, 0)
    save_path = os.path.join(save_root, img_name[:-4] + "result.png")
    cv2.imwrite(save_path, final_result)