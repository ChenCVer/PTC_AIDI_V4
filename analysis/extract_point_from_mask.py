import cv2
import os

data_root = "/home/cxj/Desktop/data/electronic_datasets/electronic_seg_point_new/train/1.ng"
save_txt_root = "/home/cxj/Desktop/data/electronic_datasets/electronic_seg_point_new/train/label"

imgs= [x for x in os.listdir(data_root) if not x.endswith("_mask.jpg")]

for idx, img_name in enumerate(imgs):
    img_path = os.path.join(data_root, img_name)
    mask_path = img_path[:-4] + "_mask.jpg"
    img = cv2.imread(img_path)
    mask = cv2.imread(mask_path)

    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    ret, binary = cv2.threshold(mask, 120, 255, cv2.THRESH_BINARY)

    cv2.namedWindow("binary", 0)
    cv2.imshow("binary", binary)

    contours, hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    gravtiy = []
    for idx, cnt in enumerate(contours):
        # 求质心位置
        mu = cv2.moments(cnt, False)
        mc = tuple([int(mu['m10'] / mu['m00']), int(mu['m01'] / mu['m00'])])
        gravtiy.append(mc)

    for idx, bbox in enumerate(gravtiy):
        cv2.circle(img, gravtiy[idx], 3, (0, 0, 255))

    cv2.namedWindow("img", 0)
    cv2.imshow("img", img)

    key = chr(cv2.waitKeyEx(0) & 255)  # 等待键盘输入, key
    if key == "q":
        continue
    elif key == "s":
        # 保存数据
        crop_img_label_txt = img_name[:-4] + ".txt"
        crop_img_det_txt = img_name[:-4] + ".txt"
        save_txt_path = os.path.join(save_txt_root, crop_img_label_txt)
        # 保存txt
        with open(save_txt_path, "w") as f:
            f.write(str(len(gravtiy)))
            for i, coord in enumerate(gravtiy):
                content = "\n" + str(coord[0]) + " " + str(coord[1])
                f.write(content)