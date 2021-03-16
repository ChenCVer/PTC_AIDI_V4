import os
import cv2
import json
import random
import numpy as np

data_root = "/home/cxj/Desktop/data/ice_bar_black_spot_coco"
save_root = "/home/cxj/Desktop/data/ice_bar_black_spot_crop"
train_json_file = os.path.join(save_root, "train.json")
test_json_file = os.path.join(save_root, "test.json")
img_folder_list = os.listdir(data_root)
border_extend = 64

count = 0
ann_count = 1

train_ann_dicts = {
    "info": {"year": 2020 - 12 - 24},
    "licenses": {"name": "chenxunjiao"},
    "images": [],
    "annotations": [],
    "categories": [{"id": 1, "name": "spot"}]
}

test_ann_dicts = {
    "info": {"year": 2020 - 12 - 24},
    "licenses": {"name": "chenxunjiao"},
    "images": [],
    "annotations": [],
    "categories": [{"id": 1, "name": "spot"}]
}

cv2.namedWindow("img", 0)
cv2.namedWindow("label", 0)
cv2.namedWindow("crop_img", 0)
cv2.namedWindow("crop_label", 0)

for idx, img_foler in enumerate(img_folder_list):
    folder_path = os.path.join(data_root, img_foler)
    img_path_root = os.path.join(folder_path, "imgs")
    labels_path_root = os.path.join(folder_path, "labels")
    img_names = sorted(os.listdir(img_path_root))
    labels = os.listdir(labels_path_root)
    labels_index = [x.split("_")[0] for x in labels]

    for jdx, img_name in enumerate(img_names):
        print("img_name = ", img_name)
        img_path = os.path.join(img_path_root, img_name)
        img = cv2.imread(img_path)
        img_idx = img_name[:-4]
        # find img_idx in labels_index
        label_name_list = []
        for i, i_l in enumerate(labels_index):
            if i_l == img_idx:
                label_name_list.append(labels[i])

        # label_path
        label_mask = np.zeros(shape=img.shape[:2], dtype=np.uint8)
        label = np.zeros_like(img)
        for kdx, label_name in enumerate(label_name_list):
            label_path = os.path.join(labels_path_root, label_name)
            sub_label = cv2.imread(label_path)
            sub_label_gray = cv2.cvtColor(sub_label, cv2.COLOR_BGR2GRAY)
            label_mask[sub_label_gray != 0] = 255
            label[label_mask != 0] = 255

        # crop img and label
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_med = cv2.medianBlur(img_gray, 3)
        ret, thresh1 = cv2.threshold(img_med, 60, 255, cv2.THRESH_BINARY)

        # find the max contour
        contours, hierarchy = cv2.findContours(thresh1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        bbox_area_list = []
        bbox_list = []
        for ldx, cnt in enumerate(contours):
            bbox = cv2.boundingRect(cnt)
            bbox_list.append(bbox)
            bbox_area_list.append(bbox[2] * bbox[3])

        max_cnt_idx = bbox_area_list.index(max(bbox_area_list))

        # crop img and label
        max_bbox = bbox_list[max_cnt_idx]
        x, y, w, h = max_bbox
        crop_img = img[y - border_extend:y + h + border_extend,
                   x - border_extend:x + w + border_extend]
        crop_label = label[y - border_extend:y + h + border_extend,
                     x - border_extend:x + w + border_extend]

        # make img_info
        img_info_dict = {"file_name": img_name[:-4] + ".png",
                         "height": crop_img.shape[0],
                         "width": crop_img.shape[1],
                         "id": count}

        if random.random() < 0.7:
            train_ann_dicts["images"].append(img_info_dict)
            # make ann_info
            gray_label = cv2.cvtColor(crop_label, cv2.COLOR_BGR2GRAY)
            _, binary_label = cv2.threshold(gray_label, 10, 255, cv2.THRESH_BINARY)
            cnts, hierarchy = cv2.findContours(binary_label, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for ldx, cnt in enumerate(cnts):
                bbox = cv2.boundingRect(cnt)
                x, y, w, h = bbox
                ann_dict = {"area": (w + 20) * (h + 20),
                            "iscrowd": 0,
                            "image_id": count,
                            "bbox": [x - 10, y - 10, w + 10, h + 10],
                            "category_id": 1,
                            "id": ann_count,
                            "ignore": 0,
                            "segmentation": []}

                train_ann_dicts["annotations"].append(ann_dict)

                ann_count += 1

            save_img_path = os.path.join(save_root, "images/train", img_name[:-4] + ".png")
            save_label_path = os.path.join(save_root, "labels", img_name[:-4] + "_mask.png")

            cv2.imwrite(save_img_path, crop_img)
            cv2.imwrite(save_label_path, crop_label)

        else:
            test_ann_dicts["images"].append(img_info_dict)
            # make ann_info
            gray_label = cv2.cvtColor(crop_label, cv2.COLOR_BGR2GRAY)
            _, binary_label = cv2.threshold(gray_label, 10, 255, cv2.THRESH_BINARY)
            cnts, hierarchy = cv2.findContours(binary_label, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for ldx, cnt in enumerate(cnts):
                bbox = cv2.boundingRect(cnt)
                x, y, w, h = bbox
                ann_dict = {"area": (w + 20) * (h + 20),
                            "iscrowd": 0,
                            "image_id": count,
                            "bbox": [x - 10, y - 10, w + 10, h + 10],
                            "category_id": 1,
                            "id": ann_count,
                            "ignore": 0,
                            "segmentation": []}

                test_ann_dicts["annotations"].append(ann_dict)

                ann_count += 1

            save_img_path = os.path.join(save_root, "images/test", img_name[:-4] + ".png")
            save_label_path = os.path.join(save_root, "labels", img_name[:-4] + "_mask.png")

            cv2.imwrite(save_img_path, crop_img)
            cv2.imwrite(save_label_path, crop_label)

        count += 1

        cv2.imshow("img", img)
        cv2.imshow("label", label)
        cv2.imshow("crop_img", crop_img)
        cv2.imshow("crop_label", crop_label)

        cv2.waitKey(1)


# save dict in json format
with open(train_json_file, 'w') as f:
    json.dump(train_ann_dicts, f)

with open(test_json_file, 'w') as f:
    json.dump(test_ann_dicts, f)