import os
import random

data_root = "/home/cxj/Desktop/data/VOC_mini/VOC2007/JPEGImages/"
train_txt = data_root + "../train.txt"
val_txt = data_root + "../val.txt"
test_txt = data_root + "../test.txt"

img_names = os.listdir(data_root)
train_data = random.sample(img_names, 800)

# 从img_names中剔除掉train_data
new_img_names = list(set(img_names) - set(train_data))
val_data = random.sample(new_img_names, 200)

renew_img_names = list(set(new_img_names) - set(val_data))
test_data = random.sample(renew_img_names, 100)

# 保存train_txt
with open(train_txt, "w") as f:
    for i, coord in enumerate(train_data):
        content = coord[:-4] + "\n"
        f.write(content)


# 保存val_txt
with open(val_txt, "w") as f:
    for i, coord in enumerate(val_data):
        content = coord[:-4] + "\n"
        f.write(content)

# 保存test_txt
with open(test_txt, "w") as f:
    for i, coord in enumerate(test_data):
        content = coord[:-4] + "\n"
        f.write(content)