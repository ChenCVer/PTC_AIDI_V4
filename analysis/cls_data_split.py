import os
import random
import shutil
from sklearn.model_selection import KFold


class CrossValidSplit(object):
    def __init__(self,
                 n_slpit=5,
                 data_root="",
                 save_root="",
                 shuffle=True,
                 random_state=None):
        self.data_root = data_root
        self.save_root = save_root
        self.krold = KFold(n_splits=n_slpit, shuffle=shuffle, random_state=random_state)
        self.classes_folder = os.listdir(self.data_root)

    def split(self):
        # 遍历每个类
        for idx, cls in enumerate(self.classes_folder):
            cls_path = os.path.join(self.data_root, cls)
            img_names = os.listdir(cls_path)
            random.shuffle(img_names)
            # 对每个类进行划分
            for idx, (train, test) in enumerate(self.krold.split(img_names)):
                # 创建文件夹
                new_train_root = "data_split_" + str(idx + 1)
                new_save_path_root = os.path.join(self.save_root, new_train_root)
                if not os.path.exists(new_save_path_root):
                    os.mkdir(new_save_path_root)

                # train
                if not os.path.exists(os.path.join(new_save_path_root, "train")):
                    os.mkdir(os.path.join(new_save_path_root, "train"))

                train_folder = os.path.join(new_save_path_root, "train", cls)
                if not os.path.exists(train_folder):
                    os.mkdir(train_folder)

                # val
                if not os.path.exists(os.path.join(new_save_path_root, "val")):
                    os.mkdir(os.path.join(new_save_path_root, "val"))

                val_folder = os.path.join(new_save_path_root, "val", cls)
                if not os.path.exists(val_folder):
                    os.mkdir(val_folder)

                # test
                if not os.path.exists(os.path.join(new_save_path_root, "test")):
                    os.mkdir(os.path.join(new_save_path_root, "test"))

                test_folder = os.path.join(new_save_path_root, "test", cls)
                if not os.path.exists(test_folder):
                    os.mkdir(test_folder)

                # 把划分的数据储存
                for train_id in train:
                    train_img_path = os.path.join(cls_path, img_names[train_id])
                    new_train_img_path = os.path.join(train_folder, img_names[train_id])
                    shutil.copy(train_img_path, new_train_img_path)

                for test_id in test:
                    val_img_path = os.path.join(cls_path, img_names[test_id])
                    new_val_img_path = os.path.join(val_folder, img_names[test_id])
                    shutil.copy(val_img_path, new_val_img_path)

                    test_img_path = os.path.join(cls_path, img_names[test_id])
                    new_test_img_path = os.path.join(test_folder, img_names[test_id])
                    shutil.copy(test_img_path, new_test_img_path)


if __name__ == '__main__':
    data_root = "/home/cxj/Desktop/data/ice_cream_bar_cls_128"
    save_root = "/home/cxj/Desktop/data/ice_cream_bar_cls_128_5_cross"
    cvs = CrossValidSplit(5, data_root, save_root=save_root)
    cvs.split()