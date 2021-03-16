import json
import numpy as np
import datetime
import os


def check_is_init_correct(func):
    def wrapper(YourLabelTranslator, *args, **kwargs):
        if YourLabelTranslator.class_name_list is None:
            raise (Exception, "Please Set class_name_list First")
        elif YourLabelTranslator.root_dirpath is None:
            raise (Exception, "Please Set root_path First")
        else:
            result = func(YourLabelTranslator, *args, **kwargs)
            return result

    return wrapper


class LabelTranslator(object):

    def __init__(self):

        self._class_name_list = None
        self._root_dirpath = None

    def set_class_name_list(self, value):
        self.class_name_list = value

    def set_root_path(self, value):
        self.root_dirpath = value

    @property
    def root_dirpath(self):
        if self._root_dirpath is None:
            Warning("root_path is None, Please Set the root_path")
        return self._root_dirpath

    @root_dirpath.setter
    def root_dirpath(self, value):
        assert isinstance(value, str)
        self._root_dirpath = value

    @property
    def class_name_list(self):
        if self._class_name_list is None:
            Warning("class_name_list is None, Please Set the class_name_list")
        return self._class_name_list

    @class_name_list.setter
    def class_name_list(self, value):
        assert isinstance(value, list)
        for v in value:
            if not isinstance(v, str):
                raise Exception("The Element of class_name_list must be string")
        self._class_name_list = value

    def save_anno(self, anno_dict, save_path):
        save_dir = os.path.join(os.getcwd(), save_path.replace(os.path.basename(save_path), ''))
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        with open(save_path, 'w') as f_coco:
            json.dump(anno_dict, f_coco)

    @check_is_init_correct
    def generate_and_save_anno(self, image_labels_list, mode, save_path):
        coco_dict = self.generate_anno(image_labels_list, mode=mode)
        self.save_anno(coco_dict, save_path=save_path)
        return coco_dict

    @check_is_init_correct
    def generate_anno(self, image_labels_list, mode):
        assert mode in ["keypoints", "detection"]
        generate_methods = {
            "detection": self.generate_one_detection_anno,
            "keypoints": self.generate_one_keypoint_anno
        }
        current_image_id = 0
        current_anno_id = 0
        image_info_list = []
        anno_info_list = []
        info, license_list = self.create_head_info()
        categories_info_list = self.create_categories_info_list()
        for image_label in image_labels_list:
            image_info, one_image_anno_info_list, current_image_id, current_anno_id \
                = generate_methods[mode](image_label, current_image_id, current_anno_id)
            image_info_list.append(image_info)
            anno_info_list.extend(one_image_anno_info_list)
        complete_coco_dict = self.create_complete_coco_dict(info, license_list, categories_info_list, image_info_list,
                                                            anno_info_list)
        return complete_coco_dict

    @check_is_init_correct
    def generate_one_detection_anno(self, img_label, lasted_image_id, lasted_anno_id):  # image and anno id start from 1
        num_of_bboxes = len(img_label.bboxes)
        current_image_id = lasted_image_id + 1
        height, width = img_label.image_shape
        relative_path = self.calc_relative_path(img_label.image_path, self.root_dirpath)
        image_info = self.create_image_info(image_id=current_image_id, relative_path=relative_path,
                                            width=width, height=height)
        one_image_anno_info_list = []
        for idx, bbox in enumerate(img_label.bboxes):
            current_anno_id = lasted_anno_id + idx + 1
            anno_info = self.create_bbox_annotations_info(current_image_id, current_anno_id, bbox)
            one_image_anno_info_list.append(anno_info)
        current_anno_id = lasted_anno_id + num_of_bboxes
        return image_info, one_image_anno_info_list, current_image_id, current_anno_id

    @check_is_init_correct
    def generate_one_keypoint_anno(self, img_label, lasted_image_id, lasted_anno_id):
        num_of_samples = len(img_label.samples)
        current_image_id = lasted_image_id + 1
        height, width = img_label.image_shape
        relative_path = self.calc_relative_path(img_label.image_path, self.root_dirpath)
        image_info = self.create_image_info(image_id=current_image_id, relative_path=relative_path,
                                            width=width, height=height)
        one_image_anno_info_list = []
        for idx, sample in enumerate(img_label.samples):
            current_anno_id = lasted_anno_id + idx + 1
            anno_info = self.create_kpts_annotations_info(current_image_id, current_anno_id, sample)
            one_image_anno_info_list.append(anno_info)
        current_anno_id = lasted_anno_id + num_of_samples
        return image_info, one_image_anno_info_list, current_image_id, current_anno_id

    def create_complete_coco_dict(self, info_part, licenses_part, categories_part, images_part, annotations_part):
        complete_coco_dict = {
            "info": info_part,
            "licenses": licenses_part,
            "categories": categories_part,
            "images": images_part,
            "annotations": annotations_part
        }
        return complete_coco_dict

    def create_head_info(self):
        info = {
            "description": "char",
            "url": "https://github.com/waspinator/pycococreator",
            "version": "0.1.0",
            "year": datetime.datetime.utcnow().year,
            "contributor": "waspinator",
            "date_created": datetime.datetime.utcnow().isoformat(' ')}
        license_list = [
            {
                "id": 1,
                "name": "Attribution-NonCommercial-ShareAlike License",
                "url": "http://creativecommons.org/licenses/by-nc-sa/2.0/"
            }
        ]
        return info, license_list

    @check_is_init_correct
    def create_categories_info_list(self):
        categories_info_list = []
        for idx, class_name in enumerate(self.class_name_list):
            categories_info = {
                "id": idx + 1,  # cls_id 0 means background, foreground start from 1
                "name": class_name,
                "supercategory": "object"  # everything is object
            }
            categories_info_list.append(categories_info)
        return categories_info_list

    def create_image_info(self, image_id, relative_path, width, height):
        image_info = {
            "id": image_id,
            "file_name": relative_path,
            "width": width,
            "height": height,
            "date_captured": datetime.datetime.utcnow().isoformat(' '),
            "license": 1,
            "coco_url": "",
            "flickr_url": ""
        }
        return image_info

    @check_is_init_correct
    def create_bbox_annotations_info(self, image_id, anno_id, bbox):
        # bbox = sample.bbox
        """
        In this function, we use bbox as input to use bbox as much as possible and avoid the impact of error
        keypoints label.
        :param image_id: image_id
        :param anno_id: anno_id
        :param bbox: bbox
        :return:
        """
        coco_bbox_xywh = [bbox.ul_point[0], bbox.ul_point[1], bbox.width, bbox.height]
        coco_bbox_xywh_int32 = list(map(int, coco_bbox_xywh))
        cls_id = self.class_name_list.index(bbox.class_name) + 1  # cls_id 0 means background, foreground start from 1
        area = int(bbox.width * bbox.height)
        anno_info = {
            "id": anno_id,
            "image_id": image_id,
            "category_id": cls_id,
            "area": area,
            "bbox": coco_bbox_xywh_int32
        }
        return anno_info

    @check_is_init_correct
    def create_kpts_annotations_info(self, image_id, anno_id, sample):
        """
        In this function, we choose Sample Instance as input because the bbox and keypoint must be in a correct group
        :param image_id: image_id
        :param anno_id: anno_id
        :param sample: sample label
        :return:
        """
        bbox = sample.bbox
        keypoint = sample.key_points
        num_keypoints = keypoint.points.shape[0]
        coco_keypoints = np.hstack([keypoint.points, np.ones((num_keypoints, 1)) * 2])  # coco's keypoint format
        cls_id = self.class_name_list.index(bbox.class_name) + 1
        coco_bbox_xywh = [bbox.ul_point[0], bbox.ul_point[1], bbox.width, bbox.height]
        coco_bbox_xywh_int32 = list(map(int, coco_bbox_xywh))
        area = int(bbox.width * bbox.height)
        anno_info = {
            "id": anno_id,
            "image_id": image_id,
            "iscrowd": 0,
            "category_id": cls_id,
            "area": area,
            "bbox": coco_bbox_xywh_int32,
            "keypoints": coco_keypoints.flatten().tolist(),
            "num_keypoints": num_keypoints
        }
        return anno_info

    def calc_relative_path(self, abspath, root_path):
        assert root_path in abspath, "root_path must be in abspath"
        relative_path = abspath.split(root_path)[1]
        if relative_path.startswith("/"):
            relative_path = relative_path[1:]
        return relative_path
