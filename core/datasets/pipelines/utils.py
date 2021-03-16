import os
import csv
import numpy as np
import cv2

__all__ = ['get_roi_dict', 'get_view_id', 'get_roi_from_poly', 'get_roi_image']


def get_roi_dict(csv_path):
    '''
    读取roi csv内容到字典，roi的格式为：
    1,[pt0_x,pt0_y,pt1_x,pt1_y] 或者
    1,[[pt0_x,pt0_y],[pt1_x,pt1_y],[pt2_x,pt2_y],[pt3_x,pt3_y]]
    :param csv_path:
    :return:
    '''

    def remove_strs(str, remove_str_list=['[', ']', ' ']):
        str_result = ''
        for s in str:
            if s not in remove_str_list:
                str_result += s
        return str_result

    if csv_path == '' or csv_path == 'None' or csv_path is None:
        return {0: []}

    filename, file_extension = os.path.splitext(csv_path)
    if not file_extension == ".csv":
        return ValueError("File is not a CSV!")

    roi_dict = dict()
    with open(csv_path, 'r') as csvfile:
        file_reader = csv.reader(csvfile)
        for row in file_reader:
            if len(row) == 0:
                continue
            assert len(row) % 4 == 1, '输入ROI格式错误'
            view_id = int(remove_strs(row[0]))
            all_roi = []
            one_roi = []
            pts = []
            for i in range(1, len(row)):
                if remove_strs(row[1], ' ')[1] != '[':  # 格式1
                    one_roi.append(int(remove_strs(row[i])))
                    if i % 4 == 0:
                        all_roi.append(one_roi)
                        one_roi = []
                    roi_dict[view_id] = all_roi
                else:  # 格式2
                    pts.append(int(remove_strs(row[i])))
                    if i % 2 == 0:
                        one_roi.append(pts)
                        pts = []
                    if i % 8 == 0:
                        all_roi.append(one_roi)
                        one_roi = []
                    roi_dict[view_id] = all_roi

        roi_poly_dict = dict()
        for key, value in roi_dict.items():
            roi_poly = []
            for roi in value:
                if not isinstance(roi[0], list):  # 格式1
                    roi_poly.append([[roi[0], roi[1]],
                                     [roi[2], roi[1]],
                                     [roi[2], roi[3]],
                                     [roi[0], roi[3]]])
                else:
                    roi_poly.append(roi)
            roi_poly_dict[key] = roi_poly

    return roi_poly_dict


def get_view_id(img_path, name_type='_id'):
    if img_path.find('/') > -1:
        img_name = img_path[img_path.rfind('/') + 1:]
    else:
        img_name = img_path
    try:
        view_id = get_view_id_recur(img_name, name_type=name_type)
    except:
        raise RuntimeError('配置文件中的nameType输入不符合规范！')
    return str(view_id)


def get_view_id_recur(img_name, name_type):
    # 首先找id，若为0，说明第一位为view id
    if name_type.find('id') == 0:
        if img_name.find('_') > 0:
            view_id_str = img_name[0:img_name.find('_')]
        else:
            view_id_str = img_name[0:img_name.find('.')]
        try:
            view_id = int(view_id_str)
        except:  # 找不到的默认为0
            view_id = 0
        return view_id
    elif name_type.find('id') > 0:
        name_type = name_type[name_type.find('_') + 1:]
        if img_name.find('_') < 0:
            return 0
        img_name = img_name[img_name.find('_') + 1:]

    return get_view_id_recur(img_name, name_type)


def get_roi_from_poly(src_shape_hw, default_roi_xy):
    roi_mask = np.zeros(src_shape_hw, dtype=np.uint8)
    roi = default_roi_xy
    roi_list_xy = np.array(roi).reshape((-1, 2))
    # bbox = [y(行)最小, y最大, x(列)最小, x最大]
    bbox = [sorted(roi_list_xy[:, 1])[0], sorted(roi_list_xy[:, 1])[-1],
            sorted(roi_list_xy[:, 0])[0], sorted(roi_list_xy[:, 0])[-1]]
    cv2.fillPoly(roi_mask, [np.array(roi)], 255)
    roi_mask = roi_mask[bbox[0]:bbox[1], bbox[2]:bbox[3]]
    return bbox, roi_mask


def get_roi_image(bbox, image):
    return image[bbox[0]:bbox[1], bbox[2]:bbox[3]]
