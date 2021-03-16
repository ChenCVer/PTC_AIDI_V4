from .label_trainslate import LabelTranslator
from .loader import BatchLoader


def calc_class_name_list(images_labels, is_debug):
    class_name_list = []
    for image_label in images_labels:
        class_name_list.extend(image_label.list_class_names())
    hist = dict((class_name, class_name_list.count(class_name)) for class_name in set(class_name_list))
    if is_debug:
        print(hist)
    return list(hist.keys())


def rrlabel_to_coco(root_path, save_path, image_type='png', mode='detection', labeltool='LabelRR', is_debug=False):
    """
    :param image_type:
    :param root_path:
    :param save_path:
    :param image_type:
    :param mode: 'detection' or 'keypoints'
    :param labeltool:
    :param is_debug:
    :return:
    """

    batch_loader = BatchLoader(labeltool, image_type)
    raw_data_image_labels = batch_loader.load(root_path)
    label_translator = LabelTranslator()
    label_translator.set_class_name_list(calc_class_name_list(raw_data_image_labels, is_debug))
    label_translator.set_root_path(root_path)
    label_translator.generate_and_save_anno(raw_data_image_labels, mode=mode, save_path=save_path)
