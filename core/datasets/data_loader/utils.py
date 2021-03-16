import os
import sys
import numpy as np


def find_classes(dir):
    """
    Finds the class folders in a dataset.
    Args:
        dir (string): Root directory path.

    Returns:
        tuple: (classes, class_to_idx) where classes are relative to (dir),
               and class_to_idx is a dictionary.
    Ensures:
        No class is a subdirectory of another.
    """
    # 在程序中查看python的版本。在代码中可以通过sys.version, 或者sys.version_info 得到
    if sys.version_info >= (3, 5):
        # Faster and available in Python 3.5 and above
        classes = [d.name for d in os.scandir(dir) if d.is_dir()]  # os.scandir函数获取的是迭代器
    else:
        classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]

    classes.sort()  # 对文件夹名进行排序, 默认按照首字母大小进行排序
    class_to_idx = {classes[i]: i for i in range(len(classes))}

    return classes, class_to_idx


def _get_all_files(root,
                   output_dict,
                   class_to_idx,
                   label_endswith=None,  # for seg
                   extensions=None,
                   exclude_extensions=None,
                   is_valid_file=None):
    if os.path.isfile(root):
        path = os.path.split(root)[0]
        folder_name = path[path.rfind("/") + 1:]
        output_dict[folder_name] = []
        file_list = os.listdir(path)
        for file_name in file_list:
            file_path = os.path.join(path, file_name)
            if extensions is not None:
                def is_valid_file(x):
                    if exclude_extensions is None:
                        return x.lower().endswith(extensions)
                    else:
                        return x.lower().endswith(extensions) and not \
                            x.lower().endswith(exclude_extensions)
            if is_valid_file(file_path):
                class_list = list(class_to_idx.keys())
                index_arr = np.array([file_path.find(x) for x in class_list])
                target = class_list[np.argmax(index_arr > 0)]
                if label_endswith is not None:
                    img_endswith = os.path.splitext(file_path)[-1]
                    label_path = file_path[:-len(img_endswith)] + label_endswith + img_endswith
                else:
                    label_path = None
                item = {
                    'path': file_path,
                    'cls_id': class_to_idx[target],  # for cls
                    'label_path': label_path  # for seg
                }
                output_dict[folder_name].append(item)

        return output_dict[folder_name]

    else:
        folder_list = os.listdir(root)
        for folder in folder_list:
            if os.path.isfile(os.path.join(root, folder)):
                path = os.path.join(root, folder)
                return _get_all_files(path, {}, class_to_idx, label_endswith, extensions,
                                      exclude_extensions, is_valid_file)
            else:
                output_dict[folder] = {}
                path = os.path.join(root, folder)
                output_dict[folder] = _get_all_files(path, output_dict[folder], class_to_idx, label_endswith,
                                                     extensions, exclude_extensions, is_valid_file)

    return output_dict


def make_dataset(dir,
                 class_to_idx,
                 label_endswith=None,  # for seg
                 extensions=None,
                 exclude_extensions=None,
                 is_valid_file=None):
    images = []
    dir = os.path.expanduser(dir)  # 对dir路径中的"~"用用户名替换掉

    if extensions is not None:

        def is_valid_file(x):
            if exclude_extensions is None:
                return x.lower().endswith(extensions)
            else:
                return x.lower().endswith(extensions) and not \
                    x.lower().endswith(exclude_extensions)

    # 遍历获取到根路径dir的所有文件, class_to_idx是一个字典.
    for target in sorted(class_to_idx.keys()):
        d = os.path.join(dir, target)
        if not os.path.isdir(d):
            continue

        # os.walk()方法用于通过在目录树中游走输出在目录中的文件名，向上或者向下。
        for root, _, fnames in sorted(os.walk(d)):
            for fname in sorted(fnames):
                path = os.path.join(root, fname)
                if label_endswith is not None:
                    img_endswith = os.path.splitext(path)[-1]
                    label_path = path[:-len(img_endswith)] + label_endswith + img_endswith
                else:
                    label_path = None
                if is_valid_file(path):
                    item = {'path': path,
                            'cls_id': class_to_idx[target],  # for cls
                            'label_path': label_path  # for seg
                            }
                    images.append(item)

    return images


def ratio_make_dataset(dir,
                       class_to_idx,
                       label_endswith=None,
                       extensions=None,
                       exclude_extensions=None,
                       is_valid_file=None):
    out_samples = {}  # 装载所有的样本文件路径
    dir = os.path.expanduser(dir)  # 对dir路径中的"~"用用户名替换掉

    out_samples = _get_all_files(dir,
                                 out_samples,
                                 class_to_idx,
                                 label_endswith,
                                 extensions,
                                 exclude_extensions,
                                 is_valid_file)

    return out_samples


def make_dataset_for_keypoints(dir,
                               label_endswith=None,
                               extensions=None,
                               exclude_extensions=None,
                               is_valid_file=None):
    images = []
    dir = os.path.expanduser(dir)  # 对dir路径中的"~"用用户名替换掉

    if extensions is not None:

        def is_valid_file(x):
            if exclude_extensions is None:
                return x.lower().endswith(extensions)
            else:
                return x.lower().endswith(extensions) and not \
                    x.lower().endswith(exclude_extensions)

    # os.walk()方法用于通过在目录树中游走输出在目录中的文件名，向上或者向下。
    for root, _, fnames in sorted(os.walk(dir)):
        for fname in sorted(fnames):
            path = os.path.join(root, fname)
            if label_endswith is not None:
                img_endswith = os.path.splitext(path)[-1]
                label_path = path[:-len(img_endswith)] + label_endswith
            else:
                label_path = None
            if is_valid_file(path):
                item = {'path': path,
                        'label_path': label_path  # for keypoints
                        }
                images.append(item)

    return images