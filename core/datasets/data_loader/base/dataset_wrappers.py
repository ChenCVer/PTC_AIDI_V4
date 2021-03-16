import math
import bisect
import numpy as np
from collections import defaultdict
from core.datasets.registry import DATASETS
from torch.utils.data.dataset import ConcatDataset as _ConcatDataset

__all__ = [
    'ConcatDataset',
    'RepeatDataset',
    'ClassBalancedDataset']


@DATASETS.register_module()
class ConcatDataset(_ConcatDataset):
    """A metrics of concatenated dataset.

    Same as :obj:`torch.utils.data.dataset.ConcatDataset`, but
    concat the group flag for image aspect ratio.

    Args:
        datasets (list[:obj:`Dataset`]): A list of datasets.
    """

    def __init__(self, datasets):
        super(ConcatDataset, self).__init__(datasets)
        self.CLASSES = datasets[0].CLASSES
        if hasattr(datasets[0], 'flag'):
            flags = []
            for i in range(0, len(datasets)):
                flags.append(datasets[i].flag)
            self.flag = np.concatenate(flags)
        else:
            self.flag = None

    def get_cat_ids(self, idx):
        """Get category ids of concatenated dataset by index

        Args:
            idx (int): Index of data.

        Returns:
            list[int]: All categories in the image of specified index.
        """

        if idx < 0:
            if -idx > len(self):
                raise ValueError(
                    'absolute value of index should not exceed dataset length')
            idx = len(self) + idx
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
        return self.datasets[dataset_idx].get_cat_ids(sample_idx)


@DATASETS.register_module()
class RepeatDataset(object):
    """A metrics of repeated dataset.

    The length of repeated dataset will be `times` larger than the original
    dataset. This is useful when the data loading time is long but the dataset
    is small. Using RepeatDataset can reduce the data loading time between
    epochs.

    Args:
        dataset (:obj:`Dataset`): The dataset to be repeated.
        times (int): Repeat times.
    """

    def __init__(self, dataset, times):
        self.dataset = dataset
        self.times = times
        self.CLASSES = dataset.CLASSES
        if hasattr(self.dataset, 'flag'):
            self.flag = np.tile(self.dataset.flag, times)
        else:
            self.flag = None

        self._ori_len = len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx % self._ori_len]

    def get_cat_ids(self, idx):
        """Get category ids of repeat dataset by index

        Args:
            idx (int): Index of data.

        Returns:
            list[int]: All categories in the image of specified index.
        """

        return self.dataset.get_cat_ids(idx % self._ori_len)

    def __len__(self):
        """Length after repetition"""
        return self.times * self._ori_len


class ClassBalancedDataset(object):
    """
    Note: 对于分类, 分割, 如果gather_flag=True, 表示要从batch角度去控制类别比例
          而本类是从epoch的角度来控制类别均衡, 因此, 如果用本类, 则不能开启gather_flag.
    """
    """A metrics of repeated dataset with repeat factor.

    Suitable for training on class imbalanced datasets like LVIS. Following
    the sampling strategy in [1], in each epoch, an image may appear multiple
    times based on its "repeat factor".
    The repeat factor for an image is a function of the frequency the rarest
    category labeled in that image. The "frequency of category c" in [0, 1]
    is defined by the fraction of images in the training set (without repeats)
    in which category c appears.
    The dataset needs to instantiate :func:`self.get_cat_ids(idx)` to support
    ClassBalancedDataset.
    The repeat factor is computed as followed.
    1. For each category c, compute the fraction # of images
        that contain it: f(c), 计算每个类所占总类的频率f(c)
    2. For each category c, compute the category-level repeat factor:
        r(c) = max(1, sqrt(t/f(c))), 计算每个类需要重复比
    3. For each image I, compute the image-level repeat factor:
        r(I) = max_{c in I} r(c), 计算每张图片需要的重复比

    References:
        .. [1]  https://arxiv.org/pdf/1903.00621v2.pdf

    Args:
        dataset (:obj:`CustomDataset`): The dataset to be repeated.
        oversample_thr (float): frequency threshold below which data is
            repeated. For categories with `f_c` >= `oversample_thr`, there is
            no oversampling. For categories with `f_c` < `oversample_thr`, the
            degree of oversampling following the square-root inverse frequency
            heuristic above.
    """

    def __init__(self, dataset, oversample_thr):

        if hasattr(dataset, 'gather_flag') and dataset.gather_flag is True:
            raise TypeError("gather_flag must be False. you should use ClsDataset, "
                            "but not RatioClsDataset, as well as for segmentation")
        self.dataset = dataset
        self.oversample_thr = oversample_thr
        self.CLASSES = dataset.CLASSES
        # repeat_factors是表示每张图片需要被复制的次数
        repeat_factors = self._get_repeat_factors(dataset, oversample_thr)
        repeat_indices = []
        for dataset_index, repeat_factor in enumerate(repeat_factors):
            # 这里就是将图片下标复制度对应的次数, 比如dataset中序列为1的图片,需要被复制3次
            # 则repeat_indices就会出现三个1. 这是基于epoch的权重随机采样.
            repeat_indices.extend([dataset_index] * math.ceil(repeat_factor))
        self.repeat_indices = repeat_indices

        flags = []
        if hasattr(self.dataset, 'flag'):  # 检测
            for flag, repeat_factor in zip(self.dataset.flag, repeat_factors):
                flags.extend([flag] * int(math.ceil(repeat_factor)))
            assert len(flags) == len(repeat_indices)
            self.flag = np.asarray(flags, dtype=np.uint8)
        else:
            self.flag = None  # 分类分割

    def _get_repeat_factors(self, dataset, repeat_thr):
        """Get repeat factor for each images in the dataset.

        Args:
            dataset (:obj:`CustomDataset`): The dataset
            repeat_thr (float): The threshold of frequency. If an image
                contains the categories whose frequency below the threshold,
                it would be repeated.

        Returns:
            list[float]: The repeat factors for each images in the dataset.
        """

        # 1. For each category c, compute the fraction # of images
        #   that contain it: f(c)
        category_freq = defaultdict(int)
        num_images = len(dataset)
        for idx in range(num_images):
            cat_ids = set(self.dataset.get_cat_ids(idx))
            for cat_id in cat_ids:
                category_freq[cat_id] += 1     # 统计每个类出现的频数
        for k, v in category_freq.items():
            category_freq[k] = v / num_images  # compute f(c)

        # 2. For each category c, compute the category-level repeat factor:
        #    r(c) = max(1, sqrt(t/f(c)))
        category_repeat = {
            cat_id: max(1.0, math.sqrt(repeat_thr / cat_freq))  # 每一类的重复因子.
            for cat_id, cat_freq in category_freq.items()
        }

        # 3. For each image I, compute the image-level repeat factor:
        #    r(I) = max_{c in I} r(c)
        repeat_factors = []
        for idx in range(num_images):
            cat_ids = set(self.dataset.get_cat_ids(idx))  # 每张图取出类别
            # 从类别的重复因子中category_repeat获取出该类需要重复的次数,
            # 然后对应的赋值给这张图片需要被复制的次数.
            repeat_factor = max({category_repeat[cat_id] for cat_id in cat_ids})
            repeat_factors.append(repeat_factor)

        return repeat_factors

    def __getitem__(self, idx):
        ori_index = self.repeat_indices[idx]
        return self.dataset[ori_index]

    def __len__(self):
        """Length after repetition"""
        return len(self.repeat_indices)
