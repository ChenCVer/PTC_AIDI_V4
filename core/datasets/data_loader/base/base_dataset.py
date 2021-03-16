from abc import ABCMeta, abstractmethod
from torch.utils.data import Dataset
from core.datasets.pipelines import Compose


# 默认后缀, 后期需要优化, 尽量写在配置文件中。
IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp',)
IMG_EXCLUDE_EXTENSIONS = ('_mask.jpg', '_mask.jpeg', '_mask.png', '_mask.bmp', ".txt")


class BaseDataset(Dataset, metaclass=ABCMeta):

    CLASSES = None

    def __init__(self,
                 img_prefix,      # 数据存放路径根文件夹
                 pipeline,        # 数据增强
                 ann_file=None,
                 extensions=None,
                 exclude_extensions=None):

        super(BaseDataset, self).__init__()
        self.img_prefix = img_prefix
        self.pipeline = Compose(pipeline)
        self.ann_file = ann_file
        self.extensions = extensions if extensions is not None else IMG_EXTENSIONS
        self.exclude_extensions = exclude_extensions if exclude_extensions is \
                                                        not None else IMG_EXCLUDE_EXTENSIONS
        self.data_infos = self.load_annotations(self.ann_file)

    @abstractmethod
    def load_annotations(self, *args, **kwargs):
        """load all data infos"""
        raise NotImplemented

    @abstractmethod
    def __len__(self):
        """get the len of data"""
        raise NotImplemented

    @abstractmethod
    def __getitem__(self, idx):
        """Get training/test data after pipeline"""
        raise NotImplemented

    @abstractmethod
    def get_cat_ids(self, idx):
        """Get category id by index."""
        raise NotImplemented

    @abstractmethod
    def evaluate(self, *args, **kwargs):
        """eval is worked in eval process"""
        raise NotImplemented