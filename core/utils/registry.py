import inspect
# inspect模块是针对模块/类/方法/功能等对象提供些有用的方法。例如可以帮助我们检查类的内容,检查方法的代码,提取和格式化方法的参数等。
from functools import partial


# 详细讲解请看: https://zhuanlan.zhihu.com/p/86808911
#             https://blog.csdn.net/leijieZhang/article/details/90747741
#             https://blog.csdn.net/qq_41375609/article/details/99549794
class Registry(object):

    def __init__(self, name):
        self._name = name
        self._module_dict = dict()

    def __repr__(self):
        # 返回一个可以用来表示对象的可打印字符串，可以理解为c++中的String。
        format_str = self.__class__.__name__ + '(name={}, items={})'.format(
            self._name, list(self._module_dict.keys()))
        return format_str

    # @property可以把方法变成属性，通过self.name 就能获得name的值
    @property
    def name(self):
        return self._name  # 因为没有定义它的setter方法，所以是个只读属性，不能通过 self.name = newname进行修改。

    @property
    def module_dict(self):
        # 同上，通过self.module_dict可以获取属性_module_dict，也是只读的
        return self._module_dict

    def get(self, key):
        # 普通方法，获取字典中指定key的value，_module_dict是一个字典，然后就可以通过self.get(key),获取value值
        return self._module_dict.get(key, None)

    def _register_module(self, module_class, force=None):
        """Register a module.

        Args:
            module (:obj:`nn.Module`): Module to be registered.
        """
        # 关键的一个方法，作用就是Register a module.
        # 在model文件夹下的py文件中，里面的class定义上面都会出现 @DETECTORS.register_module，意思就是将类当做形参，
        # 将类送入了方法register_module()中执行。@的具体用法看后面解释。
        if not inspect.isclass(module_class):  # 判断是否为类，是类的话，就为True，跳过判断
            raise TypeError('module must be a class, but got {}'.format(type(module_class)))

        module_name = module_class.__name__  # 获取类名
        if module_name in self._module_dict:  # 看该类是否已经登记在属性_module_dict中
            raise KeyError('{} is already registered in {}'.format(module_name, self.name))

        self._module_dict[module_name] = module_class  # 在module中dict新增key和value。key为类名，value为类对象

    def register_module(self, cls=None, force=False):
        if cls is None:
            return partial(self.register_module, force=force)
        self._register_module(cls, force=force)
        return cls


def build_from_cfg(cfg, registry, default_args=None):
    """Build a module from config dict.
    Args:
        cfg (dict): Config dict. It should at least contain the key "type".
        registry (:obj:`Registry`): The registry to search the type from.
        default_args (dict, optional): Default initialization arguments.
    Returns:
        obj: The constructed object.
    """
    assert isinstance(cfg, dict) and 'type' in cfg
    assert isinstance(default_args, dict) or default_args is None
    args = cfg.copy()
    obj_type = args.pop('type')
    if isinstance(obj_type, str):
        # 通过type在注册类实例register中获取对应的模型类(Faster RCNN, Mask RCNN等), 比如obj_type="Faster RCNN",
        # 则registry.get(obj_type) ===> 调用Registry中的get方法, 通过key在_module_dict中找到对应的模型类
        # obj_type="Faster RCNN", 则obj_cls对应于: <class 'core.models.objectdetection.faster_rcnn.FasterRCNN'>
        obj_cls = registry.get(obj_type)
        if obj_cls is None:
            raise KeyError('{} is not in the {} registry'.format(obj_type, registry.name))
    elif inspect.isclass(obj_type):
        obj_cls = obj_type
    else:
        raise TypeError('type must be a str or valid type, but got {}'.format(type(obj_type)))
    if default_args is not None:
        for name, value in default_args.items():
            # args是一个ConfigDict字典
            # 字典setdefault()方法和get()方法类似,返回指定键的值,如果键不在字典中,将会添加键并将值设置为一个指定值,默认为None。
            # get()和setdefault()区别: setdefault()返回的键如果不在字典中,会添加键(更新字典),而get()不会添加键。
            args.setdefault(name, value)  # 这里的意思是获取args中键为name的值, 如果name不存在, 则在args中为键name设定默认值value
    # obj_cls(**args): 这里则是对args进行解包处理, 然后实例化obj_cls, 返回一个实例
    # args是一个字典比如{"pretrained": "torchvision://resnet50", "backbone": {'type': "Resnet","out_channel": 256}}
    # 字典解包处理得到的是关键字实参
    return obj_cls(**args)
