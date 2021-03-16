from core.utils import Registry

# 生成模型组件的注册类
# #类的实例化，Registry是一个类，传入的是一个字符串。该字符串为Registry类的name属性值
MODELS = Registry('models')
BACKBONES = Registry('backbones')
NECKS = Registry('necks')
HEADS = Registry('heads')
LOSSES = Registry('loss')


__all__ = [
    'MODELS',
    'BACKBONES',
    'NECKS',
    'HEADS',
    'LOSSES',
]


"""
# 注意一个知识点:
# ①from package.module import xxx和②import package.module.xxx都会将package的__init__.py文件全部执行一遍;
# import package_name, 本质就是解释(执行)该package_name下的__init__.py文件.
"""