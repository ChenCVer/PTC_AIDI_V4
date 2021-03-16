import re
from collections import abc
import os

import numpy as np
import torch

__all__ = ['parse_rraitools_version', 're_version', 'cast_tensor_type', 'convert_model_type', 'pytorch_type']

pytorch_type = {
    'double': torch.double,
    'float32': torch.float32,
    'float16': torch.float16,
    'int64': torch.int64,
    'int32': torch.int32,
    'int16': torch.int16,
    'int8': torch.int8
}


def convert_model_type(model, dst_type):
    assert dst_type in ['double', 'float32', 'float16'], 'pytorch not support'
    if dst_type == 'double':
        model = model.double()
    elif dst_type == 'float32':
        model = model.float()
    elif dst_type == 'float16':
        model = model.half()
    elif dst_type == 'int64':
        model = model.long()
    elif dst_type == 'int32':
        model = model.int()
    elif dst_type == 'int16':
        model = model.short()
    elif dst_type == 'int8':
        model = model.byte()
    return model


def re_version(line):
    pat = '(' + '|'.join(['>=', '==', '>']) + ')'
    parts = re.split(pat, line, maxsplit=1)
    parts = [p.strip() for p in parts]
    version_str = ''.join(parts[-1].split('.'))
    return version_str


def parse_rraitools_version(fname):
    rraitools_version = ''
    if not os.path.exists(fname):
        return -1  # 无特殊含义
    with open(fname, 'r') as f:
        for line in f.readlines():
            line = line.strip()
            if line.find('rraitools') >= 0:
                rraitools_version = line
            break
    return re_version(rraitools_version)


def cast_tensor_type(inputs, src_type, dst_type):
    if isinstance(inputs, torch.Tensor):
        return inputs.to(dst_type)
    elif isinstance(inputs, str):
        return inputs
    elif isinstance(inputs, np.ndarray):
        return inputs
    elif isinstance(inputs, abc.Mapping):
        return type(inputs)({
            k: cast_tensor_type(v, src_type, dst_type)
            for k, v in inputs.items()
        })
    elif isinstance(inputs, abc.Iterable):
        return type(inputs)(
            cast_tensor_type(item, src_type, dst_type) for item in inputs)
    else:
        return inputs
