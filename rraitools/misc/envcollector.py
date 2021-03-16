# Copyright (c) , Inc. and its affiliates. All Rights Reserved.
import os
import sys
import subprocess
from collections import defaultdict
import random
import cv2
import numpy as np
from tabulate import tabulate
import torch
import torchvision
from torch.utils.collect_env import get_pretty_env_info


class EnvCollectHelper(object):
    """
    Provide some basic operation relating to code running environment.
    """
    @staticmethod
    def collect_torch_env():
        try:
            import torch.__config__

            return torch.__config__.show()
        except ImportError:
            # compatible with older versions of pytorch
            return get_pretty_env_info()

    @staticmethod
    def set_environ(gpus):
        """
        :param gpus:
        :return:
        """
        if isinstance(gpus, str):
            # 按照PCI_BUS_ID顺序从0开始排列GPU设备
            os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
            # 设置当前使用的GPU设备仅为gpus号设备  设备名称为'/gpu:gpus'
            os.environ["CUDA_VISIBLE_DEVICES"] = gpus
            return
        if not isinstance(gpus, list):
            gpus = [gpus]
        gpus = list(map(str, gpus))
        if len(gpus) == 1:
            gpus = '' + str(gpus[0])
        else:
            gpus = ','.join(gpus)
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = gpus

    @staticmethod
    def collect_env_info():
        """
        Collect all information, including version of basic libraries.

        :return: string
        """
        data = list()
        data.append(("sys.platform", sys.platform))
        data.append(("Python", sys.version.replace("\n", "")))
        data.append(("Numpy", np.__version__))
        data.append(("Opencv", cv2.__version__))
        data.append(("PyTorch", torch.__version__))
        data.append(("PyTorch Debug Build", torch.version.debug))
        try:
            data.append(("Torchvision", torchvision.__version__))
        except AttributeError:
            data.append(("Torchvision", "unknown"))

        has_cuda = torch.cuda.is_available()
        data.append(("CUDA available", has_cuda))
        if has_cuda:
            devices = defaultdict(list)
            for k in range(torch.cuda.device_count()):
                devices[torch.cuda.get_device_name(k)].append(str(k))
            for name, devids in devices.items():
                data.append(("GPU " + ",".join(devids), name))
            # 获取CUDA路径
            from torch.utils.cpp_extension import CUDA_HOME
            data.append(("CUDA_HOME", str(CUDA_HOME)))

            if CUDA_HOME is not None and os.path.isdir(CUDA_HOME):
                try:
                    nvcc = os.path.join(CUDA_HOME, "bin", "nvcc")
                    nvcc = subprocess.check_output("'{}' -V | tail -n1".format(nvcc), shell=True)
                    nvcc = nvcc.decode("utf-8").strip()
                except subprocess.SubprocessError:
                    nvcc = "Not Available"
                data.append(("NVCC", nvcc))

            cuda_arch_list = os.environ.get("TORCH_CUDA_ARCH_LIST", None)
            if cuda_arch_list:
                data.append(("TORCH_CUDA_ARCH_LIST", cuda_arch_list))

        env_str = tabulate(data) + "\n"  # tabulate直译是制表, 让python实现表格化显示。
        env_str += EnvCollectHelper.collect_torch_env()
        return env_str

    @staticmethod
    def seed_all_rng(seed=None, deterministic=True):
        """
        Set the random seed for the RNG in torch, numpy and python.

        Args:
            seed (int): if None, will use a strong random seed.
        """
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        return seed
