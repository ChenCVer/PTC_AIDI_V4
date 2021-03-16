# -*- coding: utf-8 -*-
from collections import OrderedDict
import functools
import gc
import cv2
import numpy as np
import torch
from enum import Enum

SUPPORT_ACTIVATE_FUNCTION = {
    'softmax': functools.partial(torch.softmax, dim=1),
    'sigmoid': functools.partial(torch.sigmoid),
}


class _ForwardType(Enum):
    HOOK = 0
    FORWARD = 1


def _get_names_dict(model):
    """Recursive walk to get names including path."""
    names = {}

    def _get_names(module, parent_name=""):
        for key, m in module.named_children():
            cls_name = str(m.__class__).split(".")[-1].split("'")[0]  # 获取m的类名
            num_named_children = len(list(m.named_children()))  # 获取m下的子模块
            if num_named_children > 0:
                name = parent_name + "." + key if parent_name else key
            else:
                name = parent_name + "." + cls_name + "_" + key if parent_name else key
            names[name] = m

            if isinstance(m, torch.nn.Module):
                _get_names(m, parent_name=name)

    _get_names(model)
    return names


class ClsModelOutputs(object):
    def __init__(self, net, summary):
        self._net = net
        self._summary = summary
        self.gradients = []
        self.feature = []

    def reset(self):
        self.gradients = []
        self.feature = []

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients.append(grad_output[0])

    def get_gradients(self):
        return self.gradients if len(self.gradients) <= 1 else [self.gradients[-1]]

    def save_forward(self, module, input, output):
        self.feature.append(output)

    def __call__(self, x, index=[-1], vis=False, save_gradient_flag=True):
        self.reset()
        handles = []
        for i in index:
            if i < 0:
                i = len(list(self._summary.keys())) + i
            if vis:
                print(list(self._summary.keys())[i])
            m = self._summary.get(list(self._summary.keys())[i])
            handles.append(m.register_forward_hook(self.save_forward))
            if save_gradient_flag:
                handles.append(m.register_backward_hook(self.save_gradient))
        output = self._net([x, ], img_metas=[x, ], return_loss=False)  # forward_test()
        # 移除hook
        for handle in handles:
            handle.remove()
        # 用于对付特殊的relu
        feature_map = self.feature if len(self.feature) <= 1 else [self.feature[-1]]
        return feature_map, output


class ClsBaseActivationMapping(object):
    def __init__(self, net, use_gpu=True):
        self._net = net
        self._use_gpu = use_gpu
        self._style = None
        self._summary = None
        self._hooks = None

    def set_hook_style(self, num_channel, input_shape, print_summary=True):
        self._num_channel = num_channel
        self._input_shape = input_shape
        self._print_model_structure(print_summary)  # 打印模型结构信息

    def set_forward_style(self, forward_func):
        raise NotImplementedError

    def run(self, img, feature_index=1, target=None, activate_fun='softmax'):
        raise NotImplementedError

    def _print_model_structure(self, print_summary=False):
        import torchsummaryX as summaryX
        self._net.apply(self._add_model_forward(_get_names_dict(self._net)))
        extra = torch.zeros((1, self._num_channel, self._input_shape[2], self._input_shape[3]))
        if self._use_gpu:
            extra = extra.cuda()
        with torch.no_grad():
            # 这里走base/forward_test()方法.
            # todo: 这里暂时没有什么好方法, 先这样, 后续再改掉.
            self._net([extra, ], img_metas=[extra, ], return_loss=False)  # forward_test()
        # 删除hook，防止影响
        for handle in self._hooks:
            handle.remove()
        if print_summary:
            summaryX.summary(self._net, extra)

    def _add_model_forward(self, names_dict):
        _summary = OrderedDict()
        hooks = []
        self._summary = _summary
        self._hooks = hooks

        def register_hook(module):
            def hook(module, inputs, outputs):
                module_idx = len(_summary)
                for name, item in names_dict.items():
                    if item == module:
                        key = "{}_{}".format(module_idx, name)
                _summary[key] = module

            if not module._modules:
                hooks.append(module.register_forward_hook(hook))

        return register_hook


class FeatureMapVis(ClsBaseActivationMapping):
    def __init__(self, net, use_gpu=True):
        super(FeatureMapVis, self).__init__(net, use_gpu)

    def set_hook_style(self, num_channel, input_shape, print_summary=False, post_process_func=None):
        super().set_hook_style(num_channel, input_shape, print_summary)
        self._style = _ForwardType.HOOK
        self._post_process_func = post_process_func
        self._model_out = ClsModelOutputs(self._net, self._summary)

    def set_forward_style(self, forward_func):
        self._forward_func = forward_func
        self._style = _ForwardType.FORWARD

    def run(self, img, feature_index=1, target=None, activate_fun='softmax'):
        assert self._style is not None, 'You need to select the run mode,' \
                                        'you must call set_hook_style() or set_forward_style() one of them'

        data = np.copy(img)
        if self._use_gpu:
            data = torch.from_numpy(np.array([data])).cuda()
        else:
            data = torch.from_numpy(np.array([data]))
        data = data.permute(0, 3, 1, 2)  # 通道在前
        output_fm = {}
        if self._style == _ForwardType.HOOK:
            feature_map, output = self._model_out(data, [feature_index], save_gradient_flag=False)
            if self._post_process_func is not None:
                feature_map, _ = self._post_process_func(feature_map, output)
        elif self._style == _ForwardType.FORWARD:
            feature_map = self._forward_func(data)
        else:
            raise NotImplementedError
        output_fm['feature_map'] = feature_map
        return output_fm


class ClsActivationMappingVis(ClsBaseActivationMapping):
    def __init__(self, net, use_gpu=True):
        super(ClsActivationMappingVis, self).__init__(net, use_gpu)
        self._style = None

    def set_hook_style(self, num_channel, input_shape, print_summary=True, post_process_func=None):
        super().set_hook_style(num_channel, input_shape, print_summary)
        self._style = _ForwardType.HOOK
        self._post_process_func = post_process_func
        self._cls_model_out = ClsModelOutputs(self._net, self._summary)

    def set_forward_style(self, forward_func):
        self._forward_func = forward_func
        self._style = _ForwardType.FORWARD

    @torch.no_grad()
    def run(self, img, target=None, feature_index=1, activate_fun='softmax'):
        assert self._style is not None, 'You need to select the run mode,' \
                                        'you must call set_hook_style() or set_forward_style() one of them'
        data = np.copy(img)
        if self._use_gpu:
            data = torch.from_numpy(np.array([data])).cuda()
        else:
            data = torch.from_numpy(np.array([data]))
        data = data.permute(0, 3, 1, 2)  # 通道在前
        if self._style == _ForwardType.HOOK:
            feature_map, output = self._cls_model_out(data, [feature_index], save_gradient_flag=False)
            if self._post_process_func is not None:
                feature_map, output = self._post_process_func(feature_map, output)
            return self._result_process(activate_fun, feature_map, output, target)
        elif self._style == _ForwardType.FORWARD:
            feature_map, output = self._forward_func(data)
            return self._result_process(activate_fun, feature_map, output, target)

    def _result_process(self, activate_fun, feature_map, output, target):
        if isinstance(feature_map, list) and len(feature_map) == 1:
            feature_map = feature_map[0]
        if activate_fun:
            assert activate_fun in SUPPORT_ACTIVATE_FUNCTION.keys()
            activate_fun = SUPPORT_ACTIVATE_FUNCTION[activate_fun]
            output = activate_fun(output)
        assert isinstance(output, torch.Tensor)
        if self._use_gpu:
            pred = output.cpu().data.numpy()
        else:
            pred = output.data.numpy()
        output = {'feature_map': feature_map, 'output': pred}
        if target is not None:
            pred_index = np.argmax(pred)
            if np.equal(pred_index, target):
                pred_flag = True
            else:
                pred_flag = False
            output['pred_flag'] = pred_flag
        return output


class ClsClassActivationMappingVis(ClsBaseActivationMapping):
    def __init__(self, net, use_gpu=True):
        super(ClsClassActivationMappingVis, self).__init__(net, use_gpu)
        self._style = None

    def set_hook_style(self, num_channel, input_shape, print_summary=True, post_process_func=None):
        super().set_hook_style(num_channel, input_shape, print_summary)
        self._style = _ForwardType.HOOK
        self._post_process_func = post_process_func
        self._cls_model_out = ClsModelOutputs(self._net, self._summary)

    def set_forward_style(self, forward_func, input_shape=None):
        self._input_shape = input_shape
        self._forward_func = forward_func
        self._style = _ForwardType.FORWARD

    @torch.no_grad()
    def run(self, img, target, feature_index=1, activate_fun='softmax'):
        assert self._style is not None, 'You need to select the run mode,' \
                                        'you must call set_hook_style() or set_forward_style() one of them'
        assert target is not None
        data = np.copy(img)
        if self._use_gpu:
            data = torch.from_numpy(np.array([data])).cuda()
        else:
            data = torch.from_numpy(np.array([data]))
        data = data.permute(0, 3, 1, 2)  # 通道在前
        if self._style == _ForwardType.HOOK:
            feature_map, output = self._cls_model_out(data, [feature_index], save_gradient_flag=False)
            if self._post_process_func is not None:
                feature_map, output = self._post_process_func(feature_map, output)
            return self._result_process(activate_fun, feature_map, output, target)

        elif self._style == _ForwardType.FORWARD:
            assert self._forward_func is not None
            feature_map, output = self._forward_func(data)
            return self._result_process(activate_fun, feature_map, output, target)

    def _result_process(self, activate_fun, feature_map, output, target):
        if isinstance(feature_map, list) and len(feature_map) == 1:
            feature_map = feature_map[0]
        if activate_fun:
            assert activate_fun in SUPPORT_ACTIVATE_FUNCTION.keys()
            activate_fun = SUPPORT_ACTIVATE_FUNCTION[activate_fun]
            output = activate_fun(output)
        assert isinstance(output, torch.Tensor)
        params = list(self._net.parameters())
        if self._use_gpu:
            weight_softmax = np.squeeze(params[-2].cpu().data.numpy())
            feature_map = feature_map[0].cpu().data.numpy()
            pred = output.cpu().data.numpy()
        else:
            weight_softmax = np.squeeze(params[-2].data.numpy())
            feature_map = feature_map[0].data.numpy()
            pred = output.data.numpy()
        nc, h, w = feature_map.shape
        cam = weight_softmax[target].dot(feature_map.reshape((nc, h * w)))
        cam = cam.reshape(h, w)
        cam = cam - np.min(cam)
        cam_img = cam / np.max(cam)
        cam_img = np.uint8(255 * cam_img)
        if self._input_shape is not None:
            cam_img = cv2.resize(cam_img, tuple(self._input_shape))
        output = {'cam_img': cam_img, 'output': pred}
        if target is not None:
            pred_index = np.argmax(pred)
            if np.equal(pred_index, target):
                pred_flag = True
            else:
                pred_flag = False
            output['pred_flag'] = pred_flag
        return output


class ClsGradClassActivationMappingVis(ClsBaseActivationMapping):
    def __init__(self, net, use_gpu=True):
        super(ClsGradClassActivationMappingVis, self).__init__(net, use_gpu)
        self._style = None

    def set_hook_style(self, num_channel, input_shape, print_summary=False, post_process_func=None):
        super().set_hook_style(num_channel, input_shape, print_summary)
        self._style = _ForwardType.HOOK
        self._post_process_func = post_process_func
        self._cls_model_out = ClsModelOutputs(self._net, self._summary)

    def run(self, img, target=None, feature_index=55, class_id=None, activate_fun='softmax'):
        # TODO: 每一个网络对应的最后一个卷积层的编号都不一样, 这个feature_index后期需要支持自动获取, 而不是写死
        assert self._style is not None, 'You need to select the run mode,' \
                                        'you must call set_hook_style()'
        if self._style == _ForwardType.HOOK:
            feature_map, output = self._cls_model_out(img, [feature_index], save_gradient_flag=True)
            if self._post_process_func is not None:
                feature_map, output = self._post_process_func(feature_map, output)

            if isinstance(feature_map, list) and len(feature_map) == 1:
                feature_map = feature_map[0]
            if activate_fun:
                assert activate_fun in SUPPORT_ACTIVATE_FUNCTION.keys()
                activate_fun = SUPPORT_ACTIVATE_FUNCTION[activate_fun]
                output = activate_fun(output)
            assert isinstance(output, torch.Tensor)

            if self._use_gpu:
                pred = output.cpu().data.numpy()
            else:
                pred = output.data.numpy()
            if class_id is None:
                index = np.argmax(pred)
            else:
                index = class_id

            one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)  # [1,class_num]
            one_hot[0][index] = 1  # 掩码，用于得到特定类的梯度
            one_hot = torch.from_numpy(one_hot).requires_grad_(True)
            # 当预测值和label一致时候，梯度很小，反之很大，所以梯度的具体值没有意义
            if self._use_gpu:
                one_hot = torch.sum(one_hot.cuda() * output)  # 模拟Loss操作
            else:
                one_hot = torch.sum(one_hot * output)

            one_hot.backward()  # 反向传播.

            if self._use_gpu:
                grads_val = self._cls_model_out.get_gradients()[-1].cpu().data.numpy()  # 对应特征层的误差项(敏感度), 并不是梯度
                target_map = feature_map.cpu().data.numpy()[0, :]  # 去掉batch
            else:
                grads_val = self._cls_model_out.get_gradients()[-1].data.numpy()
                target_map = feature_map.data.numpy()[0, :]

            if len(grads_val.shape) > 1:
                weights = np.mean(grads_val, axis=(2, 3))[0, :]  # 对梯度在hw维度求平均，得到权重 [channel,]
            else:
                weights = grads_val
            cam = np.zeros(target_map.shape[1:], dtype=np.float32)  # 输出特征图h，w

            for i, w in enumerate(weights):
                # 通道方向权重和
                cam += w * target_map[i, :, :]  # 输出特征图值乘上梯度权重值，相当于突出强激活区域

            # 只关注那些对预测结果起到positive influence on the class
            cam = np.maximum(cam, 0)  # 模拟relu的作用，符合论文公式
            cam = cv2.resize(cam, (self._input_shape[2], self._input_shape[3]))
            # 必须要归一化
            cam = cam - np.min(cam)
            cam = cam / (np.max(cam) - np.min(cam) + 1e-32)
            cam_img = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
            output = {'cam_img': cam_img, 'output': pred}
            if target is not None:
                if np.equal(index, target):
                    pred_flag = True
                else:
                    pred_flag = False
                output['pred_flag'] = pred_flag

            torch.cuda.empty_cache()
            gc.collect()

            return output
        else:
            raise NotImplementedError