import argparse
import os.path as osp
import mmcv
import numpy as np
import onnx
import onnxruntime as rt
import torch
from core.core import get_classes
from core.apis import show_result
from core.core import (build_model_from_cfg,
                       generate_inputs_and_wrap_model,
                        preprocess_example_input)


def pytorch2onnx(config_path,
                 checkpoint_path,
                 input_img,
                 input_shape,
                 opset_version=11,
                 show=False,
                 output_file='tmp.onnx',
                 verify=False,
                 normalize_cfg=None,
                 dataset='coco',
                 test_img=None):

    input_config = {
        'input_shape': input_shape,
        'input_path': input_img,
        'normalize_cfg': normalize_cfg
    }

    # prepare original model and meta for verifying the onnx model
    orig_model = build_model_from_cfg(config_path, checkpoint_path)
    # prepare the basic imgs and img_metas for model input
    one_img, one_meta = preprocess_example_input(input_config)
    # Prepare sample input and wrap model for ONNX export.
    model, tensor_data = generate_inputs_and_wrap_model(
        config_path, checkpoint_path, input_config)
    output_names = ['output']  # 定义输出节点名称.

    torch.onnx.export(
        model,
        tensor_data,
        output_file,
        input_names=['input'],  # 定义输入节点名称.
        output_names=output_names,
        export_params=True,
        keep_initializers_as_inputs=True,
        do_constant_folding=True,
        verbose=show,
        opset_version=opset_version)

    model.forward = orig_model.forward
    print(f'Successfully exported ONNX model: {output_file}')
    if verify:
        # 获取类别名
        # model.CLASSES = get_classes(dataset)
        # 获取类别数
        # num_classes = len(model.CLASSES)

        # check by onnx
        onnx_model = onnx.load(output_file)
        onnx.checker.check_model(onnx_model)
        if test_img is not None:
            input_config['input_path'] = test_img
            one_img, one_meta = preprocess_example_input(input_config)
            tensor_data = [one_img]

        # check the numerical value
        # get pytorch output
        pytorch_results = model(tensor_data, [[one_meta]], return_loss=False)
        pytorch_results = pytorch_results

        # get onnx output
        input_all = [node.name for node in onnx_model.graph.input]
        input_initializer = [node.name for node in onnx_model.graph.initializer]
        net_feed_input = list(set(input_all) - set(input_initializer))
        assert (len(net_feed_input) == 1)
        sess = rt.InferenceSession(output_file)
        onnx_outputs = sess.run(None, {net_feed_input[0]: one_img.detach().numpy()})
        output_names = [_.name for _ in sess.get_outputs()]
        output_shapes = [_.shape for _ in onnx_outputs]
        print(f'onnxruntime output names: {output_names}, output shapes: {output_shapes}')
        nrof_out = len(onnx_outputs)
        assert nrof_out > 0, 'Must have output'
        # 这里需要解析onnx_output的输出结果.
        cfg = mmcv.Config.fromfile(config_path)
        # 默认gpu为单卡.
        onnx_outputs = torch.Tensor(onnx_outputs[0]).to(device='cuda:0')
        onnx_results = model.head.get_results(onnx_outputs, one_meta, cfg.test_cfg)

        # visualize predictions
        if show:
            # show pytorch results
            show_result(model, one_meta['show_img'], pytorch_results, show=True, win_name="pytorch")
            # show onnx-runtime results
            show_result(model, one_meta['show_img'], onnx_results, show=True, win_name="onnx")

        # compare a part of result
        compare_pairs = [(onnx_results, pytorch_results)]
        for onnx_res, pytorch_res in compare_pairs:
            for o_res, p_res in zip(onnx_res, pytorch_res):
                np.testing.assert_allclose(
                    o_res,
                    p_res,
                    rtol=1e-03,
                    atol=1e-05,
                )
        print('The numerical values are the same between Pytorch and ONNX')


def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert model models to ONNX')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('--checkpoint',
                        default="convert_model/0113_line_model_v3_unet.pth",
                        help='checkpoint file')
    parser.add_argument('--input-img', type=str,
                        default="convert_model/seg_test.png",
                        help='Images for input')
    parser.add_argument('--show',
                        type=bool,
                        default=True,
                        help='show onnx graph')
    parser.add_argument('--output-file', type=str, default='convert_model/unet_line.onnx')
    parser.add_argument('--opset-version', type=int, default=11)
    parser.add_argument('--test-img',
                        type=str,
                        default="convert_model/seg_test.png",
                        help='Images for test')
    parser.add_argument('--dataset',
                        type=str,
                        default='SegDataset',
                        help='Dataset name for get some infos, CLASSES NAMES')
    parser.add_argument(
        '--verify',
        type=str,
        default=True,
        help='verify the onnx model output against pytorch output')
    parser.add_argument(
        '--shape',
        type=int,
        nargs='+',
        default=[256, 256],
        help='input image size')
    parser.add_argument(
        '--mean',
        type=float,
        nargs='+',
        default=[0., 0., 0.],
        help='mean value used for preprocess input data')
    parser.add_argument(
        '--std',
        type=float,
        nargs='+',
        default=[255., 255., 255.],
        help='variance value used for preprocess input data')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()

    # https://github.com/onnx/onnx/blob/master/docs/Operators.md
    assert args.opset_version == 11, 'MMDet only support opset 11 now'
    assert isinstance(args.input_img, str), "args.input_img must be not None"

    if len(args.shape) == 1:
        input_shape = (1, 3, args.shape[0], args.shape[0])
    elif len(args.shape) == 2:
        input_shape = (1, 3) + tuple(args.shape)
    else:
        raise ValueError('invalid input shape')

    assert len(args.mean) == 3 and len(args.std) == 3

    normalize_cfg = {'mean': args.mean, 'std': args.std}

    # convert model to onnx file
    pytorch2onnx(
        args.config,
        args.checkpoint,
        args.input_img,
        input_shape,
        opset_version=args.opset_version,
        show=args.show,
        output_file=args.output_file,
        verify=args.verify,
        normalize_cfg=normalize_cfg,
        dataset=args.dataset,
        test_img=args.test_img)
