import torch
import argparse
from core.core import (build_model_from_cfg,
                       generate_inputs_and_wrap_model,
                       preprocess_example_input)


def pytorch2libtorch(config_path,
                     checkpoint_path,
                     input_img,
                     input_shape,
                     show=False,
                     output_file='tmp.pt',
                     verify=False,
                     normalize_cfg=None,
                     dataset='coco',
                     test_img=None):
    input_config = {
        'input_shape': input_shape,
        'input_path': input_img,
        'normalize_cfg': normalize_cfg
    }

    # prepare original model and meta for verifying the libtorch model
    orig_model = build_model_from_cfg(config_path, checkpoint_path)
    one_img, one_meta = preprocess_example_input(input_config)
    model, tensor_data = generate_inputs_and_wrap_model(
        config_path, checkpoint_path, input_config)

    libtorch_model = torch.jit.trace(model, [tensor_data, ])

    libtorch_model.save(output_file)
    print(f'Successfully exported libtorch model: {output_file}')


def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert MMDetection models to libtorch')
    parser.add_argument('config',
                        help='The path of a model config file especially test')
    parser.add_argument('--checkpoint',
                        default="convert_model/0113_line_model_v3_unet.pth",
                        help='The path of a model checkpoint file')
    parser.add_argument('--input-img', type=str,
                        default="convert_model/seg_test.png",
                        help='The path of an input image for tracing and conversion')
    parser.add_argument('--show', action='store_true',
                        help='Determines whether to print the architecture of the exported model.')
    parser.add_argument('--output-file', type=str,
                        default="unet_seg_line.pt",
                        help='The path of output libtorch model')
    parser.add_argument('--test-img', type=str, default=None,
                        help='The path of an image to verify the exported ONNX model. '
                             'By default, it will be set to `None`, meaning it will use '
                             '`--input-img` for verification')
    parser.add_argument('--dataset', type=str, default='coco',
                        help='The dataset name for the input model. If not specified, '
                             'it will be set to `coco`')
    parser.add_argument('--verify', action='store_true',
                        help='Determines whether to verify the correctness of an exported model.')
    parser.add_argument('--shape', type=int, nargs='+', default=[512, 512],
                        help='The height and width of input tensor to the model.')
    parser.add_argument('--mean', type=float, nargs='+', default=[0., 0., 0.],
                        help='Three mean values for the input image. If not specified!')
    parser.add_argument('--std', type=float, nargs='+', default=[255., 255., 255.],
                        help='Three std values for the input image. If not specified.')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()

    if len(args.shape) == 1:
        input_shape = (1, 3, args.shape[0], args.shape[0])

    elif len(args.shape) == 2:
        input_shape = (1, 3) + tuple(args.shape)
    else:
        raise ValueError('invalid input shape')

    assert len(args.mean) == 3 and len(args.std) == 3

    normalize_cfg = {'mean': args.mean, 'std': args.std}

    # convert model to libtorch file
    pytorch2libtorch(
        args.config,
        args.checkpoint,
        args.input_img,
        input_shape,
        show=args.show,
        output_file=args.output_file,
        verify=args.verify,
        normalize_cfg=normalize_cfg,
        dataset=args.dataset,
        test_img=args.test_img)
