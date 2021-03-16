import os
from argparse import ArgumentParser
from core.apis import inference_detector, init_detector, show_result


def main():
    parser = ArgumentParser()
    parser.add_argument('--config', type=str, help='Config file',
                        default="../configs/segmentation/seg_offline_magnetic_single.py")
    parser.add_argument('--img_folder_path', type=str,  help='Image file',
                        default="/home/cxj/Desktop/data/electronic_datasets/0222数据标注/test/")
    parser.add_argument('--checkpoint', type=str, help='Checkpoint file',
                        default="/home/cxj/Desktop/data/electronic_datasets/0222数据标注/ptc_unet_0225.pth")
    parser.add_argument('--save_path', type=str, help='save result path',
                        default="/home/cxj/Desktop/results")
    parser.add_argument('--show_img', type=bool, help='save result path', default=False)
    parser.add_argument('--device', default='cuda:0', help='Device used for inference')

    args = parser.parse_args()
    # build the model from a config file and a checkpoint file
    model = init_detector(args.config, args.checkpoint, device=args.device)
    # test imgs
    img_path_list = [os.path.join(args.img_folder_path, x) for x in os.listdir(args.img_folder_path)]
    for img_path in img_path_list:
        result = inference_detector(model, img_path)
        # show the results
        img_name = os.path.split(img_path)[1]
        save_path = os.path.join(args.save_path, img_name)
        show_result(model, img_path, result, show=args.show_img, out_file=save_path)


if __name__ == '__main__':
    main()
