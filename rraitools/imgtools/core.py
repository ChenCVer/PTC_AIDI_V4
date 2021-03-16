import os
import cv2
import numpy as np

from ..filetools import FileHelper as flt

CV2_INTER_DICT = {
    'nearest': cv2.INTER_NEAREST,
    'linear': cv2.INTER_LINEAR,
    'cubic': cv2.INTER_CUBIC
}

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def float32_to_uint8(img):  # [0,1] 转换为 [0,255]
    if img.dtype in ['float32', "float64"]:
        if np.max(img) <= 1.0:
            img[img > 0] = 255
            img = np.uint8(img)
        else:
            img = np.uint8(img)
    return img


class ImageHelper(object):
    """
    Provide some basic and significant operations of image
    """

    @staticmethod
    def read_img(img_path, mode='BGR'):
        """
        Loads an image from a file. Additionally, opencv loads an image data as 'BGR',
        this function supports three image data modes, 'BGR', 'RGB' and 'GRAY'.
        You can specified loading type through parameter 'mode'.

        :param img_path: string: name of file to be loaded.
        :param mode: string: which data type is wanted.

        :return: img: numpy.ndarray.
        """
        if not os.path.exists(img_path):
            print(img_path)
            raise RuntimeError('img path do not exists!')

        img_bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)  # opencv读进来的是BGR格式
        if mode == 'RGB':
            return ImageHelper.convert_bgr_to_rgb(img_bgr)

        elif mode == 'BGR':
            return img_bgr

        elif mode == 'GRAY':
            return ImageHelper.convert_bgr_to_gray(img_bgr)

        else:
            raise Exception('Not support' + mode)

    @staticmethod
    def convert_bgr_to_rgb(img_bgr):
        """
        Converts an image from bgr space to rgb space.
        """
        assert isinstance(img_bgr, np.ndarray)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        return img_rgb

    @staticmethod
    def convert_rgb_to_bgr(img_rgb):
        """
        Converts an image from rgb space to bgr space.
        """
        assert isinstance(img_rgb, np.ndarray)
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2RGB)
        return img_bgr

    @staticmethod
    def convert_bgr_to_gray(img, keep_dim=False):
        """
        Converts an image from bgr space to gray space.
        Optional parameter 'keep_dim' controls channel of return image.

        :param img: numpy.ndarray: bgr space.
        :param keep_dim: if True, return img has same channel as input img, else, channel will reduce 1.

        """
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if keep_dim:
            gray_img = gray_img[..., None]

        return gray_img

    @staticmethod
    def convert_gray_to_bgr(img):
        """
        Converts an image from gray space to bgr space.
        """
        if img.ndim == 2:
            img = img[..., None]

        bgr_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        return bgr_img

    @staticmethod
    def show_img(imgs, window_names=None, wait_time_ms=0, is_merge=False, row_col_num=(1, -1)):
        """
        Displays an image or a list of images in specified windows or self-initiated windows.
        You can also control display wait time by parameter 'wait_time_ms'.
        Additionally, this function provides an optional parameter 'is_merge' to
        decide whether to display all imgs in a particular window 'merge'.
        Besides, parameter 'row_col_num' supports user specified merge format.
        Notice, specified format must be greater than or equal to imgs number.

        :param imgs: numpy.ndarray or list.
        :param window_names: specified or None, if None, function will create different windows as '1', '2'.
        :param wait_time_ms: display wait time.
        :param is_merge: whether to merge all images.
        :param row_col_num: merge format. default is (1, -1), image will line up to show.
                            example=(2, 5), images will display in two rows and five columns.
        """
        if not isinstance(imgs, list):
            imgs = [imgs]

        if window_names is None:
            window_names = list(range(len(imgs)))
        else:
            if not isinstance(window_names, list):
                window_names = [window_names]
            assert len(imgs) == len(window_names), 'window names does not match images!'

        if is_merge:
            merge_imgs = ImageHelper.merge_imgs(imgs, row_col_num)

            cv2.namedWindow('merge', 0)
            cv2.imshow('merge', merge_imgs)
        else:
            for img, win_name in zip(imgs, window_names):
                if img is None:
                    continue
                win_name = str(win_name)
                cv2.namedWindow(win_name, 0)
                cv2.imshow(win_name, img)

        cv2.waitKey(wait_time_ms)

    @staticmethod
    def write_img(img, save_path, debug=False):
        """
        Saves an image to a specified file.
        """
        flt.make_dirs(save_path, is_file=True, debug=debug)
        if isinstance(img, np.ndarray):
            cv2.imwrite(save_path, img)
        else:
            raise Exception('data not support!')

    @staticmethod
    def merge_imgs(imgs, row_col_num):
        """
        Merges all input images as an image with specified merge format.
        """

        from ..visualtools import random_color

        length = len(imgs)
        row, col = row_col_num

        assert row > 0 or col > 0, 'row and col cannot be negative at same time!'
        color = random_color(rgb=True).astype(np.float64)

        for img in imgs:
            cv2.rectangle(img, (0, 0), (img.shape[1], img.shape[0]), color)

        if row_col_num[1] < 0 or length < row:
            merge_imgs = np.hstack(imgs)
        elif row_col_num[0] < 0 or length < col:
            merge_imgs = np.vstack(imgs)
        else:
            assert row * col >= length, 'Imgs overboundary, not enough windows to display all imgs!'

            fill_img_list = [np.zeros(imgs[0].shape, dtype=np.uint8)] * (row * col - length)
            imgs.extend(fill_img_list)
            merge_imgs_col = []
            for i in range(row):
                start = col * i
                end = col * (i + 1)
                merge_col = np.hstack(imgs[start: end])
                merge_imgs_col.append(merge_col)

            merge_imgs = np.vstack(merge_imgs_col)

        return merge_imgs

    @staticmethod
    def convert_img_to_uint8(img, multip=255):
        """
        Converts an image to uint8 type.

        :param img numpy.ndarray
        :param multip: float: multipler
        """
        assert 0 < multip <= 255
        img = img * multip
        img[img > 0] = 255
        img = np.uint8(img)
        return img

    @staticmethod
    def convert_img_to_float32(img, divide=1.0):
        """
        Convert an image to flaot32 type.

        :param img: numpy.ndarray
        :param divide: float: divider
        """
        assert 0 < divide <= 255
        img = img / divide
        img = np.float32(img)
        return img

    @staticmethod
    def convert_binary_img(img, lower_thres=0.1, upper_thres=255, is_inv=False):
        """
        Applies a fixed-level threshold to each array element.
        The function applies fixed-level thresholding to a multiple-channel array. The function is typically
        used to get a bi-level (binary) image out of a grayscale image ( #compare could be also used for
        this purpose) or for removing a noise, that is, filtering out pixels with too small or too large
        values. After binary, image values will in (0, 255).
        Parameter 'is_inv' provides options whether to inverse binary.

        :param img:input array (multiple-channel, 8-bit or 32-bit floating point).
        :param lower_thres: lower threshold value.
        :param upper_thres: upper threshold value.
        :param is_inv: bool: whether to inverse binary.

        :return: numpy.ndarray: binary image.
        """
        assert img is not None, 'input data is none, please check your data!'
        assert len(img.shape) in (2, 3), 'input shape does not support!'
        assert upper_thres > lower_thres, 'upper threshold must be greater than lower threshold'

        if len(img.shape) == 2:
            img = np.expand_dims(img, axis=2)

        if len(img.shape) == 3 and img.shape[2] == 3:
            img = ImageHelper.convert_bgr_to_gray(img.copy())

        if is_inv:
            mode = cv2.THRESH_BINARY_INV
        else:
            mode = cv2.THRESH_BINARY

        _, img = cv2.threshold(img, lower_thres, upper_thres, mode)

        return img

    @staticmethod
    def get_size_hw(img):
        """
        Get height ans width of an image
        """
        assert img is not None, 'input img is none!'
        height, width = img.shape[:2]
        return [height, width]

    @staticmethod
    def resize_img(img, target_size=None, fx=None, fy=None, interpolation='nearest'):
        """
        Resizes an image. The function resize resizes the image src down to or up to the specified size.
         Note that the initial dst type or size are not taken into account.

        :param img: input image.
        :param target_size: output image size; if it equals zero, it is computed as:
        \f[\texttt{dsize = Size(round(fx*src.cols), round(fy*src.rows))}\f]
        Either dsize or both fx and fy must be non-zero.
        :param fx: fx scale factor along the horizontal axis; when it equals 0, it is computed as
    .   \f[\texttt{(double)dsize.width/src.cols}\f].
        :param fy: fy scale factor along the vertical axis; when it equals 0, it is computed as
    .   \f[\texttt{(double)dsize.height/src.rows}\f].
        :param interpolation: interpolation interpolation method. Supports: 'linear', 'nearest', 'cubic'.

        :return: an image with target size
        """
        assert isinstance(interpolation, str)

        if target_size is not None:
            assert isinstance(target_size, (list, tuple))
            return cv2.resize(img, tuple(target_size), interpolation=CV2_INTER_DICT[interpolation])
        else:
            assert fx is not None and fy is not None, 'without specific target size, fx and fy cannot be None!'
            assert fx != 0 and fy != 0, 'In this mode, fx and fy cannot be zero!'
            return cv2.resize(img, target_size, fx=fx, fy=fy, interpolation=CV2_INTER_DICT[interpolation])

    @staticmethod
    def rotate_img(img, rotate_center_xy, rotate_angle, flag='nearest',
                   is_keep_origin_shape=False, is_return_matrix=False):
        """
        Rotate an image with specified point and angle.
        As rotating may change the image shape to save original information,
        this function provides a parameter 'is_keep_origin_shape', if True, returns an image with same shape as input image.
        Considering rotate matrix is important, parameter 'is_return_matrix' provides to return it or not.

        :param img: input image.
        :param rotate_center_xy: which point to rotate with.
        :param rotate_angle: rotation angle.
        :param flag: interpolation interpolation method. Supports: 'linear', 'nearest', 'cubic'.
        :param is_keep_origin_shape: whether to keep output image's shape same with input image's.
        :param is_return_matrix: whether to return rotate matrix or not.

        :return: output image.
        """
        assert img is not None, 'input data cannot be none!'
        [height, width] = ImageHelper.get_size_hw(img)
        assert rotate_center_xy[0] <= width and rotate_center_xy[1] <= height, 'rotate center is out of image boundary!'

        rotate_mat = cv2.getRotationMatrix2D(rotate_center_xy, rotate_angle, 1.0)
        cos_val = np.abs(rotate_mat[0, 0])
        sin_val = np.abs(rotate_mat[0, 1])
        new_width = int(height * sin_val + width * cos_val)
        new_height = int(height * cos_val + width * sin_val)
        rotate_mat[0, 2] += (new_width / 2.) - rotate_center_xy[0]
        rotate_mat[1, 2] += (new_height / 2.) - rotate_center_xy[1]

        img = cv2.warpAffine(img, rotate_mat, (new_width, new_height), flags=CV2_INTER_DICT[flag])
        if is_keep_origin_shape:
            img = ImageHelper.crop_img(img, [new_width // 2, new_height // 2, width, height])

        if is_return_matrix:
            return img, rotate_mat
        else:
            return img

    @staticmethod
    def pad_img(img, top, bottom, left, right, pad_value):
        """
        Forms a border around an image.The function copies the source image into the middle of the destination image.
        The areas to the left, to the right, above and below the copied source image will be filled with extrapolated pixels.

        :param img: input image.
        :param top: int: pixel expanded up.
        :param bottom: int: pixel expanded down.
        :param left: int: pixel expanded left.
        :param right: int: pixel expanded right.
        :param pad_value: float.

        :return: output image.
        """
        assert img is not None, 'input data is none, please check your input!'
        pad_img = img.copy()
        # 如果输出比输入小，则不进行pad操作，否则pad到输出一样大
        if any([top, bottom, left, right]):
            pad_img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=pad_value)
        return pad_img

    @staticmethod
    def crop_img(img, crop_pts_xywh, auto_fit=False, is_return_crop_pt=False):
        """
        Crop an image with specified  point and height and width. Notice, given point should be the center point,
        height and width are desired size of output image. Parameter 'auto_fit' handles with out of boundary, if True,
        this function returns specified size but with different center point, else, it will raise exceptions.
        Also, if auto_fit is turn on, 'is_return_crop_pt' should be True, it returns real crop center point.

        :param img: input image.
        :param crop_pts_xywh: list: [pt_x, pt_y, h, w].
        :param auto_fit: bool: whether to fit out of boundary.
        :param is_return_crop_pt: whether to return crop center point.

        :return: output image.
        """

        assert img is not None, 'input img cannot be none!'
        assert len(crop_pts_xywh) == 4, 'does not match length!'

        h, w = img.shape[: 2]

        crop_center_xy = crop_pts_xywh[0:2]
        crop_size_hw = crop_pts_xywh[2:4][::-1]

        assert 0 < crop_size_hw[0] <= h and 0 < crop_size_hw[1] <= w, 'crop size cannot be greater than image shape!'

        crop_start_row = crop_center_xy[1] - crop_size_hw[0] // 2
        crop_end_row = crop_center_xy[1] - crop_size_hw[0] // 2 + crop_size_hw[0]
        crop_start_col = crop_center_xy[0] - crop_size_hw[1] // 2
        crop_end_col = crop_center_xy[0] - crop_size_hw[1] // 2 + crop_size_hw[1]

        if auto_fit:
            if crop_start_row < 0:
                crop_start_row = 0
                crop_end_row = crop_size_hw[0]

            if crop_end_row > h:
                crop_end_row = h
                crop_start_row = crop_end_row - crop_size_hw[0]

            if crop_start_col < 0:
                crop_start_col = 0
                crop_end_col = crop_size_hw[1]

            if crop_end_col > w:
                crop_end_col = w
                crop_start_col = crop_end_col - crop_size_hw[1]
        else:
            assert crop_center_xy >= [0, 0], 'center points cannot be negative!'
            assert crop_start_row >= 0 and crop_end_row <= h, 'out of boundary!'
            assert crop_start_col >= 0 and crop_end_col <= w, 'out of boundary!'

        img = img[crop_start_row:crop_end_row, crop_start_col:crop_end_col]

        if is_return_crop_pt:
            return img, [crop_start_row, crop_start_col]
        else:
            return img

    @staticmethod
    def normalize(img_bgr, mean_bgr, std_bgr):
        mean_bgr = np.array(mean_bgr, np.float32)
        std_bgr = np.array(std_bgr, np.float32)
        if all(mean_bgr <= 1) and all(std_bgr <= 1):
            img = np.asarray(img_bgr).astype(np.float32) / 255.
            img_bgr = (img - mean_bgr) / std_bgr
        else:
            img = np.asarray(img_bgr).astype(np.float32)
            img_bgr = (img - mean_bgr) / std_bgr
        return img_bgr

    @staticmethod
    def warp_prespective_img(img, src_list_xy, dst_size_xy, dst_size_hw, flag='nearest', is_return_matrix=False):
        """
        The function warpPerspective transforms the source image using two given points lists.
        This function calculates a perspective transform from four pairs of the corresponding points.
        The transform will apply on src img. Parameter 'is_return matrix' decides whether to return transform matrix.

        :param img: input image.
        :param src_list_xy: list:coordinates of quadrangle vertices in the source image.
        :param dst_size_xy: list: dst coordinates of the corresponding quadrangle vertices in the destination image.
        :param dst_size_hw: list: dsize size of the output image.
        :param flag: interpolation interpolation method. Supports: 'linear', 'nearest', 'cubic'.
        :param is_return_matrix: whether to return transform matrix.

        :return: img, (matrix)
        """
        assert len(src_list_xy) == len(dst_size_xy)
        assert tuple(dst_size_hw) > (0, 0)
        pts_src = np.float32(src_list_xy)
        pts_dst = np.float32(dst_size_xy)
        matrix = cv2.getPerspectiveTransform(pts_src, pts_dst)
        dst = cv2.warpPerspective(img, matrix, dst_size_hw[::-1], flags=CV2_INTER_DICT[flag])
        if is_return_matrix:
            return dst, matrix
        else:
            return dst

    @staticmethod
    def calc_contours(img):
        """
        Finds contours in a binary image. Input image shoule be an 8-bit single-channel image.
        Non-zero pixels are treated as 1's. Zero pixels remain 0's, so the image is treated as binary.
        So be sure input image is appropriate type. You can use 'convert_to_binary_img' first.

        :param img: input image.

        :return: tuple: all contours extracted in image.
        """

        assert img is not None, 'input data is none, please check your data!'
        assert len(img.shape) == 2, 'input shape does not support!'
        # 以下代码有问题
        # assert np.unique(img).all() in (0, 255), 'input data is not binary, please use binary_img to convert!'

        try:  # cv3
            _, contours, _ = cv2.findContours(image=img.copy(), mode=cv2.RETR_EXTERNAL,
                                              method=cv2.CHAIN_APPROX_SIMPLE)
        except:  # cv2
            contours, _ = cv2.findContours(image=img.copy(), mode=cv2.RETR_EXTERNAL,
                                           method=cv2.CHAIN_APPROX_SIMPLE)

        return contours

    @staticmethod
    def draw_contours(img, contours, index=-1, color=(255, 255, 255), thickness=-1, on_origin=True):
        """
        Draws contours outlines or filled contours with specified color.
        The function draws contour outlines in the image if thickness > 0
        or fills the area bounded by the contours if thickness < 0.
        Also, 'on_origin' provides an option to display on raw image or its copy.

        :param img: input image.
        :param contours: tuple.
        :param index: which contour is to draw. -1 means all.
        :param color: tuple.
        :param thickness: int.
        :param on_origin: bool.
        """
        assert img is not None, 'input image is none!'
        if not on_origin:
            draw_img = img.copy()
        else:
            draw_img = img

        cv2.drawContours(draw_img, contours, index, color=color, thickness=thickness)

    @staticmethod
    def calc_contour_area(contour, img_shape_hw):
        """
        Calculate area of single contour.

        :param contour: numpy.ndarray
        :param img_shape_hw:list:[height, width].

        :return: float: area.
        """
        assert contour is not None, 'contour cannot be none!'
        img_temp = np.zeros(img_shape_hw[:2], dtype=np.uint8)
        cv2.drawContours(img_temp, [contour], -1, 255, -1)
        area = np.sum(np.float32(img_temp) / 255.)
        return area

    @staticmethod
    def calc_arclength(contour, img_shape_hw):
        """
        Calculate arclength of single contour.

        :param contour: numpy.ndarray.
        :param img_shape_hw: list:[height, width].
        :return: float: arclength.
        """
        assert contour is not None, 'contour cannot be none!'
        temp = np.zeros(img_shape_hw[:2], dtype=np.uint8)
        cv2.drawContours(temp, [contour], -1, 255, 1)
        arc_length = np.sum(np.float32(temp) / 255.)
        return arc_length

    @staticmethod
    def clean_mask_edge(mask, area_thre_pix=100, distance_thre_pix=15):
        """
        Removes corner case in a mask. Input image should be a binary image.
        Tiny defect will mostly be erased by area threshold.
        Edge defect will mostly be erased by distance threshold.

        :param mask: numpy.ndarray.
        :param area_thre_pix: float.
        :param distance_thre_pix: float.
        :return: more intuitive mask.
        """
        assert mask is not None, 'Mask is none, please check your data!'

        mask_h, mask_w = mask.shape[:2]
        cnts = ImageHelper.calc_contours(mask)
        for cnt in cnts:
            x, y, w, h = cv2.boundingRect(cnt)
            cx = x + w / 2
            cy = y + h / 2
            distance = np.min([cx, mask_w - cx, cy, mask_h - cy])
            area = ImageHelper.calc_contour_area(cnt, mask.shape)
            if area < area_thre_pix or distance < distance_thre_pix:
                mask[y - 1:y + h + 1, x - 1:x + w + 1] = 0
        return mask

    @staticmethod
    def overlap_crop_img(img, shape_hw=(128, 128), overlap_hw=(0, 0)):
        """
        Sliding window overlap crop an image with specified shape.

        :param img: input image.
        :param shape_hw: tuple or list: crop_size.
        :param overlap_hw: tuple or list: Number of pixels overlapped while cropping.

        :return: list: piece_list: crop small imgs list
                 list: pts_list: crop start points list
                 frame_map: crop grid.
        """
        assert img is not None, 'input image cannot be none!'
        assert (0, 0) <= tuple(overlap_hw) <= tuple(img.shape[: 2]), 'overlap is out of boundary!'

        h, w = img.shape[:2]
        frame_map = np.zeros(img.shape[:2], dtype=np.float32)

        if h <= shape_hw[0] and w <= shape_hw[1]:
            return [img], [[0, 0]], np.ones(img.shape[:2], dtype=np.float32)

        piece_list = []
        pts_list = []
        stride_h = shape_hw[0] - overlap_hw[0]
        stride_w = shape_hw[1] - overlap_hw[1]

        for h_n in range(0, h - shape_hw[0] + stride_h, stride_h):
            for w_n in range(0, w - shape_hw[1] + stride_w, stride_w):
                h_start = h_n
                h_end = h_n + shape_hw[0]
                w_start = w_n
                w_end = w_n + shape_hw[1]
                if h_n + shape_hw[0] > h:
                    h_start = h - shape_hw[0]
                    h_end = h
                if w_n + shape_hw[1] > w:
                    w_start = w - shape_hw[1]
                    w_end = w

                piece = img[h_start:h_end, w_start:w_end]
                frame_map[h_start:h_end, w_start:w_end] += 1
                pts = [h_start, w_start]
                piece_list.append(piece)
                pts_list.append(pts)

        return piece_list, pts_list, frame_map

    @staticmethod
    def stitch_imgs_max(piece_list, pts_list, src_shape_hw, frame_map=None):
        """
        Stitch input images to a specified-sized image with start points list.
        If there exists overlap between two images, this function takes maximum value as overlap value.

        :param piece_list: list: a list of small image.
        :param pts_list: list: a list of start points.
        :param src_shape_hw: output image size.
        :param map: reserved parameter.

        :return: output image.
        """
        assert len(piece_list) == len(pts_list)
        if len(piece_list) == 1:
            return piece_list[0]

        img = np.zeros(src_shape_hw, dtype=np.float32)
        piece_size_h = piece_list[0].shape[0]
        piece_size_w = piece_list[0].shape[1]

        for piece, pt in zip(piece_list, pts_list):
            piece_of_image = img[pt[0]:pt[0] + piece_size_h, pt[1]:pt[1] + piece_size_w]

            img[pt[0]:pt[0] + piece_size_h, pt[1]:pt[1] + piece_size_w] = \
                np.where(piece_of_image > piece, piece_of_image, piece)

        return img

    @staticmethod
    def stitch_imgs_mean(piece_list, pts_list, src_shape_hw, frame_map):
        """
        Stitch input images to a specified-sized image with start points list.
        If there exists overlap between two images, this function takes mean value as overlap value.

        :param piece_list: list: a list of small image.
        :param pts_list: list: a list of start points.
        :param src_shape_hw: output image size.
        :param map: crop grid.

        :return: output image.
        """
        assert len(piece_list) == len(pts_list)
        assert map is not None, 'map cannot be none!'
        if len(piece_list) == 1:
            return piece_list[0]
        img = np.zeros(src_shape_hw, dtype=np.float32)
        piece_size_h = piece_list[0].shape[0]
        piece_size_w = piece_list[0].shape[1]
        assert len(img.shape) == 3
        channel = img.shape[2]

        for piece, pt in zip(piece_list, pts_list):
            img[pt[0]:pt[0] + piece_size_h, pt[1]:pt[1] + piece_size_w] = \
                img[pt[0]:pt[0] + piece_size_h, pt[1]:pt[1] + piece_size_w] + piece

        if len(img.shape) == 2:
            return img / frame_map
        else:
            for i in range(channel):
                img[:, :, i] = img[:, :, i] / frame_map
            return img

    @staticmethod
    def calc_cam(img, mask):

        heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
        heatmap = np.float32(heatmap) / 255

        cam = heatmap + np.float32(img)
        cam = cam / np.max(cam)

        return cam

    @staticmethod
    def calc_mean_std_bgr(file_path, include_path_suffix, exclude_path_suffix_list=None):
        files_list = flt.get_file_path_list(file_path, include_path_suffix, exclude_path_suffix_list)
        if len(files_list) == 0:
            print('---no data---')
            return
        bgr_mean_sum = np.zeros(shape=3)
        bgr_std_sum = np.zeros(shape=3)
        for _, files in enumerate(files_list):
            # print(files)
            img = ImageHelper.read_img(files)
            bgr_mean = np.mean(img, axis=(0, 1))
            bgr_std = np.std(img, axis=(0, 1))
            bgr_mean_sum += bgr_mean
            bgr_std_sum += bgr_std

        return bgr_mean_sum / len(files_list), bgr_std_sum / len(files_list)

    @staticmethod
    def get_cnts(src):
        img = src.copy()
        img = float32_to_uint8(img)
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        _, mask = cv2.threshold(img.copy(), 0.1, 255, cv2.THRESH_BINARY)
        try:  # cv3
            binary, contours, hierarchy = cv2.findContours(image=mask.copy(), mode=cv2.RETR_EXTERNAL,
                                                           method=cv2.CHAIN_APPROX_SIMPLE)
        except:  # cv2
            contours, hierarchy = cv2.findContours(image=mask.copy(), mode=cv2.RETR_EXTERNAL,
                                                   method=cv2.CHAIN_APPROX_SIMPLE)

        return contours

