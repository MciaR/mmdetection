# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os

import mmcv
from PIL import Image


def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert images to coco format without annotations')
    parser.add_argument('img_path', help='The root path of images')
    parser.add_argument(
        'classes', type=str, help='The text file name of storage class list')
    parser.add_argument(
        'out',
        type=str,
        help='The output annotation json file name, The save dir is in the '
        'same directory as img_path')
    parser.add_argument(
        'anno', type=str, help='The path of labels.json'
    )
    parser.add_argument(
        '-e',
        '--exclude-extensions',
        type=str,
        nargs='+',
        help='The suffix of images to be excluded, such as "png" and "bmp"')
    args = parser.parse_args()
    return args


def collect_image_infos(path, exclude_extensions=None):
    img_infos = []

    images_generator = mmcv.scandir(path, recursive=True)
    for image_path in mmcv.track_iter_progress(list(images_generator)):
        if exclude_extensions is None or (
                exclude_extensions is not None
                and not image_path.lower().endswith(exclude_extensions)):
            image_path = os.path.join(path, image_path)
            img_pillow = Image.open(image_path)
            img_info = {
                'filename': image_path,
                'width': img_pillow.width,
                'height': img_pillow.height,
            }
            img_infos.append(img_info)
    return img_infos


def cvt_to_coco_json(img_infos, classes, ann_file):
    image_id = 0
    coco = dict()
    coco['images'] = []
    coco['type'] = 'instance'
    coco['categories'] = []
    coco['annotations'] = []
    image_set = set()
    ann = mmcv.load(ann_file)
    ann_id = 0
    name2id_map = dict()

    for category_id, name in enumerate(classes):
        category_item = dict()
        category_item['supercategory'] = str('none')
        category_item['id'] = int(category_id)
        category_item['name'] = str(name)
        coco['categories'].append(category_item)

    for img_dict in img_infos:
        file_name = os.path.basename(img_dict['filename'])
        assert file_name not in image_set
        image_item = dict()
        image_item['id'] = int(image_id)
        image_item['file_name'] = str(file_name)
        image_item['height'] = int(img_dict['height'])
        image_item['width'] = int(img_dict['width'])
        coco['images'].append(image_item)
        image_set.add(file_name)
        name2id_map[file_name] = image_id

        image_id += 1

    for ann_dict in ann:
        file_name = ann_dict['image_id']
        anno_list = ann_dict['car_bbox']
        for _anno in anno_list:
            x1, y1, x2, y2 = _anno[:4]
            x, y, w, h = x1, y1, x2 - x1, y2 - y1
            area = w * h
            _category_id = _anno[4]

            ann_item = dict()
            ann_item['id'] = ann_id
            ann_item['image_id'] = name2id_map[file_name]
            ann_item['category_id'] = _category_id
            ann_item['segmentation'] = ''
            ann_item['area'] = area
            ann_item['bbox'] = [x, y, w, h]
            ann_item['iscrowd'] = 0

            coco['annotations'].append(ann_item)
            ann_id += 1

    return coco


def main():
    args = parse_args()
    assert args.out.endswith(
        'json'), 'The output file name must be json suffix'

    # 1 load image list info
    img_infos = collect_image_infos(args.img_path, args.exclude_extensions)

    # 2 convert to coco format data
    classes = mmcv.list_from_file(args.classes)
    coco_info = cvt_to_coco_json(img_infos, classes, args.anno)

    # 3 dump
    save_dir = os.path.join(args.img_path, '..', 'annotations')
    mmcv.mkdir_or_exist(save_dir)
    save_path = os.path.join(save_dir, args.out)
    mmcv.dump(coco_info, save_path)
    print(f'save json file: {save_path}')


if __name__ == '__main__':
    main()
