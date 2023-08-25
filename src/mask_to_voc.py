import io
import base64
import json

import PIL
from PIL import Image
import numpy as np
from skimage.measure import find_contours, approximate_polygon


def encode_image(path, format_='PNG'):
    print('----------------encoded_image, ', type(path))

    if isinstance(path, str):
        image_pil = Image.open(path)
    elif isinstance(path, np.ndarray):
        # print('----------------path, ')
        # print(type(path), path.shape, path.min(), path.max(), path.dtype)
        image_pil = Image.fromarray(path)
        # print('----------------image_pil, ', type(image_pil))
    else:
        image_pil = path
    # print('----------------image_pil, ', type(image_pil))
    with io.BytesIO() as f:
        image_pil.save(f, format=format_)
        f.seek(0)
        image_data = f.read()
        image_data = base64.b64encode(image_data).decode("utf-8")
    # print('----------------image_data, ', type(image_data))
    return image_data


def get_contours_simplified(mask, tolerance=2.5):
    """获取voc json注释文件
    mask : numpy array or PIL.Image
        必须. 提取其中的
    """

    mask = np.array(mask)

    contours_simplified = []

    for contour in find_contours(mask, 0):

        coords = approximate_polygon(contour, tolerance)

        points = []
        for x, y in zip(coords[:, 1], coords[:, 0]):
            points.append([x, y])
        contours_simplified.append(points)

        print("Number of coordinates: origin: {} simplefied: {}".format(len(contour), len(coords)))

    return contours_simplified


def get_voc(mask, image=None, tolerance=2.5, label="null"):
    print('------------------------------1, get_voc')
    if isinstance(mask, str):
        image_path = mask
        mask = PIL.Image.open(image_path)
    else:
        image_path = ''

    if isinstance(image, str):
        image_path = image
        image = PIL.Image.open(image_path)
    else:
        image_path = ''

    print("------------------get_contours_simplified")
    contours_simplified = get_contours_simplified(mask, tolerance=tolerance)
    print('------------------------------2, get_voc')

    print(type(image))
    print(type(mask))
    if isinstance(mask, np.ndarray):
        height, width = mask.shape[0], mask.shape[1]
    else:
        width, height = mask.size

    print('------------------------------3, get_voc')

    if image is not None:
        print(type(image))
        image_data = encode_image(image)
    else:
        print(type(mask))
        image_data =  encode_image(mask)

    result = {
        "version": "4.5.4",
        "flags": {},
        "shapes": [],
        "imagePath": image_path,
        "imageData": image_data,
        "imageHeight": height,
        "imageWidth": width
    }
    print('------------------------------4, get_voc')
    #     print(result['shapes'])
    for i, points in enumerate(contours_simplified):

        c = {
            "label": label,
            "points": points,
            "group_id": "null",
            "shape_type": "polygon",
            "flags": {}
        }

        result['shapes'].append(c)
    print('------------------------------5, get_voc')
    return result


def save_voc_json(result, save_to='result.json'):
    with open(save_to, 'w') as fp:
        json.dump(result, fp, sort_keys=True, indent=2)


# 用法
# result = get_voc(mask='001_msk.png', image='001.png', tolerance=0.5, label='lung')
# save_voc_json(result)