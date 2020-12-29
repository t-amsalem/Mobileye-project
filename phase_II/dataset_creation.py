import glob
import os
from os.path import join
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from phase_I.run_attention import find_tfl_lights


def pad_with_zeros(vector: np.ndarray, pad_width: tuple, iaxis, kwargs) -> None:
    vector[:pad_width[0]] = 0
    vector[-pad_width[1]:] = 0


def crop(image: np.ndarray, labels, x_coord: int, y_coord: int, num: int, data) -> None:
    crop_image = image[x_coord:x_coord + 81, y_coord:y_coord + 81]
    save_image(crop_image, data)
    save_label(labels, num)


def coord_to_crop(image_label: np.array, image: np.array, x_coord: list, y_coord: list,
                  first: int, last: int, labels, data) -> None:

    count1, count2, flag1, flag2, index_traffic, index_not_traffic = 0, 0, 0, 0, -1, -1

    if first > last:
        side = -1
    else:
        side = 1

    for i in range(first, last, side):
        if not flag1 or not flag2:

            if image_label[y_coord[i], x_coord[i]] == 19:
                if not flag1:
                    flag1 = 1
                    index_traffic = i

            else:
                if not flag2:
                    flag2 = 1
                    index_not_traffic = i
        else:
            break

    if flag1 and flag2:
        crop(image, labels, x_coord[index_traffic], y_coord[index_traffic], 1, data)
        crop(image, labels, x_coord[index_not_traffic], y_coord[index_not_traffic], 0, data)


def save_image(crop_image: np.ndarray, data) -> None:
    np.array(crop_image, dtype=np.uint8).tofile(data)


def save_label(labels, label: int) -> None:
    labels.write(label.to_bytes(1, byteorder='big', signed=False))


def change(data_file) -> None:
    with open(f".{data_file}/data_mirror.bin", "ab") as data:
        images = np.memmap(join(data_file + '/data.bin'), mode='r', dtype=np.uint8).reshape(
            [-1] + list((81, 81)) + [3])

        for img in images:
            img = img[..., ::-1, :]
            save_image(img, data)


def get_coord(image: np.ndarray) -> (list, list):
    x_red, y_red, x_green, y_green = find_tfl_lights(image, some_threshold=42)

    return x_red + x_green, y_red + y_green


def get_list_images(dir_name: str, image_path: str, postfix: str) -> list:
    list_images = []
    for root, dirs, files in os.walk(f'{image_path}/{dir_name}'):
        for dir in dirs:
            list_images = glob.glob(os.path.join(f'{image_path}/{dir_name}/' + dir, postfix))

    return list_images


def set_data(dir_name: str) -> None:
    image_path = 'data/leftImg8bit'
    label_path = 'data/gtFine'

    list_images = get_list_images(dir_name, image_path, '*_leftImg8bit.png')
    list_labels = get_list_images(dir_name, label_path, '*_gtFine_labelIds.png')

    for im_path, la_path in zip(list_images, list_labels):
        image = np.array(Image.open(im_path))
        label = np.array(Image.open(la_path))
        x_coord, y_coord = get_coord(image)

        if not len(x_coord):
            continue

        image = np.pad(image, 40, pad_with_zeros)[:, :, 40:43]
        with open(f"dataset/{dir_name}/data.bin", "ab") as data:
            with open(f"dataset/{dir_name}/labels.bin", "ab") as labels:
                coord_to_crop(label, image, x_coord, y_coord, 0, len(x_coord) - 1, labels, data)
                coord_to_crop(label, image, x_coord, y_coord, len(x_coord) - 1, 0, labels, data)


def data_set() -> None:
    set_data('val')
    set_data('train')


def main():
    data_set()


if __name__ == '__main__':
    main()

