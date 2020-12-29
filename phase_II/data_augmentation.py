import os
import glob

import numpy as np
import matplotlib.pyplot as plt
import imgaug as ia
import imgaug.augmenters as iaa
import random
from PIL import Image
from phase_I.run_attention import find_tfl_lights
from phase_II.dataset_creation import pad_with_zeros


def load_tfl_data(url: str, suffix: str) -> list:
    data_list = []

    for subdir, dirs, files in os.walk(url):
        for directory in dirs:
            data_list += glob.glob(os.path.join(url + '\\' + directory, suffix))
    return data_list


def load_data(dir_set: str) -> dict:
    image_suffix = '*_leftImg8bit.png'
    label_suffix = '*_labelIds.png'

    url_image = "D:\\Mobileye\\Mobileye\\phase_II\\data\\leftImg8bit\\"
    url_label = "D:\\Mobileye\\Mobileye\\phase_II\\data\\gtFine\\"

    return {'images': load_tfl_data(url_image + dir_set, image_suffix),
            'labels': load_tfl_data(url_label + dir_set, label_suffix)}


def get_random_pixel(pixels: list) -> (int, int):
    random_p = random.choice(pixels)
    index_of_random_p = pixels.index(random_p)

    return random_p, index_of_random_p


def call_write_to_bin_file(x_coords: list, y_coords: list, label_image: np.ndarray,
                           padded_image: np.ndarray, data, labels, data_name: str) -> None:

    count = 0
    pixels_of_tfl = [p for p in zip(x_coords, y_coords) if label_image[p[0], p[1]] == 19]
    pixels_of_not_tfl = [p for p in zip(x_coords, y_coords) if label_image[p[0], p[1]] != 19]

    while pixels_of_tfl and pixels_of_not_tfl and count < 3:
        count += 1

        rand_tfl, index_rand_of_tfl = get_random_pixel(pixels_of_tfl)
        pixels_of_tfl = pixels_of_tfl[:index_rand_of_tfl] + pixels_of_tfl[index_rand_of_tfl + 1:]
        cropped_image = padded_image[rand_tfl[0]:rand_tfl[0] + 81, rand_tfl[1]:rand_tfl[1] + 81, 40:43]
        write_image_to_binary_file(data, cropped_image, data_name)
        write_label_to_binary_file(labels, 1, data_name)
        print("TFL\n")

        rand_not_tfl, index_rand_of_not_tfl = get_random_pixel(pixels_of_not_tfl)
        pixels_of_not_tfl = pixels_of_not_tfl[:index_rand_of_not_tfl] + pixels_of_not_tfl[index_rand_of_not_tfl + 1:]
        cropped_image = padded_image[rand_not_tfl[0]:rand_not_tfl[0] + 81, rand_not_tfl[1]:rand_not_tfl[1] + 81, 40:43]
        write_image_to_binary_file(data, cropped_image, data_name)
        write_label_to_binary_file(labels, 0, data_name)
        print("NOT TFL\n")


def darken_image(image: np.ndarray) -> np.ndarray:
    contrast = iaa.GammaContrast(gamma=2.0)
    contrast_image = contrast.augment_images(image)

    return contrast_image


def bright_image(image: np.ndarray) -> np.ndarray:
    contrast = iaa.GammaContrast(gamma=0.5)
    contrast_image = contrast.augment_images(image)

    return contrast_image


def add_noise(image: np.ndarray) -> np.ndarray:
    gaussian_noise = iaa.AdditiveGaussianNoise(10, 20)
    noised_image = gaussian_noise.augment_image(image)

    return noised_image


def write_image_to_binary_file(file_name, image: np.ndarray, data_name: str) -> None:
    image.tofile(file_name, sep="", format="%s")

    if data_name == 'train':
        flipped_image = np.fliplr(image)
        flipped_image.tofile(file_name, sep="", format="%s")
        plt.imshow(flipped_image)

        darken = darken_image(image)
        darken.tofile(file_name, sep="", format="%s")
        ia.imshow(darken)

        bright = bright_image(image)
        bright.tofile(file_name, sep="", format="%s")
        ia.imshow(bright)

        noise_image = add_noise(image)
        noise_image.tofile(file_name, sep="", format="%s")
        plt.imshow(noise_image)


def write_label_to_binary_file(file_name, label: int, data_name: str) -> None:
    file_name.write(label.to_bytes(1, byteorder='big', signed=False))

    if data_name == 'train':
        file_name.write(label.to_bytes(1, byteorder='big', signed=False))
        file_name.write(label.to_bytes(1, byteorder='big', signed=False))
        file_name.write(label.to_bytes(1, byteorder='big', signed=False))
        file_name.write(label.to_bytes(1, byteorder='big', signed=False))


def increasing_the_data(image_path: str, label_path: str, dir_name: str) -> None:
    with open(f"dataset/{dir_name}/data.bin", "ab") as data:
        with open(f"dataset/{dir_name}/labels.bin", "ab") as labels:
            image = np.array(Image.open(image_path))
            padded_image = np.pad(image, 40, pad_with_zeros)
            label = np.array(Image.open(label_path))
            red_x, red_y, green_x, green_y = find_tfl_lights(image)
            call_write_to_bin_file(red_y + green_y, red_x + green_x, label, padded_image, data, labels, dir_name)
            print("Finish")


def main():
    dir_name_t = 'train'
    tfl_data_t = load_data(dir_name_t)

    for image_path, label_path in zip(*tfl_data_t.values()):
        increasing_the_data(image_path, label_path, dir_name_t)

    dir_name_v = 'val'
    tfl_data_v = load_data(dir_name_v)

    for image_path, label_path in zip(*tfl_data_v.values()):
        increasing_the_data(image_path, label_path, dir_name_v)


if __name__ == '__main__':
    main()
