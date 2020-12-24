try:
    import os
    import json
    import glob
    import argparse
    import numpy as np
    from scipy import signal as sg
    import scipy.ndimage as ndimage
    from scipy.ndimage.filters import maximum_filter
    from PIL import Image
    import matplotlib.pyplot as plt

except ImportError:
    raise


def non_max_suppression(dim, result, c_image, color):
    x = []
    y = []
    image_height, image_width = result.shape[:2]

    for i in range(0, image_height - dim, dim):
        for j in range(0, image_width - dim, dim):
            index = np.argmax(result[i:i + dim, j:j + dim])
            a = np.amax(result[i:i + dim, j:j + dim])

            if a > 80:
                x.append(index // dim + i)
                y.append(index % dim + j)
                c_image[index // dim + i, index % dim + j] = color

    return x, y


def red_picture(im_red, filter_kernel, c_image):
    grad = sg.convolve2d(im_red, filter_kernel, boundary='symm', mode='same')
    result = ndimage.maximum_filter(grad, size=5)
    x_red, y_red = non_max_suppression(14, result, c_image, [255, 0, 0])

    return x_red, y_red


def green_picture(im_green, filter_kernel, c_image):
    grad = sg.convolve2d(im_green, filter_kernel, boundary='symm', mode='same')
    result = ndimage.maximum_filter(grad, size=5)
    x_green, y_green = non_max_suppression(18, result, c_image, [0, 255, 0])

    return x_green, y_green


def print_picture(im_green, c_image, result, grad):
    fig, (ax_orig, ax_mag, ax_ang) = plt.subplots(3, 1, figsize=(12, 30))
    ax_orig.imshow(im_green)
    ax_orig.set_title('Original')
    ax_orig.set_axis_off()
    ax_mag.imshow(np.absolute(grad))
    ax_mag.set_title('Gradient magnitude')
    ax_mag.set_axis_off()
    ax_ang.imshow(np.absolute(result))
    ax_ang.set_title('Gradient orientation')
    ax_ang.set_axis_off()
    fig.show()
    fig, (max_mag) = plt.subplots(1, 1, figsize=(6, 15))
    max_mag.set_axis_off()
    max_mag.imshow(np.absolute(c_image))
    fig.show()


def get_kernel():
    kernel = np.array([[-2 / 324, -2 / 324, -2 / 324, -2 / 324, -2 / 324, -2 / 324, -2 / 324, -2 / 324, -2 / 324, -2 / 324, -2 / 324, -2 / 324, -2 / 324, -2 / 324, -2 / 324, -2/324, -2/324, -2/324],
                       [-2 / 324, -2 / 324, -1 / 324, -1 / 324, -1 / 324, -1 / 324, -1 / 324, -1 / 324, -1 / 324, -1 / 324, -1 / 324, -1 / 324, -1 / 324, -1 / 324, -1 / 324, -1/324, -2/324, -2/324],
                       [-2 / 324, -1 / 324, -1 / 324, -1 / 324, -1 / 324, -1 / 324, -1 / 324, -1 / 324, -1 / 324, -1 / 324, -1 / 324, -1 / 324, -1 / 324, -1 / 324, -1 / 324, -1/324, -1/324, -2/324],
                       [-2 / 324, -1 / 324, -1 / 324, -1 / 324, -1 / 324, -1 / 324, -1 / 324, -1 / 324, -1 / 324, -1 / 324, -1 / 324, -1 / 324, -1 / 324, -1 / 324, -1 / 324, -1/324, -1/324, -2/324],
                       [-2 / 324, -1 / 324, -1 / 324, -1 / 324, -1 / 324, -2 / 324, -2 / 324, -1 / 324, -1 / 324, -1 / 324, -1 / 324, -1 / 324, -2 / 324, -1 / 324, -1 / 324, -1/324, -1/324, -2/324],
                       [-2 / 324, -1 / 324, -1 / 324, -1 / 324, -1 / 324, -2 / 324, -2 / 324, -2 / 324, -2 / 324, -2 / 324, -2 / 324, -2 / 324, -2 / 324, -1 / 324, -1 / 324, -1/324, -1/324, -2/324],
                       [-2 / 324, -1 / 324, -1 / 324, -1 / 324, -1 / 324, -2 / 324, 11 / 324, 11 / 324, 11 / 324, 11 / 324, 11 / 324, 11 / 324, -2 / 324, -1 / 324, -1 / 324, -1/324, -1/324, -2/324],
                       [-2 / 324, -1 / 324, -1 / 324, -1 / 324, -1 / 324, -2 / 324, 11 / 324, 11 / 324, 11 / 324, 11 / 324, 11 / 324, 11 / 324, -2 / 324, -1 / 324, -1 / 324, -1/324, -1/324, -2/324],
                       [-2 / 324, -1 / 324, -1 / 324, -1 / 324, -1 / 324, -2 / 324, 11 / 324, 11 / 324, 11 / 324, 11 / 324, 11 / 324, 11 / 324, -2 / 324, -1 / 324, -1 / 324, -1/324, -1/324, -2/324],
                       [-2 / 324, -1 / 324, -1 / 324, -1 / 324, -1 / 324, -2 / 324, 11 / 324, 11 / 324, 11 / 324, 11 / 324, 11 / 324, 11 / 324, -2 / 324, -1 / 324, -1 / 324, -1/324, -1/324, -2/324],
                       [-2 / 324, -1 / 324, -1 / 324, -1 / 324, -1 / 324, -2 / 324, 11 / 324, 11 / 324, 11 / 324, 11 / 324, 11 / 324, 11 / 324, -2 / 324, -1 / 324, -1 / 324, -1/324, -1/324, -2/324],
                       [-2 / 324, -1 / 324, -1 / 324, -1 / 324, -1 / 324, -2 / 324, 11 / 324, 11 / 324, 11 / 324, 11 / 324, 11 / 324, 11 / 324, -2 / 324, -1 / 324, -1 / 324, -1/324, -1/324, -2/324],
                       [-2 / 324, -1 / 324, -1 / 324, -1 / 324, -1 / 324, -2 / 324, -2 / 324, -2 / 324, -2 / 324, -2 / 324, -2 / 324, -2 / 324, -2 / 324, -1 / 324, -1 / 324, -1/324, -1/324, -2/324],
                       [-2 / 324, -1 / 324, -1 / 324, -1 / 324, -1 / 324, -2 / 324, -2 / 324, -1 / 324, -1 / 324, -1 / 324, -1 / 324, -2 / 324, -2 / 324, -1 / 324, -1 / 324, -1/324, -1/324, -2/324],
                       [-2 / 324, -1 / 324, -1 / 324, -1 / 324, -1 / 324, -1 / 324, -1 / 324, -1 / 324, -1 / 324, -1 / 324, -1 / 324, -1 / 324, -1 / 324, -1 / 324, -1 / 324, -1/324, -1/324, -2/324],
                       [-2 / 324, -1 / 324, -1 / 324, -1 / 324, -1 / 324, -1 / 324, -1 / 324, -1 / 324, -1 / 324, -1 / 324, -1 / 324, -1 / 324, -1 / 324, -1 / 324, -1 / 324, -1/324, -1/324, -2/324],
                       [-2 / 324, -2 / 324, -1 / 324, -1 / 324, -1 / 324, -1 / 324, -1 / 324, -1 / 324, -1 / 324, -1 / 324, -1 / 324, -1 / 324, -1 / 324, -1 / 324, -1 / 324, -1/324, -2/324, -2/324],
                       [-2 / 324, -2 / 324, -2 / 324, -2 / 324, -2 / 324, -2 / 324, -2 / 324, -2 / 324, -2 / 324, -2 / 324, -2 / 324, -2 / 324, -2 / 324, -2 / 324, -2 / 324, -2/324, -2/324, -2/324]])  # Gx + j*Gy
    return kernel


def find_tfl_lights(c_image: np.ndarray, **kwargs):
    im_red = c_image[:, :, 0]
    im_green = c_image[:, :, 1]

    filter_kernel = get_kernel()
    y_red, x_red = red_picture(im_red, filter_kernel, c_image)
    y_green, x_green = green_picture(im_green, filter_kernel, c_image)

    return x_red, y_red, x_green, y_green


def show_image_and_gt(image, objs, fig_num=None):
    plt.figure(fig_num).clf()
    plt.imshow(image)
    labels = set()

    if objs is not None:
        for o in objs:
            poly = np.array(o['polygon'])[list(np.arange(len(o['polygon']))) + [0]]
            plt.plot(poly[:, 0], poly[:, 1], 'r', label=o['label'])
            labels.add(o['label'])

        if len(labels) > 1:
            plt.legend()


def test_find_tfl_lights(image_path, json_path=None, fig_num=None):
    image = np.array(Image.open(image_path))
    if json_path is None:
        objects = None

    else:
        gt_data = json.load(open(json_path))
        what = ['traffic light']
        objects = [o for o in gt_data['objects'] if o['label'] in what]

    show_image_and_gt(image, objects, fig_num)
    red_x, red_y, green_x, green_y = find_tfl_lights(image, some_threshold=42)
    plt.plot(red_x, red_y, 'ro', color='r', markersize=4)
    plt.plot(green_x, green_y, 'ro', color='g', markersize=4)


def main(argv=None):
    parser = argparse.ArgumentParser("Test TFL attention mechanism")
    parser.add_argument('-i', '--image', type=str, help='Path to an image')
    parser.add_argument("-j", "--json", type=str, help="Path to json GT for comparison")
    parser.add_argument('-d', '--dir', type=str, help='Directory to scan images in')

    args = parser.parse_args(argv)
    default_base = 'data'

    if args.dir is None:
        args.dir = default_base
    flist = glob.glob(os.path.join(args.dir, '*_leftImg8bit.png'))

    for image in flist:
        json_fn = image.replace('_leftImg8bit.png', '_gtFine_polygons.json')
        if not os.path.exists(json_fn):
            json_fn = None
        test_find_tfl_lights(image, json_fn)

    if len(flist):
        print("You should now see some images, with the ground truth marked on them. Close all to quit.")

    else:
        print("Bad configuration?? Didn't find any picture to show")
    plt.show(block=True)


if __name__ == '__main__':
    main()
