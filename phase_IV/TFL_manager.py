from PIL import Image
from tensorflow.python.keras.models import load_model
from phase_I.run_attention import find_tfl_lights
from phase_II.dataset_creation import pad_with_zeros
from phase_III.SFM_standAlone import *
from phase_III import SFM


class TFL:
    def __init__(self, pkl_path: str, prev_frame_id: int, curr_frame_id: int):
        with open(pkl_path, 'rb') as pkl_file:
            data = pickle.load(pkl_file, encoding='latin1')
            self.focal = data['flx']
            self.pp = data['principle_point']
            self.list_EM = []
            self.prev_frame = None
            self.init_EM(data, prev_frame_id, curr_frame_id)

    def init_EM(self, data: pickle, prev_frame_id: int, curr_frame_id: int) -> None:
        for j in range(prev_frame_id+1, curr_frame_id+1):
            EM = np.eye(4)

            for i in range(prev_frame_id, curr_frame_id):
                EM = np.dot(data['egomotion_' + str(i) + '-' + str(i + 1)], EM)
            self.list_EM.append(EM)

    def visual_images(self, image_path: str, current_frame: FrameContainer, auxiliary_traffic: list,
                      candidate: np.array, auxiliary: list, dist: list) -> None:

        fig, (ax_source_lights, ax_is_tfl, ax_dist) = plt.subplots(3, 1, figsize=(12, 30))
        ax_source_lights.imshow(current_frame.img)
        x_, y_ = candidate[:, 1], candidate[:, 0]
        ax_source_lights.scatter(y_, x_, c=auxiliary, s=1)
        ax_source_lights.set_title('Source lights')

        ax_is_tfl.imshow(current_frame.img)
        x_, y_ = np.array(current_frame.traffic_light)[:, 1], np.array(current_frame.traffic_light)[:, 0]
        ax_is_tfl.scatter(y_, x_, c=auxiliary_traffic, s=1)
        ax_is_tfl.set_title('Traffic lights')

        ax_dist.set_title('Distances of tfl')
        if dist != list():
            x_cord, y_cord, image_dist = self.get_coords_of_tfl(current_frame, image_path)
            ax_dist.imshow(image_dist)
            for i in range(len(x_cord)):
                ax_dist.text(x_cord[i], y_cord[i], r'{0:.1f}'.format(dist[i]), color='r')

        fig.show()

    def get_coords_of_tfl(self, current_frame: FrameContainer, image_path: str) -> (list, list, np.array):
        image = np.array(Image.open(image_path))
        curr_p = current_frame.traffic_light
        x_cord = [p[0] for p in curr_p]
        y_cord = [p[1] for p in curr_p]

        return x_cord, y_cord, image

    def run_product(self, image_path: str, index: int) -> None:
        current_frame = FrameContainer(image_path)
        current_frame.EM = self.list_EM[index-1]

        # part1
        candidate, auxiliary = self.detection_of_source_lights(image_path)

        # part2
        traffic, auxiliary_traffic = self.is_tfl(candidate, auxiliary, image_path)
        current_frame.traffic_light = traffic

        # part3
        dists = []
        if index != 0 and traffic != [] and self.prev_frame.traffic_light != []:
            dists = self.get_tfl_distance(current_frame)

        self.visual_images(image_path, current_frame, auxiliary_traffic, candidate, auxiliary, dists)
        self.prev_frame = current_frame

    def get_tfl_distance(self, current_frame: FrameContainer) -> list:
        curr_container = SFM.calc_TFL_dist(self.prev_frame, current_frame, self.focal, self.pp)

        return curr_container.traffic_lights_3d_location[:, 2]

    def is_tfl(self, candidate: np.array, auxiliary: list, image_path: str) -> (list, list):
        traffic = []
        auxiliary_traffic = []
        model = load_model("model.h5")
        image = np.array(Image.open(image_path))
        image_pad = np.pad(image, 40, pad_with_zeros)[:, :, 40:43]

        for i, coord in enumerate(candidate):
            cropped_images = self.crop_image(image_pad, coord)
            result = network(cropped_images, model)
            if result:
                traffic.append(coord)
                auxiliary_traffic.append(auxiliary[i])

        return traffic, auxiliary_traffic

    def detection_of_source_lights(self, image_path: str) -> (np.array, list):
        x_red, y_red, x_green, y_green = find_tfl_lights(np.array(Image.open(image_path)), some_threshold=42)
        x_coord = x_red + x_green
        y_coord = y_red + y_green
        candidate = np.array([[x_coord[i], y_coord[i]] for i in range(len(x_coord))])
        auxiliary = ['r' if i < len(x_red) else 'g' for i in range(len(x_coord))]

        return candidate, auxiliary

    def get_source_lights(self, image_path: str, candidate: np.array, auxiliary: list) -> np.array:
        image = np.array(Image.open(image_path))
        x, y = 0, 1

        for i in range(len(candidate)):
            image[candidate[i][y], candidate[i][x]] = auxiliary[i]

        return image

    def crop_image(self, image: np.ndarray, coord: np.ndarray) -> np.ndarray:
        cropped_image = image[coord[1]:coord[1] + 81, coord[0]:coord[0] + 81]

        return cropped_image


class FrameContainer(object):
    def __init__(self, img_path: str):
        self.img = plt.imread(img_path)
        self.traffic_light = []
        self.traffic_lights_3d_location = []
        self.EM = []
        self.corresponding_ind = []
        self.valid = []


def network(images: np.ndarray, model) -> bool:
    crop_shape = (81, 81)
    predictions = model.predict(images.reshape([-1] + list(crop_shape) + [3]))

    return predictions[0][1] > 0.97
