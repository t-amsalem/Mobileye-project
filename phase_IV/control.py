from phase_IV.TFL_manager import TFL
import matplotlib.pyplot as plt


class Controller:

    def __init__(self):
        with open("data/pls_files/play_list.pls", "r+") as data:
            read_lines = data.readlines()
            image_path = [line[:-1] for line in read_lines]
            self.path_images = image_path[2:]
            self.tfl_manager = TFL(image_path[0], int(image_path[1]), int(image_path[1])+len(image_path[2:])-1)
            self.first_frame = image_path[1]

    def run(self) -> None:
        for i in range(len(self.path_images)):
            self.tfl_manager.run_product(self.path_images[i], i)
        plt.show(block=True)


def main():
    controller = Controller()
    controller.run()


if __name__ == '__main__':
    main()
