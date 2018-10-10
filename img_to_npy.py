import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
import os
import argparse


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', help='path to img folder', required=True)
    parser.add_argument('--output', help='name of output folder', default='data.npy')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arguments()
    data = []
    output = args.output
    images_folder = args.input
    images = os.listdir(images_folder)
    for img in images:
        data.append(np.array(Image.open(images_folder + img), dtype='uint8'))

    np.save(args.output, data)

    # # visually testing our output
    # img_array = np.load(args.output + '.npy')
    #
    # for im in img_array:
    #     plt.imshow(im)
    #     plt.show()
