import random
from skimage.io import imread
from skimage.transform import resize
import numpy as np
from augmentation import flipping_img_and_msk, rotate_cclk_img_and_msk, rotate_clk_img_and_msk, zoom_img_and_msk

"""
Some lines borrowed from https://www.kaggle.com/petrosgk/keras-vgg19-0-93028-private-lb
"""


def mybatch_generator_train(zip_list, img_rows, img_cols, batch_size, shuffle=True, max_possible_input_value=65536):
    number_of_batches = np.ceil(len(zip_list) / batch_size)
    if shuffle:
        random.shuffle(zip_list)
    counter = 0

    while True:
        if shuffle:
            random.shuffle(zip_list)

        batch_files = zip_list[batch_size * counter:batch_size * (counter + 1)]
        image_list = []
        mask_list = []

        for file, mask in batch_files:

            image_red = imread(file[0])
            image_green = imread(file[1])
            image_blue = imread(file[2])
            image_nir = imread(file[3])

            mask = imread(mask)

            image = np.stack((image_red, image_green, image_blue, image_nir), axis=-1)

            image = resize(image, (img_rows, img_cols), preserve_range=True, mode='symmetric')
            mask = resize(mask, (img_rows, img_cols), preserve_range=True, mode='symmetric')

            rnd_flip = np.random.randint(2, dtype=int)
            rnd_rotate_clk = np.random.randint(2, dtype=int)
            rnd_rotate_cclk = np.random.randint(2, dtype=int)
            rnd_zoom = np.random.randint(2, dtype=int)

            if rnd_flip == 1:
                image, mask = flipping_img_and_msk(image, mask)

            if rnd_rotate_clk == 1:
                image, mask = rotate_clk_img_and_msk(image, mask)

            if rnd_rotate_cclk == 1:
                image, mask = rotate_cclk_img_and_msk(image, mask)

            if rnd_zoom == 1:
                image, mask = zoom_img_and_msk (image, mask)

            mask = mask[..., np.newaxis]
            mask /= 255
            image /= max_possible_input_value
            image_list.append(image)
            mask_list.append(mask)

        counter += 1
        image_list = np.array(image_list)
        mask_list = np.array(mask_list)
        yield (image_list, mask_list)

        if counter == number_of_batches:
            if shuffle:
                random.shuffle(zip_list)
            counter = 0


def mybatch_generator_validation(zip_list, img_rows, img_cols, batch_size, shuffle=False, max_possible_input_value=65536):
    number_of_batches = np.ceil(len(zip_list) / batch_size)
    if shuffle:
        random.shuffle(zip_list)
    counter = 0

    while True:

        batch_files = zip_list[batch_size * counter:batch_size * (counter + 1)]
        image_list = []
        mask_list = []

        for file, mask in batch_files:
            image_red = imread(file[0])
            image_green = imread(file[1])
            image_blue = imread(file[2])
            image_nir = imread(file[3])

            mask = imread(mask)

            image = np.stack((image_red, image_green, image_blue, image_nir), axis=-1)

            image = resize(image, (img_rows, img_cols), preserve_range=True, mode='symmetric')
            mask = resize(mask, (img_rows, img_cols), preserve_range=True, mode='symmetric')

            mask = mask[..., np.newaxis]
            mask /= 255
            image /= max_possible_input_value
            image_list.append(image)
            mask_list.append(mask)

        counter += 1
        image_list = np.array(image_list)
        mask_list = np.array(mask_list)
        yield (image_list, mask_list)

        if counter == number_of_batches:
            counter = 0


def mybatch_generator_prediction(tstfiles, img_rows, img_cols, batch_size, max_possible_input_value=65536):
    number_of_batches = np.ceil(len(tstfiles) / batch_size)
    counter = 0

    while True:

        beg = batch_size * counter
        end = batch_size * (counter + 1)
        batch_files = tstfiles[beg:end]
        image_list = []

        for file in batch_files:

            image_red = imread(file[0])
            image_green = imread(file[1])
            image_blue = imread(file[2])
            image_nir = imread(file[3])

            image = np.stack((image_red, image_green, image_blue, image_nir), axis=-1)

            image = resize (image, ( img_rows, img_cols), preserve_range=True, mode='symmetric')


            image /= max_possible_input_value
            image_list.append(image)

        counter += 1
        # print('counter = ', counter)
        image_list = np.array(image_list)

        yield (image_list)

        if counter == number_of_batches:
            counter = 0

