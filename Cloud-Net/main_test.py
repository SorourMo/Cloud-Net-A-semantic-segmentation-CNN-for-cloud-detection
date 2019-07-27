
from __future__ import print_function
import os
import numpy as np
import cloud_net_model
from generators import mybatch_generator_prediction
import tifffile as tiff
import pandas as pd
from utils import get_input_image_names


def prediction():
    model = cloud_net_model.model_arch(input_rows=in_rows,
                                       input_cols=in_cols,
                                       num_of_channels=num_of_channels,
                                       num_of_classes=num_of_classes)
    model.load_weights(weights_path)

    print("\nExperiment name: ", experiment_name)
    print("Prediction started... ")
    print("Input image size = ", (in_rows, in_cols))
    print("Number of input spectral bands = ", num_of_channels)
    print("Batch size = ", batch_sz)

    imgs_mask_test = model.predict_generator(
        generator=mybatch_generator_prediction(test_img, in_rows, in_cols, batch_sz, max_bit),
        steps=np.ceil(len(test_img) / batch_sz))

    print("Saving predicted cloud masks on disk... \n")

    pred_dir = experiment_name + '_train_192_test_384'
    if not os.path.exists(os.path.join(PRED_FOLDER, pred_dir)):
        os.mkdir(os.path.join(PRED_FOLDER, pred_dir))

    for image, image_id in zip(imgs_mask_test, test_ids):
        image = (image[:, :, 0]).astype(np.float32)
        tiff.imsave(os.path.join(PRED_FOLDER, pred_dir, str(image_id)), image)


GLOBAL_PATH = 'path to 38-cloud dataset'
TRAIN_FOLDER = os.path.join(GLOBAL_PATH, 'Training')
TEST_FOLDER = os.path.join(GLOBAL_PATH, 'Test')
PRED_FOLDER = os.path.join(GLOBAL_PATH, 'Predictions')


in_rows = 384
in_cols = 384
num_of_channels = 4
num_of_classes = 1
batch_sz = 10
max_bit = 65535  # maximum gray level in landsat 8 images
experiment_name = "Cloud-Net_trained_on_38-Cloud_training_patches"
weights_path = os.path.join(GLOBAL_PATH, experiment_name + '.h5')


# getting input images names
test_patches_csv_name = 'test_patches_38-cloud.csv'
df_test_img = pd.read_csv(os.path.join(TEST_FOLDER, test_patches_csv_name))
test_img, test_ids = get_input_image_names(df_test_img, TEST_FOLDER, if_train=False)

prediction()
