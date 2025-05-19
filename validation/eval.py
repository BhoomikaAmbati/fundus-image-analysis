#Babyyy
import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import numpy as np
import pandas as pd
import cv2
from glob import glob
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras.utils import CustomObjectScope 
from tensorflow.keras.models import Model
from sklearn.metrics import accuracy_score, f1_score, jaccard_score, precision_score, recall_score
from train.metrics import dice_loss, dice_coef, iou
import time 

results_folder = "results/v2"
fileOut = os.path.join(results_folder,"result_activation_17_d4_layer_default") #folder name to save output
mode = ""           # 'rgb' to get rgb output, else default is greyscale image output

H = 512
W = 512


def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def read_image(path):
    x = cv2.imread(path, cv2.IMREAD_COLOR)
    x = cv2.resize(x, (W, H))
    ori_x = x
    x = x/255.0
    x = x.astype(np.float32)
    return ori_x, x

def read_mask(path):
    x = cv2.imread(path, cv2.IMREAD_GRAYSCALE)  ## (512, 512)
    x = cv2.resize(x, (W, H))
    ori_x = x
    x = x/255.0
    x = x.astype(np.int32)
    return ori_x, x

def load_data(path):
    x = sorted(glob(os.path.join(path, "image", "*.png")))
    y = sorted(glob(os.path.join(path, "mask", "*.png")))
    return x, y

def save_results(ori_x, ori_y, y_pred, save_image_path):
    line = np.ones((H, 10, 3)) * 255

    ori_y = np.expand_dims(ori_y, axis=-1)
    ori_y = np.concatenate([ori_y, ori_y, ori_y], axis=-1)

    y_pred = np.expand_dims(y_pred, axis=-1)
    y_pred = np.concatenate([y_pred, y_pred, y_pred], axis=-1) * 255

    cat_images = np.concatenate([ori_x, line, ori_y, line, y_pred], axis=1)
    cv2.imwrite(save_image_path, cat_images)

def save_results_rgb(ori_x, ori_y, y_pred, save_image_path):
    line = np.ones((y_pred.shape[0], 10, 3)) * 255

    pred_image = np.zeros((y_pred.shape[0], y_pred.shape[1], 3))
    _y_pred = y_pred[:, :]
    _ori_y = ori_y[:, :]
    pred_image[:, :, 0] = ((_y_pred > 0.5) & (_ori_y <= 128)) * 255
    pred_image[:, :, 1] = ((_y_pred > 0.5) & (_ori_y  > 128)) * 255
    pred_image[:, :, 2] = ((_ori_y  > 128) & (_y_pred <= 0.5 )) * 255

    print(" saving result", save_image_path)
    cv2.imwrite(save_image_path, pred_image)

def save_intermediate_output(d4, save_intermediate_dir, name):
    os.makedirs(save_intermediate_dir, exist_ok=True)  # Create directory if not exists

    # Ensure d4 has the correct shape (batch_size, H, W, channels)
    if len(d4.shape) == 3:  # If (512, 512, 64), add batch dimension
        d4 = np.expand_dims(d4, axis=0)

    for i in range(d4.shape[-1]):  # Loop through all 64 feature maps
        channel_image = d4[0, :, :, i]  # Extract single channel (shape: 512x512)

        # Normalize image to 0-255 for visualization
        channel_image = (channel_image - np.min(channel_image)) / (np.max(channel_image) - np.min(channel_image) + 1e-8) * 255
        channel_image = channel_image.astype(np.uint8)

        save_path = os.path.join(save_intermediate_dir, f"{name}_channel_{i}.png")
        cv2.imwrite(save_path, channel_image)  # Save each feature map as an image

if __name__ == "__main__":
    """ Save the results in this folder """
    outPut_dir = fileOut
    create_dir(outPut_dir)

    """ Load the model """
    with CustomObjectScope({'iou': iou, 'dice_coef': dice_coef, 'dice_loss': dice_loss}):
        full_model = tf.keras.models.load_model("model/model.h5")
    intermediate_layer_name = "activation_17"  # Change to the correct layer name
    intermediate_model = Model(inputs=full_model.input, outputs=full_model.get_layer(intermediate_layer_name).output)
    """ Load the dataset """
    #for layer in full_model.layers:
    #print(full_model.get_layer('conv2d_transpose_3'))
    dataset_path = os.path.join("Data","test")
    test_x, test_y = load_data(dataset_path)

    """ Make the prediction and calculate the metrics values """
    SCORE = []
    start = time.time()
    for x, y in tqdm(zip(test_x, test_y), total=len(test_x)):
        """ Extracting name """
        name = x.split("\\")[-1].split(".")[0]

        """ Read the image and mask """
        ori_x, x = read_image(x)
        ori_y, y = read_mask(y)
        #intermediate_output = intermediate_model.predict(np.expand_dims(x, axis=0))[0]
        """ Prediction """
        y_pred = full_model.predict(np.expand_dims(x, axis=0))[0]
        y_pred = y_pred > 0.5
        y_pred = y_pred.astype(np.int32)
        y_pred = np.squeeze(y_pred, axis=-1)
        print(np.shape(y_pred))
        """ Saving the images """
        save_image_path = f"{outPut_dir}/{name}.png"
        if(mode == 'rgb'):
            save_results_rgb(ori_x, ori_y, y_pred, save_image_path)
        else:
            save_results(ori_x, ori_y, y_pred, save_image_path)
        hiddenfname = "_hidden_layer_d4"
        hiddenFinal = name + hiddenfname
        """ save_intermediate_path = f"{outPut_dir}/{hiddenFinal}"
        save_intermediate_output(intermediate_output, save_intermediate_path, name) """

        """ Flatten the array """
        y = y.flatten()
        y_pred = y_pred.flatten()

        """ Calculate the metrics """
        acc_value = accuracy_score(y, y_pred)
        f1_value = f1_score(y, y_pred, labels=[0, 1], average="binary")
        jac_value = jaccard_score(y, y_pred, labels=[0, 1], average="binary")
        recall_value = recall_score(y, y_pred, labels=[0, 1], average="binary")
        precision_value = precision_score(y, y_pred, labels=[0, 1], average="binary")
        SCORE.append([name, acc_value, f1_value, jac_value, recall_value, precision_value])
    end = time.time()
    score = [s[1:] for s in SCORE]
    score = np.mean(score, axis=0)
    print(f"Accuracy: {score[0]:0.5f}")
    print(f"F1: {score[1]:0.5f}")
    print(f"Jaccard: {score[2]:0.5f}")
    print(f"Recall: {score[3]:0.5f}")
    print(f"Precision: {score[4]:0.5f}")
    print(f"Time taken by 64 channel model: {end - start}")

    """ Saving """
    df = pd.DataFrame(SCORE, columns=["Image", "Acc", "F1", "Jaccard", "Recall", "Precision"])
    df.to_csv("model/score_all_2.csv")