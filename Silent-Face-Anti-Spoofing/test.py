# -*- coding: utf-8 -*-
# @Time : 20-6-9 下午3:06
# @Author : zhuying
# @Company : Minivision
# @File : test.py
# @Software : PyCharm

import os
import cv2
import numpy as np
import argparse
import warnings
import time

from src.anti_spoof_predict import AntiSpoofPredict
from src.generate_patches import CropImage
from src.utility import parse_model_name

warnings.filterwarnings('ignore')


def test(image_data, model_dir, device_id):
    model_test = AntiSpoofPredict(device_id)
    image_cropper = CropImage()

    # Read image data from the file object
    image_data.seek(0)
    image_array = np.frombuffer(image_data.read(), dtype=np.uint8)

    # Check if the image array is empty
    if len(image_array) == 0:
        return {"error": "Empty image buffer"}

    # Decode the image array
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

    image_bbox = model_test.get_bbox(image)
    prediction = np.zeros((1, 3))
    test_speed = 0

    # Sum the prediction from single model's result
    for model_name in os.listdir(model_dir):
        h_input, w_input, model_type, scale = parse_model_name(model_name)
        param = {
            "org_img": image,
            "bbox": image_bbox,
            "scale": scale,
            "out_w": w_input,
            "out_h": h_input,
            "crop": scale is not None,
        }
        img = image_cropper.crop(**param)
        start = time.time()
        prediction += model_test.predict(img, os.path.join(model_dir, model_name))
        test_speed += time.time() - start

    # Draw result of prediction
    label = np.argmax(prediction)
    return label


if __name__ == "__main__":
    desc = "test"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument(
        "--device_id",
        type=int,
        default=0,
        help="Which GPU id to use [0/1/2/3]")
    parser.add_argument(
        "--model_dir",
        type=str,
        default="./resources/anti_spoof_models",
        help="Model library used for testing")
    parser.add_argument(
        "--image_name",
        type=str,
        default="unknown.jpg",
        help="Image used for testing")
    args = parser.parse_args()
    test(args.image_name, args.model_dir, args.device_id)
