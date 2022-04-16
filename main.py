import argparse
import os
import re
import time
from datetime import timedelta

import cv2
import numpy
import pandas as pd
from tqdm import tqdm
from video_pipeline_lighttrack import tracking_processing


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_folder", "-i", action="store", dest="input_folder", type=str, default="test_video"
    )
    parser.add_argument(
        "--output_folder",
        "-o",
        action="store",
        dest="output_folder",
        type=str,
        default="teste_video_l",
    )
    parser.add_argument(
        "--pose_model", "-p", action="store", dest="pose_model", type=str, default="rmppe"
    )
    parsed_args = parser.parse_args()
    input_folder = parsed_args.input_folder
    output_folder = parsed_args.output_folder
    pose_model = parsed_args.pose_model
    tracking_model = tracking_processing()
    tracking_model.init_models(pose_model_type=pose_model)
    if output_folder not in os.listdir():
        os.mkdir(output_folder)
    video_path = []
    for video in tqdm(os.listdir(input_folder)):
        video_path.append(os.path.join(input_folder, video))
        tracking_model.video_processing(
            video_path, output_folder, draw_labels=True, blur_face=False
        )


if __name__ == "__main__":
    main()
