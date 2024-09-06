# Tutorial: 3D Pose Estimation Pipeline for Mouse Behavior Analysis

## Introduction

This tutorial will guide you through the process of using our 3D pose estimation pipeline for mouse behavior analysis. The pipeline combines multiple tools and steps to process multi-camera video data, perform pose estimation, and generate 3D reconstructions of mouse movements.

## Prerequisites

Before starting, ensure you have the following:

1. Python 3.7+ installed
2. Required libraries: SLEAP, Aniposelib, OpenCV, NumPy, Pandas, h5py, toml, matplotlib, mayavi
3. GPU with CUDA support (recommended for faster processing)

## Pipeline Overview

The pipeline consists of the following main steps:

1. Project setup
2. Video preparation
3. Camera calibration
4. Creation of annotation projects
5. 2D pose estimation
6. 3D triangulation
7. Angle computation and data analysis
8. Visualization

Let's go through each step in detail.

## 1. Project Setup

First, we need to set up the project structure.

```python
import os

cameras = ["Camera_Back_Right", "Camera_Front_Right", "Camera_Front_Left", "Camera_Side_Left", "Camera_Side_Right", "Camera_Top_Left"]
session_list = ["M033_2024_04_10_10_00"]
main_dir = "E:/complete-projects/"

for session in session_list:
    path = main_dir + session
    if not os.path.exists(path):
        print("Creating directory: ", path)
        os.makedirs(path)
    for camera in cameras:
        path = main_dir + session + "/" + camera
        if not os.path.exists(path):
            print("Creating directory: ", path)
            os.makedirs(path)
```

This script creates a directory structure for your project, organizing it by session and camera.

## 2. Video Preparation

Before processing, you may need to prepare your videos. This could involve cropping, resizing, or extracting specific frames.

```python
import cv2
import os

def create_video_from_images(image_folder, save_video_name, output_width=1280, output_height=720, fps=5):
    images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
    images = sorted(images)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(save_video_name, fourcc, fps, (output_width, output_height))

    for image in images:
        frame = cv2.imread(os.path.join(image_folder, image))
        if frame is not None:
            resized_frame = cv2.resize(frame, (output_width, output_height))
            video.write(resized_frame)
    
    video.release()
```

This function allows you to create a video from a series of images, which can be useful for creating annotation videos or visualizing results.

## 3. Camera Calibration

Camera calibration is a crucial step for accurate 3D reconstruction. Use the `calibration.py` script provided in the SLEAP-Anipose package for this step. Ensure you have calibration videos or images for each camera.

```bash
python calibration.py --session_name SESSION_NAME --vid_dir VIDEO_DIRECTORY --board_path BOARD_CONFIG_PATH
```

## 4. Creation of Annotation Projects

To create annotation projects for SLEAP:

```python
import attr
import numpy as np
import random
import os
import json
import matplotlib.pyplot as plt
from sleap.io.video import Video
from sleap import Labels, Video, LabeledFrame, Instance, Skeleton
from sleap.info.feature_suggestions import FeatureSuggestionPipeline

pipeline = FeatureSuggestionPipeline(
    per_video=50,
    scale=0.25,
    sample_method="stride",
    feature_type="hog",
    brisk_threshold=10,
    n_components=10,
    n_clusters=10,
    per_cluster=5,
)

for session in session_list:
    video_dir = f'E:/video-recordings/{session}_cameras'
    save_main_dir = f'E:/annotations/{session}_annotations'
    for camera_name in cameras:
        save_dir = f'{save_main_dir}/{session}_{camera_name}_annotations'
        os.makedirs(save_dir, exist_ok=True)
        video_path = f'{video_dir}/{session}_{camera_name}.avi'
        video = Video.from_filename(filename=video_path)
        videos = [video]

        pipeline.run_disk_stage(videos)
        frame_data = pipeline.run_processing_state()
        items = frame_data.items

        for item in items:
            frame_idx = item.frame_idx
            frame = video.get_frame(frame_idx)
            plt.imsave(os.path.join(save_dir, f'{session}_{camera_name}_frame_{frame_idx}.png'), frame)
```

This script creates annotation projects for each camera view, extracting frames for annotation.

## 5. 2D Pose Estimation

Use SLEAP for 2D pose estimation. First, annotate a subset of frames in the SLEAP GUI. Then, train a model and run inference on all videos.

```bash
sleap-train CONFIG_FILE LABELED_DATA.slp
sleap-track VIDEO_FILE.mp4 -m MODEL_PATH.json --tracking.tracker none -o OUTPUT_FILE.slp
```

## 6. 3D Triangulation

After obtaining 2D pose estimations, use Anipose for 3D triangulation:

```python
import numpy as np
import pandas as pd
import anipose
import h5py
import toml
import os
from anipose import triangulate

project_dir = "E:/complete-projects/SESSION_NAME"
config = toml.load(os.path.join(project_dir, 'config.toml'))

# Load 2D tracking data
tracks = np.array(h5py.File(os.path.join(project_dir, "POSE_ESTIMATION_FILE.h5"), 'r')['tracks'])

# Perform triangulation
points3d = triangulate(tracks, config)

# Save the results
np.save(os.path.join(project_dir, '3d_points.npy'), points3d)
```

## 7. Angle Computation and Data Analysis

After 3D reconstruction, compute joint angles and analyze the data:

```python
import pandas as pd
import numpy as np
from anipose.compute_angles import compute_angles

config_path = os.path.join(project_dir, f'{session_name}_config.toml')
config = toml.load(config_path)
labels_fname = f'{project_dir}/{session_name}_pose_estimation.csv'
outname = f'{project_dir}/{session_name}_angles.csv'

compute_angles(config, labels_fname, outname)

# Load and combine 3D points and angles
pts_df = pd.read_csv(f'{project_dir}/{session_name}_pose_estimation.csv')
angle_df = pd.read_csv(outname)
angle_df = angle_df[['left_elbow_angle', 'right_elbow_angle', 'left_knee_angle', 'right_knee_angle', "left_ankle_angle", "right_ankle_angle"]]

combined_df = pd.concat([pts_df, angle_df], axis=1)
combined_csv_path = f'{project_dir}/{session_name}_3dpts_angles.csv'
combined_df.to_csv(combined_csv_path, index=False)
```

## 8. Visualization

Visualize your results using matplotlib or other visualization libraries:

```python
import matplotlib.pyplot as plt

df = pd.read_csv(outname)
df = df[['left_elbow', 'right_elbow', 'left_knee', 'right_knee', "left_ankle", "right_ankle"]]
df = df[400:500]

plt.figure(figsize=(20, 5))
df.plot()
plt.ylabel('angle (degree)')
plt.xlabel('frame')
plt.legend(fontsize='x-small')
plt.savefig(f'{project_dir}/angles.pdf')
plt.show()
```

## Tips and Best Practices

1. Ensure consistent lighting and camera setups across all recording sessions.
2. Use a high-quality calibration board and capture it from various angles for better calibration.
3. Annotate a diverse set of frames for training the 2D pose estimation model.
4. Regularly back up your data and results.
5. Use version control for your analysis scripts.

## Troubleshooting

- If calibration fails, check your calibration board setup and try recapturing calibration videos.
- For poor 2D pose estimation results, try annotating more diverse frames or adjusting the model architecture.
- If 3D triangulation produces unrealistic results, double-check your camera calibration and 2D pose estimates.

## Conclusion

This pipeline provides a comprehensive approach to 3D pose estimation for mouse behavior analysis. By following these steps and adapting them to your specific needs, you can generate valuable insights from your multi-camera recordings. Remember to validate your results and consult the documentation of individual tools (SLEAP, Anipose) for more detailed information on specific components.