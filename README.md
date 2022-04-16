# Person pose tracking
The pipeline includes 2d person pose estimator and 3d person pose estimator

```
$ python main.py --input_folder=`path_to_videos` --output_folder=`path_to_save_vis_videos` --device=0 --model='rmppe'
```
- ```--input_folder``` - directory with videofiles
- ```--output_folder``` - directory with labeled videofiles
- ```--device``` - GPU 
- ```--pose_model```
    - ```rmppe``` - Realtime Multi-Person 2D Pose Estimation using Part Affinity Field 
    - ```rmppe_224``` - ```rmppe``` using TensorRT
    - ```snwbpe``` - Single-Network Whole-Body Pose Estimation
- Person 3D pose estimation model is 3D human pose estimation in video with temporal convolutions and semi-supervised training
- Person Pose tracking models is LightTrack Siamese Graph Convolutional Network

![Alt Text](https://github.com/Aigul95/pose_estimation_pub/blob/master/demo.gif)



