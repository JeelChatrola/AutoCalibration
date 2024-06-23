# AutoCalib

In this project, we implement famous Zhang's Camera Calibration method from scratch. 
We estimate parameters of the camera like the focal length, distortion coefficients and principle point is called Camera Calibration.

### Requirements

1. NumPy
2. OpenCV
3. SciPy
4. tqdm

### Implementation 


Highlight of the major steps include from the paper: 
A Flexible New Technique for Camera Calibration - [Zhang's Paper link](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/tr98-71.pdf)

1. Initial Parameter Estimation 
2. Solving for approximate intrinsic and extrinsic parameters (assuming no distortion)
3. Non-linear Geometric Error Minimization using Least Squares Optimization


To run the code please clone the repo and execute:

```
python3 Camera_Calib_Zhang.py

```
one should place the images to be used for calibration in **Calibration_Imgs/** folder.
This will save results in the **results/** folder along with intermediate images and final camera parameters. 


### Results

After comparison with CV2 library implementation we found out that results are extremely close (given that we only use low level distortion coefficients).
In the third image Red circle shows original corners and green shows reprojected corners

| Initial | Reprojected | Comparison | 
|:-------:|:-------:|:-------:|
|<img src="/results/initiate_corners/chessboard_corners_1.jpg" width="300">|<img src="/results/reprojected/1.jpg" width="300">|<img src="/results/comparison/1.jpg" width="300"> |
