#!/usr/bin/env python3

import math
import numpy as np
import scipy
import cv2

import os
import shutil
import tqdm


class CameraCalibration:
    def __init__(self):
        '''
        Initialize the class with the following attributes:
        '''
        self.path = 'Calibration_Imgs'
        self.vis = False
        self.chessboard_size = (9, 6)

        self.object_points = [] # 3D points (X,Y)
        self.image_points = [] # 2D points (u,v)

        self.size_of_chessboard_squares_mm = 21.5
        x, y = np.mgrid[1:self.chessboard_size[1]+1, 1:self.chessboard_size[0]+1]
        self.object_points = np.stack([x, y], axis=-1) * self.size_of_chessboard_squares_mm
        self.object_points = self.object_points.reshape(-1, 2) # Flatten the array to shape (N, 2)

        self.criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        self.chessboard_criteria = (cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)

        # Camera Parameters
        self.homography_set = []

    def read_images(self):
        images = []

        for imgname in os.listdir(self.path):
            if imgname.endswith('.jpg'):
                img = cv2.imread(os.path.join(self.path, imgname))
                if img is not None:
                    images.append(img)
                    if self.vis:
                        cv2.imshow('image', img)
                        cv2.waitKey(0)
                        cv2.destroyAllWindows()

        return images

    def detect_corners(self, images):
        '''
        Detect the corners of the chessboard in the images
        '''
        results_folder = "results/initiate_corners/"
        if os.path.exists(results_folder):
            shutil.rmtree(results_folder)
        os.makedirs(results_folder)

        image_counter = 1
        for img in tqdm.tqdm(images, desc="Detecting Chessboard Corner ",bar_format='{l_bar}{bar:20}{r_bar}'):
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            ret, corners = cv2.findChessboardCorners(gray, self.chessboard_size, None)

            if ret == True:                
                corners_sub = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), self.criteria)
                self.image_points.append(corners_sub)

                chessimg = cv2.drawChessboardCorners(img, self.chessboard_size, corners_sub, ret)
                
                filename = f"{results_folder}/chessboard_corners_{image_counter}.jpg" 
                cv2.imwrite(filename, chessimg)  
                image_counter += 1

            if self.vis:
                cv2.waitKey(0)
                cv2.destroyAllWindows()

        self.image_points = np.array(self.image_points)

    def compute_homography(self, img_pts):

        img_pts = np.array(img_pts)        
        n = len(img_pts)
        A = np.empty((n*2, 9))

        for i in range(n):
            u, v = img_pts[i][0]
            X, Y = self.object_points[i,0], self.object_points[i,1]

            # Constructing matrix A for the homography calculation
            A[2*i] = np.asarray([-X, -Y, -1, 0, 0, 0, u*X, u*Y, u])
            A[2*i + 1] = np.asarray([0, 0, 0, -X, -Y, -1, v*X, v*Y, v])

        # Perform SVD on A to get the homography matrix
        _, _, V = np.linalg.svd(A, full_matrices=True)
        H = V[-1,:].reshape(3, 3)
        H = H / H[2,2]

        return H
    
    def compute_homography_matrices(self, img_pts):
        for i in tqdm.tqdm(range(len(img_pts)), desc="Computing H Matrices ",bar_format='{l_bar}{bar:20}{r_bar}'):
            H = self.compute_homography(img_pts[i])
            self.homography_set.append(H)

        return self.homography_set
    
    def get_Vij(self, H,i,j):
        H = H.T
        V = np.array([[H[i][0] * H[j][0]], [H[i][0] * H[j][1] + H[i][1] * H[j][0]], [H[i][1] * H[j][1]],
                    [H[i][2] * H[j][0] + H[i][0] * H[j][2]], [H[i][2] * H[j][1] + H[i][1] * H[j][2]], [H[i][2] * H[j][2]]])
        return V.T
    
    def compute_B(self, H_set):
        V = []
        for i in tqdm.tqdm(range(len(H_set)), desc="Computing B Matrix...",bar_format='{l_bar}{bar:20}{r_bar}'):
            h = H_set[i]
            
            V12 = self.get_Vij(h, 0, 1)
            V11 = self.get_Vij(h, 0, 0)
            V22 = self.get_Vij(h, 1, 1)
            
            V.append(V12)
            V.append(V11 - V22)

        V = np.array(V)
        V = V.reshape(-1, 6)
        
        _,_,V_T = np.linalg.svd(V, full_matrices=True)  
        b = V_T[-1]

        B = np.array([[b[0], b[1], b[3]], 
                    [b[1], b[2], b[4]], 
                    [b[3], b[4], b[5]]])
                          
        return B
    
    def compute_extrinsic(self, K, Hset):
        R = []
        T = []
        
        for H in tqdm.tqdm(Hset, desc="Computing Extrinsic Parameters...",bar_format='{l_bar}{bar:20}{r_bar}'):
            h1 = H[:, 0]
            h2 = H[:, 1]
            h3 = H[:, 2]

            lamda = 1 / np.linalg.norm(np.dot(np.linalg.inv(K), h1), ord=2)

            r1 = lamda * np.dot(np.linalg.inv(K), h1)
            r2 = lamda * np.dot(np.linalg.inv(K), h2)
            t = np.dot(np.linalg.inv(K), h3) * lamda
            
            R.append(np.array([r1, r2]))
            T.append(t)

        return np.array(R),np.array(T)
    

    def compute_intrinsic(self,B):
        v0 = (B[0,1]*B[0,2] - B[0,0]*B[1,2]) / (B[0,0]*B[1,1] - B[0,1]**2)

        lamda = B[2,2] - (B[0,2]**2 + v0*(B[0,1]*B[0,2] - B[0,0]*B[1,2])) / B[0,0]
    
        alpha = math.sqrt(lamda / B[0,0])

        beta = math.sqrt((lamda * B[0,0]) / (B[0,0]*B[1,1] - B[0,1]**2))
        
        gamma = -B[0,1] * alpha**2 * beta / lamda

        u0 = gamma*v0 / beta - B[0,2]*alpha**2 / lamda

        K = np.array([[alpha, gamma, u0],
                        [0, beta, v0],
                        [0, 0, 1]])
        
        return K

    
    def loss_function(self, params, R, T, image_points, world_points):
        alpha, gamma, beta, u0, v0, k1, k2 = params
        K = np.array([[alpha, gamma, u0], [0, beta, v0], [0, 0, 1]])
        
        error = []
        reprojected_corners = []

        for i in range(len(image_points)):
            img_corners = image_points[i]
            RT = np.vstack((R[i], T[i]))
            RT = RT.T
            H = np.dot(K, RT)
            temp_error = 0
            temp_reprojected_corners = []

            for j in range(image_points[i].shape[0]):
                world_point = world_points[j]
                world_point = np.append(world_point, 1)
                world_point = world_point.reshape(-1, 1)
                world_point = world_point.T

                reprojected_point = np.matmul(RT, world_point.T)
                reprojected_point = reprojected_point / reprojected_point[2]

                corner_point_orig = img_corners[j]
                corner_point_orig = np.array(
                    [corner_point_orig[0, 0], corner_point_orig[0, 1], 1]
                )

                corner_point = np.matmul(H, world_point.T)
                corner_point = corner_point / corner_point[2]

                x = reprojected_point[0]
                y = reprojected_point[1]
                u = corner_point[0]
                v = corner_point[1]

                r = np.square(x) + np.square(y)
                u_hat = u + (u - u0) * (k1 * r + k2 * np.square(r))
                v_hat = v + (v - v0) * (k1 * r + k2 * np.square(r))


                corner_hat = np.array([u_hat[0], v_hat[0], 1], dtype=np.float32)
                temp_reprojected_corners.append(
                    np.array((corner_hat[0], corner_hat[1]))
                )

                temp_error += np.linalg.norm((corner_point_orig - corner_hat), ord=2)
        
            error.append(temp_error / image_points[i].shape[0])
            reprojected_corners.append(temp_reprojected_corners)

        return np.array(error), np.array(reprojected_corners)

    def optimization_function(self, params, R, T, image_points, world_points):
        error, _ = self.loss_function(params, R, T, image_points, world_points)
        return error.flatten()
    
    def save_original_corners(self, images, corners, path):
        for i in range(len(images)):
            img = images[i].copy()
            img = cv2.drawChessboardCorners(img, (9, 6), corners[i], True)
            output_path = path + str(i+1) + ".jpg"
            cv2.imwrite(output_path, img)

    def save_corners(self, images, corners, projected, path):
        for i in range(len(images)):
            img = images[i].copy()
            for j in range(len(corners[i])):
                # Draw outer circles without filling
                img = cv2.circle(
                    img,
                    (int(corners[i][j][0][0]), int(corners[i][j][0][1])),
                    10,
                    (0, 0, 255),
                    1,  # Set thickness to a positive value for outer circle without filling
                )
                img = cv2.circle(
                    img,
                    (int(projected[i][j][0][0]), int(projected[i][j][0][1])),
                    10,
                    (0, 255, 0),
                    1,  # Set thickness to a positive value for outer circle without filling
                )

            output_path = path + str(i+1) + ".jpg"
            cv2.imwrite(output_path, img)

if __name__ == '__main__':
    calib = CameraCalibration()
    
    print('Reading Calibration Images...')
    images = calib.read_images()
    print("")

    calib.detect_corners(images)
    print("")

    H_set = calib.compute_homography_matrices(calib.image_points)
    print("")

    B = calib.compute_B(H_set)
    print(B)
    print("")
    
    print('Computing Intrinsic Parameters...')
    K = calib.compute_intrinsic(B)
    print(K)
    print("")

    R,T = calib.compute_extrinsic(K, H_set)
    print("")

    # Re-projection Error for initial guess
    print('Calculating Re-projection Error for Initial Guess...')
    K_distortion = np.array([0, 0])
    intrinsic_params = np.array(
        [K[0, 0], K[0, 1], K[1, 1], K[0, 2], K[1, 2], K_distortion[0], K_distortion[1]]
    )

    reprojection_error, pts = calib.loss_function(
        intrinsic_params, R, T, calib.image_points, calib.object_points
    )
    
    print('Optimizing Parameters...')
    optimized_params = scipy.optimize.least_squares(
            calib.optimization_function,
            intrinsic_params,
            args=(R, T, calib.image_points, calib.object_points),
            method="lm",
        )

    res_params = optimized_params.x

    # New Intrinsic Parameters
    optimized_K = np.array([[res_params[0], res_params[1], res_params[3]], 
                            [0, res_params[2], res_params[4]], 
                            [0, 0, 1]])
                              
    optimized_K_distortion = np.array([res_params[5], res_params[6]])

    print("Optimized Intrinsic Parameters: \n", optimized_K)
    print("")
    print("Optimized Distortion Parameters: \n", optimized_K_distortion)
    print("")

    print('Calculating Re-projection Error for Optimized Parameters...')
    opt_reprojection_error, opt_reprojection_pts = calib.loss_function(
        intrinsic_params, R, T, calib.image_points, calib.object_points
    )

    print("Initial error: ", np.mean(reprojection_error))
    print("Optimized error: ", np.mean(opt_reprojection_error))
    print("")

    print("Saving Camera Parameters...")
    np.save("results/intrinsic_params.npy", optimized_K)
    np.save("results/distortion_params.npy", optimized_K_distortion)
    
    print('Reprojection with Optimized Parameter ...')
    reprojected_corners_mod = []
    for i in range(len(opt_reprojection_pts)):
        temp = []
        for j in range(len(opt_reprojection_pts[i])):
            temp.append([[opt_reprojection_pts[i][j][0], opt_reprojection_pts[i][j][1]]])
        reprojected_corners_mod.append(temp)
    
    reprojected_corners_mod = np.array(reprojected_corners_mod)
    
    print('Saving Results...')
    calib.save_original_corners(calib.read_images(), reprojected_corners_mod, "results/reprojected/")
    calib.save_corners(calib.read_images(), calib.image_points, reprojected_corners_mod, "results/comparison/")
    print("")

    print("Calibration Complete!")