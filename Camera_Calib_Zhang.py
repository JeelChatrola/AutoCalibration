import math
import numpy as np
import scipy
import cv2
import os
import glob
import tqdm


class CameraCalibration:
    def __init__(self):
        '''
        Initialize the class with the following attributes:
        '''
        self.path = 'Calibration_Imgs'
        self.vis = False
        self.checkerboard_size = (6, 9)

        self.object_points = [] # 3D points (X,Y)
        self.image_points = [] # 2D points (u,v)

        self.size_of_chessboard_squares_mm = 12.5
        self.object_pt = np.zeros((self.checkerboard_size[0] * self.checkerboard_size[1], 2), np.float32)
        self.object_pt[:, :2] = np.mgrid[0:self.checkerboard_size[0], 0:self.checkerboard_size[1]].T.reshape(-1, 2) * self.size_of_chessboard_squares_mm 
        
        self.criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        self.checkerboard_criteria = (cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)

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
        for img in tqdm.tqdm(images, desc="Processing Images ",bar_format='{l_bar}{bar:20}{r_bar}'):
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            ret, corners = cv2.findChessboardCorners(gray, self.checkerboard_size, None)

            if ret == True:
                self.object_points.append(self.object_pt)
                
                corners_sub = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), self.criteria)
                self.image_points.append(corners)
                        
                chessimg = cv2.drawChessboardCorners(img, self.checkerboard_size, corners_sub, ret)
                
            if self.vis:
                cv2.imshow('chessboard corners', chessimg)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

    def compute_homography(self, img_pts, obj_pts):

        img_pts = np.array(img_pts)
        obj_pts = np.array(obj_pts)

        n = len(img_pts[0])        
        A = np.empty((n*2, 9))

        for i in range(n):
            u, v = img_pts[0][i]
            X, Y = obj_pts[i][:2]

            # Construct the rows of matrix A for the homography calculation
            A[2*i] = np.asarray([X, Y, 1, 0, 0, 0, -u*X, -u*Y, -u])
            A[2*i+1] = np.asarray([0, 0, 0, X, Y, 1, -v*X, -v*Y, -v])

        # Perform SVD on A to get the homography matrix
        _, _, V = np.linalg.svd(A, full_matrices=True)
        H = V[-1,:].reshape(3, 3)
        H = H/H[2,2]
    
        return H
    
    def compute_homography_matrices(self, img_pts, obj_pts):
        H_set = []
        for i in tqdm.tqdm(range(len(img_pts)), desc="Computing H Matrices ",bar_format='{l_bar}{bar:20}{r_bar}'):
            # H1,_=cv2.findHomography(obj_pts[i],img_pts[i],cv2.RANSAC,5.0)
            H = self.compute_homography(img_pts[i], obj_pts[i])
            # print(H-H1)
            H_set.append(H)

        return H_set

    def vij(self, h1, h2):
        vij=np.array([h1[0]*h2[0], h1[0]*h2[1] + h1[1]*h2[1], h1[1]*h2[1], 
                      h1[2]*h2[0] + h1[0]*h2[2], h1[2]*h2[1] + h1[1]*h2[2], h1[2]*h2[2]])
        
        return vij.T
    
    def compute_B(self, H_set):
        B = np.zeros((3,3))
        
        for h in H_set:
            h1 = h[:,0]
            h2 = h[:,1]

            V12 = self.vij(h1,h2)     
            V11 = self.vij(h1,h1)
            V22 = self.vij(h2,h2)

            V = np.vstack((V12.T, (V11 - V22).T))

            _,_,V = np.linalg.svd(V)  
            b = V[-1, :]

            B = np.array([[b[0], b[1], b[3]], 
                          [b[1], b[2], b[4]], 
                          [b[3], b[4], b[5]]])
                          
        return B


if __name__ == '__main__':
    calib = CameraCalibration()
    
    print('Reading Calibration Images...')
    images = calib.read_images()

    print('Detecting Checkerboard corners...')
    calib.detect_corners(images)

    print('Computing Homography Matrix...')
    H_set = calib.compute_homography_matrices(calib.image_points, calib.object_points)
    # print(H_set)

    print('Computing B Matrix...')
    C = calib.compute_B(H_set)
    print(C)
    
    print('Calibrating Camera...')
