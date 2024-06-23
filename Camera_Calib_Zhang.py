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
        self.chessboard_size = (9, 6)

        self.object_points = [] # 3D points (X,Y)
        self.image_points = [] # 2D points (u,v)

        self.size_of_chessboard_squares_mm = 12.5
        y, x = np.mgrid[1:self.chessboard_size[1]+1, 1:self.chessboard_size[0]+1]
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
        for img in tqdm.tqdm(images, desc="Detecting Chessboard Corner ",bar_format='{l_bar}{bar:20}{r_bar}'):
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            ret, corners = cv2.findChessboardCorners(gray, self.chessboard_size, None)

            if ret == True:                
                corners_sub = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), self.criteria)
                self.image_points.append(corners_sub)

                chessimg = cv2.drawChessboardCorners(img, self.chessboard_size, corners_sub, ret)
                
            if self.vis:
                cv2.imshow('chessboard corners', chessimg)
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

            # Construct the rows of matrix A for the homography calculation
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
        b = -V_T[-1]

        B = np.array([[b[0], b[1], b[3]], 
                    [b[1], b[2], b[4]], 
                    [b[3], b[4], b[5]]])
                          
        return B
    
    def compute_extrinsic(self, K, Hset):
        R = []
        T = []
        
        for H in tqdm.tqdm(Hset):
            h1 = H[:, 0]
            h2 = H[:, 1]
            h3 = H[:, 2]

            lamda = 1/ np.linalg.norm(np.dot(np.linalg.inv(K), h1), 2)

            r1 = lamda * np.dot(np.linalg.inv(K), h1)
            r2 = lamda * np.dot(np.linalg.inv(K), h2)
            r3 = np.cross(r1, r2)

            t = np.dot(np.linalg.inv(K), h3) / lamda
            
            r = np.vstack((r1, r2, r3)).T
            
            R.append(r)
            T.append(t)

        return np.array(R),np.array(T)
    

    def compute_intrinsic(self,B):
        # v0 = B12B13 - B11B23 / B11B22 - B12^2
        v0 = (B[0,1]*B[0,2] - B[0,0]*B[1,2]) / (B[0,0]*B[1,1] - B[0,1]**2)

        # lamda = B22 - (B12^2 + v0(B11B12 - B12B13)) / B11
        lamda = B[2,2] - (B[0,2]**2 + v0*(B[0,1]*B[0,2] - B[0,0]*B[1,2])) / B[0,0]
    
        # alpha = sqrt(lamda / B11)
        alpha = math.sqrt(lamda / B[0,0])

        # beta = sqrt(lamda * B11 / (B11B22 - B12^2))

        beta = math.sqrt((lamda * B[0,0]) / (B[0,0]*B[1,1] - B[0,1]**2))
        
        # gamma = -B12 * alpha^2 * beta / lamda
        gamma = -B[0,1] * alpha**2 * beta / lamda

        # u0 = gamma*v0 / beta - B13*alpha^2 / lamda
        u0 = gamma*v0 / beta - B[0,2]*alpha**2 / lamda

        # Arrange these in a matrix
        K = np.array([[alpha, gamma, u0],
                        [0, beta, v0],
                        [0, 0, 1]])
        
        return K
    
    def optimize_parameters(self):
        pass


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

    print('Computing Extrinsic Parameters...')
    R,T = calib.compute_extrinsic(K, H_set)
    print("Rotation Matrix ",R.shape)
    print("Translation ",T.shape)
    
    # print('Optimizing Parameters: ')
    # calib.optimize_parameters()