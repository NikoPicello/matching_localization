import sys
import numpy as np
import cv2
import glob
import os

folder_path = '/'.join(sys.path[0].split('/')[:-1]) + '/'
resources_path = folder_path + 'resources/'
calib_img_folder = resources_path + 'calib_images/'
images = glob.glob(calib_img_folder + '*.jpg')

# Define the dimensions of checkerboard
CHECKERBOARD = (7, 10)

# stop the iteration when specified
# accuracy, epsilon, is reached or
# specified number of iterations are completed.
criteria = (cv2.TERM_CRITERIA_EPS +
      cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)


# Vector for 3D points
threedpoints = []

# Vector for 2D points
twodpoints = []


# 3D points real world coordinates
objectp3d = np.zeros((1, CHECKERBOARD[0]
          * CHECKERBOARD[1],
          3), np.float32)
objectp3d[0, :, :2] = np.mgrid[0:CHECKERBOARD[0],
              0:CHECKERBOARD[1]].T.reshape(-1, 2)
prev_img_shape = None

print(images)
for filename in images:
  image = cv2.imread(filename)
  # lwr = np.array([0, 0, 143])
  # upr = np.array([179, 61, 252])
  # hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
  # msk = cv2.inRange(hsv, lwr, upr)
  # krn = cv2.getStructuringElement(cv2.MORPH_RECT, (50, 30))
  # dlt = cv2.dilate(msk, krn, iterations=5)
  # res = 255 - cv2.bitwise_and(dlt, msk)
  # res = np.uint8(res)
  # res = cv2.resize(res, (868, 1156))
  # cv2.imshow('res', res)
  # cv2.waitKey(0)
  res = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

  # Find the chess board corners
  # If desired number of corners are
  # found in the image then ret = true
  ret, corners = cv2.findChessboardCorners(
          res, CHECKERBOARD,
          cv2.CALIB_CB_ADAPTIVE_THRESH
          + cv2.CALIB_CB_FAST_CHECK +
          cv2.CALIB_CB_NORMALIZE_IMAGE)
  print(ret)

  # If desired number of corners can be detected then,
  # refine the pixel coordinates and display
  # them on the images of checker board
  if ret == True:
    threedpoints.append(objectp3d)

    # Refining pixel coordinates
    # for given 2d points.
    corners2 = cv2.cornerSubPix(
      res, corners, (7, 10), (-1, -1), criteria)

    twodpoints.append(corners2)

    # Draw and display the corners
    image = cv2.drawChessboardCorners(image,
                    CHECKERBOARD,
                    corners2, ret)
  # cv2.imshow('img', image)
  # cv2.waitKey(0)

cv2.destroyAllWindows()

h, w = image.shape[:2]


# Perform camera calibration by
# passing the value of above found out 3D points (threedpoints)
# and its corresponding pixel coordinates of the
# detected corners (twodpoints)
ret, matrix, distortion, r_vecs, t_vecs = cv2.calibrateCamera(
  threedpoints, twodpoints, res.shape[::-1], None, None)


# Displaying required output
print(" Camera matrix:")
print(matrix)

print("\n Distortion coefficient:")
print(distortion)

print("\n Rotation Vectors:")
print(r_vecs)

print("\n Translation Vectors:")
print(t_vecs)
