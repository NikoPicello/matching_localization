import sys
import argparse
import cv2
import glob
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import os
import torch

from depthanything.depth_anything_v2.dpt import DepthAnythingV2
from scipy.optimize import least_squares

mkpts0 = np.array(
         [(606.,  73.),
          (608.,  87.),
          (616.,  94.),
          (275., 314.),
          (468., 343.),
          (475.,  86.),
          (226.,  94.),
          (507., 100.),
          (283., 326.),
          (597., 458.)]
)
# mkpts0 = np.int32(mkpts0)
mkpts1 = np.array(
         [(583., 231.),
          (576., 240.),
          (588., 245.),
          (469., 381.),
          (495., 399.),
          (497., 212.),
          (374., 175.),
          (498., 223.),
          (429., 363.),
          (622., 477.)]
)
# mkpts1 = np.int32(mkpts1)

t = np.array([[0.42767296], [-0.42097881], [0.79992042]])

R = np.array([
    [0.64000037, 0.35647223, -0.68068133],
    [-0.34700468, 0.92447866, 0.15788276],
    [0.68555618, 0.13515459, 0.71536421]
])

# def stereo_to_worold(

def pixel_to_world(pixel_coords, distance, K):
  print(distance)
  distance = np.median(distance)
  print(distance)
  input()
  u, v = pixel_coords[0], pixel_coords[1]
  pixel_coords_homogeneous = np.array([u, v, 1])
  normalized_coords = np.linalg.inv(K) @ pixel_coords_homogeneous
  X_c, Y_c, Z_c = normalized_coords * distance
  return np.array([X_c, Y_c, Z_c])

def residuals(scale, points1, points2, t):
  T = scale * t.flatten()
  predicted_distances = np.linalg.norm(points2 - (points1 + T), axis=1)
  measured_distances = np.linalg.norm(points2 - points1, axis=1)
  return measured_distances - predicted_distances

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Depth Anything V2')

  parser.add_argument('--img-path', type=str)
  parser.add_argument('--input-size', type=int, default=518)
  parser.add_argument('--outdir', type=str, default='./vis_depth')

  parser.add_argument('--encoder', type=str, default='vitl', choices=['vits', 'vitb', 'vitl', 'vitg'])

  parser.add_argument('--pred-only', dest='pred_only', action='store_true', help='only display the prediction')
  parser.add_argument('--grayscale', dest='grayscale', action='store_false', help='do not apply colorful palette')

  args = parser.parse_args()

  DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

  model_configs = {
      'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
      'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
      'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
      'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
  }

  folder_path    = '/'.join(sys.path[0].split('/')[:-1]) + '/'
  resources_path = folder_path + 'resources/'
  refs_folder    = resources_path + 'refs/'
  views_folder   = resources_path + 'views/'


  with open('./calib.txt') as f:
    lines = f.readlines()
  K = np.fromstring(lines[1], sep=' ').reshape((3,3))
  # since the image is scaled by a factor of 4 I'm re-scaling the camera matrix as well (note that the focal length and the central points in the camera matrix are indeed measured in pixels)!
  K[0,0]  /= 4
  K[1,1]  /= 4
  K[:2,2] /= 4
  dist_coef = np.fromstring(lines[3], sep=' ')


  depth_anything = DepthAnythingV2(**{**model_configs[args.encoder], 'max_depth': 10.0})
  depth_anything.load_state_dict(torch.load(f'depthanything/checkpoints/depth_anything_v2_metric_hypersim_{args.encoder}.pth', map_location='cpu'))
  depth_anything = depth_anything.to(DEVICE).eval()

  cmap = cm.get_cmap('Spectral_r')


  ref_image  = cv2.imread(refs_folder + 'lib_ref.jpg')
  view_image = cv2.imread(views_folder + 'lib_view.jpg')
  ref_image  = cv2.resize(ref_image, (ref_image.shape[1] // 4, ref_image.shape[0] // 4))
  view_image = cv2.resize(view_image, (view_image.shape[1] // 4, view_image.shape[0] // 4))

  ref_depth  = depth_anything.infer_image(ref_image, args.input_size)
  view_depth = depth_anything.infer_image(view_image, args.input_size)

  print(ref_depth[int(mkpts0[0][1]), int(mkpts0[0][0])])

  # CONVERT MATCHING POINTS FROM 2D COORDS TO 3D COORDS
  mkpts0_3d = np.array([pixel_to_world(pt, (ref_depth[int(pt[1])-3:int(pt[1])+4, int(pt[0])-3:int(pt[0])+4]), K) for pt in mkpts0])
  mkpts1_3d = np.array([pixel_to_world(pt, (view_depth[int(pt[1])-3:int(pt[1])+4, int(pt[0])-3:int(pt[0])+4]), K) for pt in mkpts1])

  success, rot, trans, _ = cv2.solvePnPRansac(
      mkpts0_3d,
      mkpts1,
      K,
      distCoeffs=dist_coef
  )

  rot, _ = cv2.Rodrigues(rot)
  camera_pos = np.hstack((rot, trans))
  print(camera_pos)




  # OPTIMIZE THE SCALE FACTOR
  # initial_scale = 1.0
  # result = least_squares(residuals, initial_scale, args=(mkpts0_3d, mkpts1_3d, t))
  # optimal_scale = result.x[0]
  # T = optimal_scale * t

  # depth = (ref_depth - ref_depth.min()) / (ref_depth.max() - ref_depth.min()) * 255.0
  # depth = depth.astype(np.uint8)

  # if args.grayscale:
  #   depth = np.repeat(depth[..., np.newaxis], 3, axis=-1)
  # else:
  #   depth = (cmap(depth)[:, :, :3] * 255)[:, :, ::-1].astype(np.uint8)

  plt.figure(figsize=(8, 6))
  plt.imshow(ref_depth)
  plt.title('Depth Map')
  plt.axis('off')
  plt.show()

  # if args.pred_only:
  #   cv2.imwrite(os.path.join(args.outdir, os.path.splitext(os.path.basename(filename))[0] + '.png'), depth)
  # else:
  #   split_region = np.ones((raw_image.shape[0], 50, 3), dtype=np.uint8) * 255
  #   combined_result = cv2.hconcat([raw_image, split_region, depth])

  #   cv2.imwrite(os.path.join(args.outdir, os.path.splitext(os.path.basename(filename))[0] + '.png'), combined_result)
