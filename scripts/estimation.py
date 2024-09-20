import sys
import torch
import cv2
import numpy as np

from config import parse_args
from pose_estimator import camera_pose_predictor

def main():
  folder_path    = '/'.join(sys.path[0].split('/')[:-1]) + '/'
  resources_path = folder_path + 'resources/'
  refs_path      = resources_path + 'refs/'
  views_path     = resources_path + 'views/'

  args = parse_args()
  device = 'cuda' if torch.cuda.is_available() and not args.force_cpu else 'cpu'
  print('Running inference on device \"{}\"'.format(device))
  da_checkpoint  = f'depthanything/checkpoints/depth_anything_v2_metric_hypersim_{args.encoder}.pth'
  with open('./calib.txt') as f:
    lines = f.readlines()
  K = np.fromstring(lines[1], sep=' ').reshape((3,3))
  K[0,0]  /= 4
  K[1,1]  /= 4
  K[:2,2] /= 4
  dist_coef = np.fromstring(lines[3], sep=' ')

  superglue_config = {
    'superpoint': {
      'nms_radius'         : args.nms_radius,
      'keypoint_threshold' : args.keypoint_threshold,
      'max_keypoints'      : args.max_keypoints
    },
    'superglue': {
      'weights'             : args.superglue,
      'sinkhorn_iterations' : args.sinkhorn_iterations,
      'match_threshold'     : args.match_threshold,
    }
  }

  depthanything_config = {
      'encoder'    : args.encoder,
      'max_depth'  : args.max_depth,
      'input_size' : args.input_size,
      'state_dict' : da_checkpoint,
  }

  pose_predictor = camera_pose_predictor(refs_path, K, dist_coef, superglue_config, depthanything_config, device)

  cam_frame = cv2.imread(views_path + 'desk2_view.jpg')
  cam_frame = cv2.resize(cam_frame, (cam_frame.shape[1] // 4, cam_frame.shape[0] // 4))
  cam_pose = pose_predictor.infer_pose(cam_frame)
  print(cam_pose)

if __name__ == '__main__':
  main()
