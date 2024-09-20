import time
import argparse
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from scipy.optimize import least_squares
from config import parse_args
from depthanything.depth_anything_v2.dpt import DepthAnythingV2
from superglue.models.matching import Matching
from superglue.models.utils import (AverageTimer, VideoStreamer,
                                    make_matching_plot_fast, frame2tensor)

torch.set_grad_enabled(False)

da_configs = {
  'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
  'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
  'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
  'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
}

class camera_pose_predictor():
  def __init__(self, ref_folder_path, camera_matrix, dist_coeffs,
      sg_config, da_config, device):

    self.ref_folder = ref_folder_path
    self.dist_coefs = dist_coeffs
    self.K = camera_matrix
    self.device = device
    self.da_input_size = da_config['input_size']

    self.superglue = Matching(sg_config).eval().to(device)
    self.keys = ['keypoints', 'scores', 'descriptors']

    self.depthanything = DepthAnythingV2(**{**da_configs[da_config['encoder']], 'max_depth': da_config['max_depth']})
    self.depthanything.load_state_dict(torch.load(da_config['state_dict']))
    self.depthanything.eval().to(device)

  def infer_pose(self, cam_frame):
    # for now let's stick to one ref frame at each inference, then it should be changed so to have a way to distinguish which one to use
    ref_frame = cv2.imread(self.ref_folder + 'desk2_ref.jpg')
    ref_frame = cv2.resize(ref_frame, (ref_frame.shape[1] // 4, ref_frame.shape[0] // 4))

    ref_depth = self.depthanything.infer_image(ref_frame, self.da_input_size)
    cam_depth = self.depthanything.infer_image(cam_frame, self.da_input_size)

    ref_frame = cv2.cvtColor(ref_frame, cv2.COLOR_BGR2GRAY)
    ref_frame = frame2tensor(ref_frame, self.device)
    last_data = self.superglue.superpoint({'image': ref_frame})
    last_data = {k+'0': last_data[k] for k in self.keys}
    last_data['image0'] = ref_frame


    cam_frame = cv2.cvtColor(cam_frame, cv2.COLOR_BGR2GRAY)
    cam_frame = frame2tensor(cam_frame, self.device)

    pred = self.superglue({**last_data, 'image1': cam_frame})
    kpts_ref = last_data['keypoints0'][0].cpu().numpy()
    kpts_cam = pred['keypoints1'][0].cpu().numpy()
    matches = pred['matches0'][0].cpu().numpy()
    confidence = pred['matching_scores0'][0].cpu().numpy()

    valid = matches > -1
    mkpts_ref = kpts_ref[valid]
    mkpts_cam = kpts_cam[matches[valid]]

    _, E, R, t, mask = cv2.recoverPose(mkpts_ref, mkpts_cam, self.K, self.dist_coefs, self.K, self.dist_coefs)
    mkpts_ref = mkpts_ref[mask.ravel() == 1]
    mkpts_cam = mkpts_cam[mask.ravel() == 1]

    mkpts_ref_3d = np.array([self.px2wrld(pt, ref_depth[int(pt[1]), int(pt[0])], self.K) for pt in mkpts_ref])
    mkpts_cam_3d = np.array([self.px2wrld(pt, cam_depth[int(pt[1]), int(pt[0])], self.K) for pt in mkpts_cam])

    success, rot, trans = cv2.solvePnP(
        mkpts_ref_3d,
        mkpts_cam,
        self.K,
        distCoeffs=self.dist_coefs
    )

    rot, _ = cv2.Rodrigues(rot)
    camera_pos = np.hstack((rot, trans))
    return camera_pos

  def px2wrld(self, px_coords, distance, K):
    u, v = px_coords[0], px_coords[1]
    px_coords_homogeneous = np.array([u, v, 1])
    normalized_coords = np.linalg.inv(K) @ px_coords_homogeneous
    X_c, Y_c, Z_c = normalized_coords * distance
    return np.array([X_c, Y_c, Z_c])
