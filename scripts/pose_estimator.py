import time
import argparse
import cv2
import torch
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from scipy.optimize import least_squares
from config import parse_args
from depthanything.depth_anything_v2.dpt import DepthAnythingV2
from superglue.models.matching import Matching
from superglue.models.utils import (AverageTimer, VideoStreamer,
                                    make_matching_plot_fast, frame2tensor)

torch.set_grad_enabled(False)
matplotlib.use('TkAgg')

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
    ref_frame = cv2.imread(self.ref_folder + 'ground_robot_ref.jpg')
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
    # mkpts_cam_3d = np.array([self.px2wrld(pt, cam_depth[int(pt[1]), int(pt[0])], self.K) for pt in mkpts_cam])

    # plt.figure(figsize=(8, 6))
    # plt.imshow(ref_depth)
    # plt.title('Depth Map')
    # plt.axis('off')
    # plt.show()


    # for p3d, p2d in zip(mkpts_ref_3d, mkpts_cam):
    #   print(p3d)
    #   print(p2d)
    #   image = cam_frame.cpu().numpy().copy()
    #   image = np.squeeze(image, 0)
    #   image = np.transpose(image, (1,2,0))
    #   image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    #   cv2.circle(image, (int(p2d[0]), int(p2d[1])), 5, (0, 0, 255), -1)  # Red color
    #   cv2.imshow('ref_frame', image)
    #   cv2.waitKey(0)


    success, rot, trans, inliers = cv2.solvePnPRansac(
        mkpts_ref_3d,
        mkpts_cam,
        self.K,
        distCoeffs=self.dist_coefs
    )

    # success, rot, trans = cv2.solvePnP(
    #     mkpts_ref_3d,
    #     mkpts_cam,
    #     self.K,
    #     distCoeffs=self.dist_coefs
    # )


    # Reprojection using solvePnP
    # projected_points_pnp, _ = cv2.projectPoints(mkpts_ref_3d, rot, trans, self.K, self.dist_coefs)
    # reprojection_error_pnp = np.mean(np.linalg.norm(mkpts_cam.reshape(-1, 2) - projected_points_pnp.reshape(-1, 2), axis=1))

    # # Reprojection using solvePnPRansac
    # projected_points_ransac, _ = cv2.projectPoints(mkpts_ref_3d[inliers], rot_r, trans_r, self.K, self.dist_coefs)
    # reprojection_error_ransac = np.mean(np.linalg.norm(mkpts_cam[inliers].reshape(-1, 2) - projected_points_ransac.reshape(-1, 2), axis=1))

    # print(f"Reprojection Error (solvePnP): {reprojection_error_pnp}")
    # print(f"Reprojection Error (solvePnPRansac): {reprojection_error_ransac}")

    # self.visualize_reprojection(cam_frame.cpu().numpy(), mkpts_cam, projected_points_ransac)

    rot, trans = cv2.solvePnPRefineLM(
        mkpts_ref_3d,
        mkpts_cam,
        self.K,
        distCoeffs=self.dist_coefs,
        rvec=rot,
        tvec=trans
    )
    rot, _ = cv2.Rodrigues(rot)
    camera_pos = np.hstack((rot, trans))

    return camera_pos

  def px2wrld(self, px_coords, distance, K):
    # print(px_coords)
    # print(distance)
    u, v = px_coords[0], px_coords[1]
    px_coords_homogeneous = np.array([u, v, 1])
    normalized_coords = np.linalg.inv(K) @ px_coords_homogeneous
    X_c, Y_c, Z_c = normalized_coords * distance
    return np.array([X_c, Y_c, Z_c])

  def visualize_reprojection(self, image, camera_points, reprojected_points):
    image = np.squeeze(image, 0)
    image = np.transpose(image, (1,2,0))
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    for p in camera_points:
      cv2.circle(image, (int(p[0]), int(p[1])), 5, (0, 0, 255), -1)  # Red color

    for p in reprojected_points:
      cv2.circle(image, (int(p[0][0]), int(p[0][1])), 5, (0, 255, 0), -1)  # Green color

    cv2.imshow('visualize reprojection', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

