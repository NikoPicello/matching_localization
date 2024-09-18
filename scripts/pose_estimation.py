# image_016.jpg,41.3836782,2.1804202,0
# image_017.jpg,41.3836782,2.1804202,287.106515478015

import sys
import time
import argparse
import cv2
import torch
import numpy as np
import matplotlib.cm as cm

from config import parse_args
from pathlib import Path
from superglue.models.matching import Matching
from superglue.models.utils import (AverageTimer, VideoStreamer,
                                    make_matching_plot_fast, frame2tensor)

torch.set_grad_enabled(False)


def main():
  folder_path    = '/'.join(sys.path[0].split('/')[:-1]) + '/'
  resources_path = folder_path + 'resources/'
  refs_folder    = resources_path + 'refs/'
  views_folder   = resources_path + 'views/'

  opt = parse_args()
  device = 'cuda' if torch.cuda.is_available() and not opt.force_cpu else 'cpu'
  print('Running inference on device \"{}\"'.format(device))

  config = {
    'superpoint': {
      'nms_radius': opt.nms_radius,
      'keypoint_threshold': opt.keypoint_threshold,
      'max_keypoints': opt.max_keypoints
    },
    'superglue': {
      'weights': opt.superglue,
      'sinkhorn_iterations': opt.sinkhorn_iterations,
      'match_threshold': opt.match_threshold,
    }
  }
  matching = Matching(config).eval().to(device)
  keys = ['keypoints', 'scores', 'descriptors']

  # DEFINE REFERENCE FRAME
  ref_frame = cv2.imread(refs_folder + 'ground_robot_ref.jpg')
  ref_frame = cv2.resize(ref_frame, (868, 1156))
  ref_frame = cv2.cvtColor(ref_frame, cv2.COLOR_BGR2GRAY)

  ref_tframe = frame2tensor(ref_frame, device)
  last_data = matching.superpoint({'image': ref_tframe})
  last_data = {k+'0': last_data[k] for k in keys}
  last_data['image0'] = ref_tframe
  last_frame = ref_frame
  last_image_id = 0

  # cumulative_time = 0
  # cumulative_it = 0

  # for i in range(100):
  #   start = time.time()

  # DEFINE CURRENT FRAME TO MATCH WITH THE REFERENCE ONE
  cur_frame = cv2.imread(views_folder + 'ground_robot_view.jpg')
  cur_frame = cv2.resize(cur_frame, (868, 1156))
  cur_frame = cv2.cvtColor(cur_frame, cv2.COLOR_BGR2GRAY)

  cur_tframe = frame2tensor(cur_frame, device)
  pred = matching({**last_data, 'image1': cur_tframe})
  kpts0 = last_data['keypoints0'][0].cpu().numpy()
  kpts1 = pred['keypoints1'][0].cpu().numpy()
  matches = pred['matches0'][0].cpu().numpy()
  confidence = pred['matching_scores0'][0].cpu().numpy()
  #   if i > 15:
  #     cumulative_time += (time.time() - start)
  #     cumulative_it += 1

  # print(cumulative_it)
  # print(cumulative_time / cumulative_it)

  valid = matches > -1
  mkpts0 = kpts0[valid]
  mkpts1 = kpts1[matches[valid]]

  # COMPUTE THE FOUNDAMENTA MATRIX
  mkpts0 = np.int32(mkpts0)
  mkpts1 = np.int32(mkpts1)
  F, mask = cv2.findFundamentalMat(mkpts0, mkpts1, cv2.FM_LMEDS)

  mkpts0 = mkpts0[mask.ravel() == 1]
  mkpts1 = mkpts1[mask.ravel() == 1]

  # TODO: THE FOUNDAMENTAL MATRIX HAS BEEN FOUND, NOW WE NEED TO COMPUTE THE RELATIVE TRANSLATION BETWEEN THE 2 VIEWS. HAVING THE FOUNDAMENTAL MATRIX AND THE INTRINSIC PARAMETERS OF THE CAMERA IT SHOULD BE EASY

  color = cm.jet(confidence[valid])
  text = [
      'SuperGlue',
      'Keypoints: {}:{}'.format(len(kpts0), len(kpts1)),
      'Matches: {}'.format(len(mkpts0))
  ]
  k_thresh = matching.superpoint.config['keypoint_threshold']
  m_thresh = matching.superglue.config['match_threshold']
  small_text = [
      'Keypoint Threshold: {:.4f}'.format(k_thresh),
      'Match Threshold: {:.2f}'.format(m_thresh),
      'Image Pair: {:06}:{:06}'.format('016', '017'),
  ]
  out = make_matching_plot_fast(
      last_frame, cur_frame, kpts0, kpts1, mkpts0, mkpts1, color, text,
      path=None, show_keypoints=opt.show_keypoints, small_text=small_text)

  if not opt.no_display:
    cv2.imshow('SuperGlue matches', out)
    cv2.waitKey(0)
    key = chr(cv2.waitKey(1) & 0xFF)
    if key == 'q':
        vs.cleanup()
        print('Exiting (via q) demo_superglue.py')
    elif key == 'n':  # set the current frame as anchor
        last_data = {k+'0': pred[k+'1'] for k in keys}
        last_data['image0'] = frame_tensor
        last_frame = frame
        last_image_id = (vs.i - 1)
    elif key in ['e', 'r']:
        # Increase/decrease keypoint threshold by 10% each keypress.
        d = 0.1 * (-1 if key == 'e' else 1)
        matching.superpoint.config['keypoint_threshold'] = min(max(
            0.0001, matching.superpoint.config['keypoint_threshold']*(1+d)), 1)
        print('\nChanged the keypoint threshold to {:.4f}'.format(
            matching.superpoint.config['keypoint_threshold']))
    elif key in ['d', 'f']:
        # Increase/decrease match threshold by 0.05 each keypress.
        d = 0.05 * (-1 if key == 'd' else 1)
        matching.superglue.config['match_threshold'] = min(max(
            0.05, matching.superglue.config['match_threshold']+d), .95)
        print('\nChanged the match threshold to {:.2f}'.format(
            matching.superglue.config['match_threshold']))
    elif key == 'k':
        opt.show_keypoints = not opt.show_keypoints

if __name__ == '__main__':
  main()
