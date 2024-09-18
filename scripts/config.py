import argparse
import cv2

def parse_args():
    print('Parsing arguments ...')
    parser = argparse.ArgumentParser(
      description='Arguments for SuperGlue',
      formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--input',                type=str,                      default='0',                         help='ID of a USB webcam, URL of an IP camera, or path to an image directory or movie file')
    parser.add_argument('--output_dir',           type=str,                      default=None,                        help='Directory where to write output frames (If None, no output)')
    parser.add_argument('--image_glob',           type=str,      nargs='+',      default=['*.png','*.jpg','*.jpeg'],  help='Glob if a directory of images is specified')
    parser.add_argument('--skip',                 type=int,                      default=1,                           help='Images to skip if input is a movie or directory')
    parser.add_argument('--max_length',           type=int,                      default=1000000,                     help='Maximum length if input is a movie or directory')
    parser.add_argument('--resize',               type=int,      nargs='+',      default=[480, 640],                  help='Resize the input image before running inference. If two numbers, '
                                                                                                                           'resize to the exact dimensions, if one number, resize the max '
                                                                                                                           'dimension, if -1, do not resize')
    parser.add_argument('--superglue',            choices={'indoor', 'outdoor'}, default='outdoor',                   help='SuperGlue weights')
    parser.add_argument('--max_keypoints',        type=int,                      default=-1,                          help='Maximum number of keypoints detected by Superpoint(\'-1\' keeps all keypoints)')
    parser.add_argument('--keypoint_threshold',   type=float,                    default=0.005,                       help='SuperPoint keypoint detector confidence threshold')
    parser.add_argument('--nms_radius',           type=int,                      default=4,                           help='SuperPoint Non Maximum Suppression (NMS) radius (Must be positive)')
    parser.add_argument('--sinkhorn_iterations',  type=int,                      default=20,                          help='Number of Sinkhorn iterations performed by SuperGlue')
    parser.add_argument('--match_threshold',      type=float,                    default=0.2,                         help='SuperGlue match threshold')
    parser.add_argument('--show_keypoints',       action='store_true',                                                help='Show the detected keypoints')
    parser.add_argument('--no_display',           action='store_true',                                                help='Do not display images to screen. Useful if running remotely')
    parser.add_argument('--force_cpu',            action='store_true',                                                help='Force pytorch to run in CPU mode.')

    args = parser.parse_args()
    if len(args.resize) == 2 and args.resize[1] == -1:
      args.resize = args.resize[0:1]
    if len(args.resize) == 2:
      print('Will resize to {}x{} (WxH)'.format(
             args.resize[0], args.resize[1]))
    elif len(args.resize) == 1 and args.resize[0] > 0:
      print('Will resize max dimension to {}'.format(args.resize[0]))
    elif len(args.resize) == 1:
      print('Will not resize images')
    else:
      raise ValueError('Cannot specify more than two integers for --resize')

    if args.output_dir is not None:
      print('==> Will write outputs to {}'.format(args.output_dir))
      Path(args.output_dir).mkdir(exist_ok=True)

    # Create a window to display the demo.
    if not args.no_display:
        cv2.namedWindow('SuperGlue matches', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('SuperGlue matches', 640*2, 480)
    else:
        print('Skipping visualization, will not show a GUI.')
    return args
