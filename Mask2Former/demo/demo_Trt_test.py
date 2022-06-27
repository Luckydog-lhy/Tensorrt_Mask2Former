import argparse
import glob
import multiprocessing as mp
import os

# fmt: off
import sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
# fmt: on

import tempfile
import time
import warnings

import cv2
import numpy as np
import tqdm

from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.utils.logger import setup_logger

from mask2former import add_maskformer2_config
from predictor import VisualizationDemo


# constants
WINDOW_NAME = "mask2former demo"


def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    return cfg


def get_parser():
    parser = argparse.ArgumentParser(description="maskformer2 demo for builtin configs")
    parser.add_argument(
        "--config-file",
        default="configs/coco/panoptic-segmentation/maskformer2_R50_bs16_50ep.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument("--webcam", action="store_true", help="Take inputs from webcam.")
    parser.add_argument("--video-input", help="Path to video file.")
    parser.add_argument(
        "--input",
        nargs="+",
        help="A  input images path ",
    )
    parser.add_argument(
        "--output",
        help="A file or directory to save output visualizations. "
        "If not given, will show output in an OpenCV window.",
    )

    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
        help="Minimum score for instance predictions to be shown",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )
    return parser


def test_opencv_video_format(codec, file_ext):
    with tempfile.TemporaryDirectory(prefix="video_format_test") as dir:
        filename = os.path.join(dir, "test_file" + file_ext)
        writer = cv2.VideoWriter(
            filename=filename,
            fourcc=cv2.VideoWriter_fourcc(*codec),
            fps=float(30),
            frameSize=(10, 10),
            isColor=True,
        )
        [writer.write(np.zeros((10, 10, 3), np.uint8)) for _ in range(30)]
        writer.release()
        if os.path.isfile(filename):
            return True
        return False
import torch
def model_time(args,cfg,TRT) -> object:
    model = VisualizationDemo(cfg)
    img = read_image(args.input[0], format="BGR")
    model.trace_model(img, TRT=TRT)
    for i in range(10):
        model.run_test(img,TRT=False)

    start_time, end_time= torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
    start_time.record()
    result_output = []
    for path in tqdm.tqdm(args.input, disable=not args.output):
        # use PIL, to be consistent with evaluation
        img = read_image(path, format="BGR")
        predictions = model.run_test(img,TRT=False)
        torch.cuda.synchronize()
        result_output.append(
            predictions.data.cpu().detach().numpy()
        )
    end_time.record()
    cost_time = start_time.elapsed_time(end_time)
    logger.info(
        "ori cost time {:.2f}ms".format(
            cost_time,
        )
    )

    for i in range(10):
        model.run_test(img,TRT=True)
    start_time, end_time = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    result_trt_output = []
    for path in tqdm.tqdm(args.input, disable=not args.output):
        # use PIL, to be consistent with evaluation
        img = read_image(path, format="BGR")
        predictions = model.run_test(img,TRT=True)
        result_trt_output.append(
            predictions.cpu().detach().numpy()
        )
    end_time.record()
    cost_time = start_time.elapsed_time(end_time)
    logger.info(
        "trt cost time {:.2f}ms".format(
            cost_time,
        )
    )
    return cost_time,result_output


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    args = get_parser().parse_args()
    setup_logger(name="fvcore")
    logger = setup_logger()
    logger.info("Arguments: " + str(args))

    cfg = setup_cfg(args)
    args.input = [os.path.join(args.input[0],input) for input in os.listdir(args.input[0])]
    # trt_cost_time, trt_output = model_time(args, cfg, TRT=True)
    ori_cost_time,ori_output = model_time(args, cfg, TRT=True)


    pass



