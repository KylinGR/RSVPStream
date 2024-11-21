import os
import urllib
import traceback
import time
import sys
import numpy as np
from rknn.api import RKNN
import torch.nn as nn
import torch

ONNX_MODEL = "./model_local.onnx"
RKNN_MODEL = "./model_local.rknn"
DATASET = "./dataset.txt"           # 没用到
IMAGE = "./dataset/test/test.jpg"   # 没用到
QUANTIZE_ON = False

# intput_batchsize = 32
# intput_row = 250
# intput_line = 60

intput_row = 32
intput_line = 15000


def export_rknn_inference():
    # Create RKNN object
    rknn = RKNN(verbose=True)

    # pre-process config
    print("--> Config model")
    rknn.config(
        # mean_values=[[0, 0, 0]],
        # std_values=[[255, 255, 255]],
        target_platform="rk3588",
    )
    print("done")

    # Load ONNX model
    print("--> Loading model")
    ret = rknn.load_onnx(model=ONNX_MODEL)
    if ret != 0:
        print("Load model failed!")
        exit(ret)
    print("done")

    # Build model
    print("--> Building model")
    ret = rknn.build(do_quantization=QUANTIZE_ON, dataset=DATASET, rknn_batch_size=1)
    if ret != 0:
        print("Build model failed!")
        exit(ret)
    print("done")

    # Export RKNN model
    print("--> Export rknn model")
    ret = rknn.export_rknn(RKNN_MODEL)
    if ret != 0:
        print("Export rknn model failed!")
        exit(ret)
    print("done")
    print("`````````````````````````````````````````````````````````````")
    # Init runtime environment
    print("--> Init runtime environment")
    ret = rknn.init_runtime()
    if ret != 0:
        print("Init runtime environment failed!")
        exit(ret)

    # inference here
    np.random.seed(0)
    # random_matrix = np.random.randn(intput_batchsize, intput_row, intput_line).astype(np.float32)
    random_matrix = np.random.randn(intput_row, intput_line).astype(np.float32)
    print("Matrix shape:", random_matrix.shape)
    outputs = rknn.inference(inputs=[random_matrix])
    output_numpy0 = outputs[0]
    # output_numpy1 = outputs[1]
    print("Output0 shape:", output_numpy0.shape)
    # print("Output1 shape:", output_numpy1.shape)
    print("inference done")

    rknn.release()
    print("release done")


if __name__ == "__main__":
    print("This is main ...")
    outputs = export_rknn_inference()

