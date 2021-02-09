import uvicorn
from fastapi import FastAPI
from fastapi import File, Form, UploadFile
from starlette.responses import FileResponse
from starlette.middleware.cors import CORSMiddleware

from io import BytesIO
from PIL import Image
from base64 import b64encode

from mmdet.apis import init_detector, inference_detector, show_result_pyplot
import mmcv
import numpy as np
from mmcv.visualization.color import color_val
from mmcv.image import imread, imwrite

import shutil
from pathlib import Path

import cv2

import argparse
import os
import os.path as osp
import pdb
import random

from mmcv import Config

from mmdet.core import rotated_box_to_poly_single
from mmdet.datasets import build_dataset


def show_result_rbox(
    img,
    detections,
    class_names,
    scale=1.0,
    threshold=0.2,
    colormap=None,
    show_label=True,
):
    assert isinstance(class_names, (tuple, list))
    if colormap:
        assert len(class_names) == len(colormap)
    img = mmcv.imread(img)
    color_white = (255, 255, 255)

    for j, name in enumerate(class_names):
        if colormap:
            color = colormap[j]
        else:
            color = (
                random.randint(0, 256),
                random.randint(0, 256),
                random.randint(0, 256),
            )
        try:
            dets = detections[j]
        except:
            pdb.set_trace()
        # import ipdb;ipdb.set_trace()
        for det in dets:
            score = det[-1]
            det = rotated_box_to_poly_single(det[:-1])
            bbox = det[:8] * scale
            if score < threshold:
                continue
            bbox = list(map(int, bbox))

            for i in range(3):
                cv2.line(
                    img,
                    (bbox[i * 2], bbox[i * 2 + 1]),
                    (bbox[(i + 1) * 2], bbox[(i + 1) * 2 + 1]),
                    color=color,
                    thickness=2,
                    lineType=cv2.LINE_AA,
                )
            cv2.line(
                img,
                (bbox[6], bbox[7]),
                (bbox[0], bbox[1]),
                color=color,
                thickness=2,
                lineType=cv2.LINE_AA,
            )
            if show_label:
                cv2.putText(
                    img,
                    "%s %.3f" % (class_names[j], score),
                    (bbox[0], bbox[1] + 10),
                    color=color_white,
                    fontFace=cv2.FONT_HERSHEY_COMPLEX,
                    fontScale=0.5,
                )
    return img

dota_colormap = [
    (54, 67, 244),
    (99, 30, 233),
    (176, 39, 156),
    (183, 58, 103),
    (181, 81, 63),
    (243, 150, 33),
    (212, 188, 0),
    (136, 150, 0),
    (80, 175, 76),
    (74, 195, 139),
    (57, 220, 205),
    (59, 235, 255),
    (0, 152, 255),
    (34, 87, 255),
    (72, 85, 121),
]

hrsc2016_colormap = [
    (54, 67, 244)
]

models_dict = {
    "s2anet-dota": {
        "config": "/mnt/data/ntro-demo/satellite-backend/s2anet/models/s2anet_r50_fpn_1x_dota.py",
        "checkpoint": "/mnt/data/ntro-demo/satellite-backend/s2anet/models/s2anet_r50_fpn_1x_converted-11c9c5f4.pth",
        "colormap": dota_colormap
    },
    "s2anet-hrsc2016": {
        "config": "/mnt/data/ntro-demo/satellite-backend/s2anet/models/s2anet_r101_fpn_3x_hrsc2016.py",
        "checkpoint": "/mnt/data/ntro-demo/satellite-backend/s2anet/models/s2anet_r101_fpn_3x_hrsc2016_converted-4a4548e1.pth",
        "colormap": hrsc2016_colormap
    },
}

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def hello():
    return {"hello": "world"}


@app.post("/detect")
async def detect(
    image: UploadFile = File(...),
    threshold: float = Form(...),
    model_name: str = Form(...),
):

    model = init_detector(
        models_dict[model_name]["config"],
        models_dict[model_name]["checkpoint"],
        device="cuda:0",
    )

    file_location = Path("tmp")
    file_location.mkdir(parents=True, exist_ok=True)
    file_location = file_location / image.filename

    image.file.seek(0)
    with file_location.open("wb") as buffer:
        shutil.copyfileobj(image.file, buffer)

    result = inference_detector(model, str(file_location))

    out_img = show_result_rbox(str(file_location),
                       result,
                       model.CLASSES,
                       scale=1.0,
                       threshold=threshold,
                       colormap=models_dict[model_name]['colormap'])
    
    # line that fixed it
    _, encoded_img = cv2.imencode(".png", out_img)

    encoded_img = b64encode(encoded_img)

    scores = list(
        map(
            lambda x: [] if x.size == 0 else list(map(lambda y: y[4], x.tolist())),
            result,
        )
    )
    scores = list(
        map(lambda x: list(filter(lambda y: y >= float(threshold), x)), scores)
    )

    return {
        "result": encoded_img,
        "scores": scores,
        "labels": model.CLASSES,
    }
