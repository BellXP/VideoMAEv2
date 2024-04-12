import io

import cv2
import numpy as np
from decord import VideoReader, cpu

try:
    from petrel_client.client import Client
    petrel_backend_imported = True
except (ImportError, ModuleNotFoundError):
    petrel_backend_imported = False


def get_video_loader(use_petrel_backend: bool = True,
                     enable_mc: bool = True,
                     conf_path: str = None):
    if petrel_backend_imported and use_petrel_backend:
        _client = Client(conf_path=conf_path, enable_mc=enable_mc)
    else:
        _client = None

    def _loader(video_path):
        if _client is not None and 's3:' in video_path:
            video_path = io.BytesIO(_client.get(video_path))

        vr = VideoReader(video_path, num_threads=1, ctx=cpu(0))
        return vr

    return _loader


def get_image_loader(use_petrel_backend: bool = True,
                     enable_mc: bool = True,
                     conf_path: str = None):
    if petrel_backend_imported and use_petrel_backend:
        _client = Client(conf_path=conf_path, enable_mc=enable_mc)
    else:
        _client = None

    def _loader(frame_path, dsize=None):
        if _client is not None and 's3:' in frame_path:
            img_bytes = _client.get(frame_path)
        else:
            with open(frame_path, 'rb') as f:
                img_bytes = f.read()

        img_np = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(img_np, cv2.IMREAD_COLOR)
        img = cv2.resize(img, dsize)
        cv2.cvtColor(img, cv2.COLOR_BGR2RGB, img)
        return img

    return _loader


def get_skeleton_image_loader():
    from PIL import Image
    from scipy import interpolate

    # NOTE: kind should be in ['linear', 'cubic', 'quintic']
    def interpolate_img(img, old_size, new_size, kind='linear'):
        y = np.asarray([i * (1.0 / img.shape[0]) for i in range(img.shape[0])])
        x = np.asarray([i * (1.0 / img.shape[1]) for i in range(img.shape[1])])
        f = interpolate.interp2d(x, y, img, kind=kind)
        new_y = np.arange(0, 1, 1 / new_size[0])
        new_x = np.arange(0, 1, 1 / new_size[1])
        return f(new_x, new_y)

    mean=[0.485, 0.456, 0.406]
    std=[0.229, 0.224, 0.225]

    def norm_depth_image(img):
        max_depth = img.max()
        img_copy = img.copy()
        img_copy[img_copy == 0] = max_depth + 1
        min_depth = img_copy.min()
        depth_range = max_depth - min_depth
        if depth_range > 0:
            img[img != 0] -= min_depth
            img = img / depth_range
        img = img * std + mean
        return img

    def _loader(frame_path, dsize=None):
        img_np = np.asarray(Image.open(frame_path))
        y_range, x_range = img_np.shape
        if dsize is not None and (y_range != dsize[0] or x_range != dsize[1]):
            img_np = interpolate_img(img_np, (y_range, x_range), dsize)
        img_np = np.repeat(np.expand_dims(img_np, axis=2), 3, axis=2)
        img_np = norm_depth_image(img_np).astype(np.float32)
        return img_np

    return _loader