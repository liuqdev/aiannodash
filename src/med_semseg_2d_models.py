import os
import glob
import time 
import pickle
from pathlib import Path
from PIL import Image

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
from sklearn.model_selection import train_test_split

from src.data import LiverDataset, blend, Pad, Crop, Resize
from src.models import UNet, PretrainedUNet
from src.metrics import jaccard, dice

# 设置GPU
on_server = False
data_parallel = True
gpu_ids = '0'
if on_server:
    gpu_ids = '1,2,3,4,5,6'
os.environ['CUDA_VISIBLE_DEVICES'] = gpu_ids

#
# # 一些参数变量
# IMG_HEIGTH, IMG_WIDTH = 512, 512  # 制定图像大小
# origin_filename = r"../data/liver_infer/005.png"
# # origin_filename = r"data/liver_data/train/images/008.png"
#
# data_folder = Path("data", "liver_data", "train")
# origins_folder = data_folder / "images"
# masks_folder = data_folder / "masks"
# models_folder = Path("../pretrained")
# images_folder = Path("../images")
# ckpt_name = "unet_liver_seg_torch{}x{}.pt".format(IMG_HEIGTH, IMG_WIDTH)


class MedSeg:

    def __init__(
            self,
            model_name,  # 选择模型
            ckpt_name,  # 权重路径
            data_parallel=True,
            gpu_ids='0',
            on_server=False,
            device='cuda'
    ):
        self.model_name = model_name
        self.ckpt_name = ckpt_name
        self.data_parallel = data_parallel
        self.gpu_ids = gpu_ids
        self.on_server = on_server

        if torch.cuda.is_available() and device=='cuda':
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")


    def load_model(self):
        # 选择模型
        if self.model_name.lower() in ['unet']:
            net = UNet(in_channels=1, out_channels=2, batch_norm=True)
            # ckpt_name = "unet_liver_seg_torch{}x{}.pt".format(IMG_HEIGTH, IMG_WIDTH)
        elif self.model_name.lower() in ['unet_vgg11']:
            net = PretrainedUNet(in_channels=1, out_channels=2, batch_norm=True)
            #ckpt_name = "pretrained_unet_liver_seg_torch{}x{}.pt".format(IMG_HEIGTH, IMG_WIDTH)


        # print(net)
        if len(self.gpu_ids) > 1:
            net = torch.nn.DataParallel(net)

        # 加载权重
        print('loading checkpoint... ', self.ckpt_name)
        # 在本地， 但是权重是data parallel得到的
        if self.data_parallel:  # 1 权重是data parallel， 0 单个GPU
            if not on_server:
                net.load_state_dict({k.replace('module.', ''): v for k, v in torch.load(self.ckpt_name).items()})
            else:
                net.load_state_dict(torch.load(self.ckpt_name, map_location=torch.device("cpu")))
        else:
            net.load_state_dict(torch.load(self.ckpt_name, map_location=torch.device("cpu")))

        net.to(self.device)
        net.eval()

        return net

    def predict(self, origin_filename, img_size=(512, 512), display=False, save_to=None):

        ret = {}

        # 加载数据
        if isinstance(origin_filename, str):
            origin = Image.open(origin_filename).convert("P")
        elif isinstance(origin_filename, np.ndarray):
            origin = Image.fromarray(origin_filename).convert("P")

        origin = torchvision.transforms.functional.resize(origin, img_size)
        origin = torchvision.transforms.functional.to_tensor(origin) - 0.5


        print('loading model ...')
        net = self.load_model()


        # 预测
        with torch.no_grad():
            origin = torch.stack([origin])
            origin = origin.to(self.device)
            out = net(origin)
            softmax = torch.nn.functional.log_softmax(out, dim=1)
            out = torch.argmax(softmax, dim=1)
            origin = origin[0].to("cpu")
            out = out[0].to("cpu")

            ret['origin'] = torchvision.transforms.functional.to_pil_image(origin + 0.5).convert("RGB")
            ret['out'] = out.numpy()

        img_blended = blend(origin, out)
        ret['blended'] = img_blended

        if save_to:
            img_blended.save(save_to)

        if display:
            # 可视化
            plt.figure(figsize=(20, 10))
            pil_origin = torchvision.transforms.functional.to_pil_image(origin + 0.5).convert("RGB")

            plt.subplot(1, 2, 1)
            plt.title("origin image")
            plt.grid(False)
            plt.imshow(np.array(pil_origin))

            plt.subplot(1, 2, 2)
            plt.title("blended origin + predict")
            plt.grid(False)
            img_blended = blend(origin, out)

            plt.imshow(np.array(img_blended))
            if save_to:
                plt.savefig(save_to)
            plt.show()

        return ret


def run(input_image, model_name, ckpt_name, img_size=(512, 512)):
    med_seg = MedSeg(model_name=model_name, ckpt_name=ckpt_name)
    result = med_seg.predict(
        origin_filename=input_image,
        img_size=img_size,
        display=False,
        save_to=None,
    )

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return result


def main():
    IMG_HEIGTH, IMG_WIDTH = 512, 512  # 制定图像大小
    models_folder = Path("../pretrained")
    ckpt_name = models_folder / "unet_liver_seg_torch{}x{}.pt".format(IMG_HEIGTH, IMG_WIDTH)

    input_image = r"../data/liver_infer/100.png"

    model_names = ['unet', 'unet_vgg11']
    model_name = model_names[0]
    med_seg = MedSeg(model_name=model_name, ckpt_name=ckpt_name)

    result = med_seg.predict(origin_filename=input_image, display=True, save_to='.')

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()