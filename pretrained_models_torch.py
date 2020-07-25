import re
import os
import torch
import glob
import numpy as np
from urllib.parse import urlparse
from PIL import Image

import torch
import matplotlib.pyplot as plt
import torchvision.transforms as T
import torch.utils.model_zoo as model_zoo
from torchvision import models


def download_model(url, dst_path):
    parts = urlparse(url)
    filename = os.path.basename(parts.path)
    
    HASH_REGEX = re.compile(r'-([a-f0-9]*)\.')
    hash_prefix = HASH_REGEX.search(filename).group(1)
    model_zoo._download_url_to_file(url, os.path.join(dst_path, filename), hash_prefix, True)
    
    return filename


def load_model(model_name, ckpts_dir):
    model  = eval('models.%s(init_weights=False)' % model_name)
    path_format = os.path.join(ckpts_dir, '{}_[a-z0-9]*.pth'.format(model_name))
    model_path = glob.glob(path_format)[0]
    
    model.load_state_dict(torch.load(model_path))
    return model


# 自然图像分割任务的几种模型
class COCOPretrainedModels:

    model_urls = {
        'fcn_resnet50_coco': 'https://download.pytorch.org/models/fcn_resnet50_coco-1167a1af.pth',
        'fcn_resnet101_coco': 'https://download.pytorch.org/models/fcn_resnet101_coco-7ecb50ca.pth',
        'deeplabv3_resnet50_coco': 'https://download.pytorch.org/models/deeplabv3_resnet50_coco-cd0a2569.pth',
        'deeplabv3_resnet101_coco': 'https://download.pytorch.org/models/deeplabv3_resnet101_coco-586e9e4e.pth',        
    }

    # 数据预处理统一规格后输入到网络
    trf = T.Compose([
        T.Resize(256),
        # T.CenterCrop(224), 
        T.ToTensor(), 
        # 由于使用的是在imagenet上预训练的权重, 所以这里mean, std是imagenet数据集全体的
        # RGB三个通道上的均值和方差, 用来进行normalization
        T.Normalize(
            mean = [0.485, 0.456, 0.406], 
            std = [0.229, 0.224, 0.225])
        ]
    )

    def load_model(self, model_name, ckpts_dir=None):
        model = None
        if model_name.lower() not in {'fcn_resnet101', 'deeplabv3_resnet101'}:
            print("No implementation error, please input deeplabv3_resnet101 or fcn_resnet101")
            return
        # 给定了checkpoint的路径
        if ckpts_dir:
            checkpoint_path = glob.glob(ckpts_dir+'/{}_[a-z0-9]*.pth'.format(model_name))
            checkpoint = torch.load(checkpoint_path, strict=False)
            if model_name.lower()=='fcn_resnet101':
                fcn_resnet101 = models.segmentation.fcn_resnet101(pretrained=False)
                fcn_resnet101.load_state_dict(checkpoint, strict=False)
                model=fcn_resnet101.eval()
            elif model_name.lower()=='deeplabv3_resnet101':
                deeplabv3_resnet101 = models.segmentation.deeplabv3_resnet101(pretrained=False)
                deeplabv3_resnet101.load_state_dict(checkpoint, strict=False)
                model=deeplabv3_resnet101.eval()
        else:
            if model_name.lower()=='fcn_resnet101':
                fcn_resnet101 = models.segmentation.fcn_resnet101(pretrained=True)
                model = fcn_resnet101.eval()
            elif model_name.lower()=='deeplabv3_resnet101':
                deeplabv3_resnet101 = models.segmentation.deeplabv3_resnet101(pretrained=True)
                model=deeplabv3_resnet101.eval()
        
        return model


    # Define the helper function
    # COCO dataset colors (仅仅只对COCO数据集的21类进行颜色编码)
    def decode_segmap(self, image, nc=21):

        label_colors = np.array([(0, 0, 0),  # 0=background
                # 1=aeroplane, 2=bicycle, 3=bird, 4=boat, 5=bottle
                (128, 0, 0), (0, 128, 0), (128, 128, 0), (0, 0, 128), (128, 0, 128),
                # 6=bus, 7=car, 8=cat, 9=chair, 10=cow
                (0, 128, 128), (128, 128, 128), (64, 0, 0), (192, 0, 0), (64, 128, 0),
                # 11=dining table, 12=dog, 13=horse, 14=motorbike, 15=person
                (192, 128, 0), (64, 0, 128), (192, 0, 128), (64, 128, 128), (192, 128, 128),
                # 16=potted plant, 17=sheep, 18=sofa, 19=train, 20=tv/monitor
                (0, 64, 0), (128, 64, 0), (0, 192, 0), (128, 192, 0), (0, 64, 128)])

        r = np.zeros_like(image).astype(np.uint8)
        g = np.zeros_like(image).astype(np.uint8)
        b = np.zeros_like(image).astype(np.uint8)
    
        for l in range(0, nc):
            idx = image == l
            r[idx] = label_colors[l, 0]
            g[idx] = label_colors[l, 1]
            b[idx] = label_colors[l, 2]
        
        rgb = np.stack([r, g, b], axis=2)
        return rgb


    def segment(self, net, img_path, show_orig=False, show_output=False, dev='cuda'):
        dev = torch.device('cuda' if dev=='cuda' else 'cpu')
        img = Image.open(img_path)
        if show_orig: 
            plt.imshow(img)
            # plt.axis('off')
            plt.show()
        # Comment the Resize and CenterCrop for better inference results
        trf = self.trf
        inp = trf(img).unsqueeze(0).to(dev)
        out = net.to(dev)(inp)['out']
        om = torch.argmax(out.squeeze(), dim=0).detach().cpu().numpy()
        rgb = self.decode_segmap(om)
        if show_output:
            plt.imshow(rgb)
            # plt.axis('off')
            plt.show()    


    def infer_time(self, net, img_path, dev='cuda'):
        import time
        img = Image.open(img_path)
        trf = self.trf
        
        inp = trf(img).unsqueeze(0).to(dev)
        
        st = time.time()
        out1 = net.to(dev)(inp)
        et = time.time()
        
        return et - st


    

def main():
    coco_semseg = COCOPretrainedModels()
    # model = coco_semseg.load_model(model_name='fcn_resnet101')
    model = coco_semseg.load_model(model_name='deeplabv3_resnet101')
    print(model)

    coco_semseg.segment(net=model, 
        img_path=r'D:\home\intern_project\AIAnnoLab\aiannodash\data\coco_samples\person.png',
        show_orig=True,
        show_output=True
    )


if __name__ == '__main__':
    main()
    