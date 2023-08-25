import os
import sys
from pathlib import Path
import argparse
import time

import cv2
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split
import albumentations as albu
from albumentations.pytorch import ToTensor
import segmentation_models_pytorch as smp


class DemoLiverDataset(Dataset):
    """

    Args:
        images_fps (list):
        masks_fps (list):
        augmentation (albumentations.Compose):
        preprocessing (albumentations.Compose):
        classes (list):
    """

    CLASSES = ['unlabelled', 'liver']

    classes_dict = {'unlabelled': 0, 'liver': 255}

    def __init__(self, images_fps, masks_fps,
                 classes,
                 augmentation=None, preprocessing=None,
                 ):

        self.images_fps = images_fps
        self.masks_fps = masks_fps

        # convert str names to class valued on masks
        assert isinstance(classes, list)
        self.class_values = [self.classes_dict[cls.lower()] for cls in classes]

        # data augmentation and preprocessing数据增强和预处理
        self.augmentation = augmentation
        self.preprocessing = preprocessing

    def __len__(self):
        return len(self.images_fps)

    def __getitem__(self, index):
        _image = cv2.imread(str(self.images_fps[index]))
        _image = cv2.cvtColor(_image, cv2.COLOR_BGR2RGB)
        _mask = cv2.imread(str(self.masks_fps[index]), 0)  # orginal mask is single channel原始的mask图像就是单通道的

        # extract certain classes from mask (e.g. liver) 从mask中提取特定的类
        _masks = [(_mask == v) for v in self.class_values]
        _mask = np.stack(_masks, axis=-1).astype('float')  # covert int8 to float将原来的0-255区间转换为0-1区间

        # apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=_image, mask=_mask)
            _image, _mask = sample['image'], sample['mask']

        # apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=_image, mask=_mask)
            _image, _mask = sample['image'], sample['mask']

        sample = {'image': _image, 'label': _mask}

        # return sample
        return _image, _mask  # 为了配合segmentation_models_pytorch使用


def pre_transforms(image_size=224):
    return [albu.Resize(image_size, image_size, p=1)]


def hard_transforms():
    result = [
        albu.RandomRotate90(),
        albu.Cutout(),
        albu.RandomBrightnessContrast(
            brightness_limit=0.2, contrast_limit=0.2, p=0.3
        ),
        albu.GridDistortion(p=0.3),
        albu.HueSaturationValue(p=0.3)
    ]

    return result


# transforms 用于图像增强的变换
def resize_transforms(image_size=224):
    """resize transforms调整原始图像的大小用于训练

    Parameters
    ----------
    image_size :

    Returns
    -------

    """
    BORDER_CONSTANT = 0
    pre_size = int(image_size * 1.5)

    random_crop = albu.Compose([
        albu.SmallestMaxSize(pre_size, p=1),
        albu.RandomCrop(image_size, image_size, p=1)
    ])

    rescale = albu.Compose([albu.Resize(image_size, image_size, p=1)])

    random_crop_big = albu.Compose([
        albu.LongestMaxSize(pre_size, p=1),
        albu.RandomCrop(image_size, image_size, p=1)
    ])

    # Converts the image to a square of size image_size x image_size
    result = [albu.OneOf([random_crop, rescale, random_crop_big], p=1)]

    return result


# post transforms 后处理
def post_transforms():
    # we use ImageNet image normalization
    # and convert it to torch.Tensor
    # 使用ImageNet的标准化方法
    # 以及将样本转换为torch.Tensor
    return [albu.Normalize(), ToTensor()]


def compose(transforms_to_compose):
    # combine all augmentations into one single pipeline
    # 将所有变换处理为一个流程
    result = albu.Compose([
        item for sublist in transforms_to_compose for item in sublist
    ])
    return result


def get_training_augmentation():
    train_transform = [

        albu.HorizontalFlip(p=0.5),

        albu.ShiftScaleRotate(scale_limit=0.5, rotate_limit=0, shift_limit=0.1, p=1, border_mode=0),

        albu.PadIfNeeded(min_height=320, min_width=320, always_apply=True, border_mode=0),
        albu.RandomCrop(height=320, width=320, always_apply=True),

        albu.IAAAdditiveGaussianNoise(p=0.2),
        albu.IAAPerspective(p=0.5),

        albu.OneOf(
            [
                albu.CLAHE(p=1),
                albu.RandomBrightness(p=1),
                albu.RandomGamma(p=1),
            ],
            p=0.9,
        ),

        albu.OneOf(
            [
                albu.IAASharpen(p=1),
                albu.Blur(blur_limit=3, p=1),
                albu.MotionBlur(blur_limit=3, p=1),
            ],
            p=0.9,
        ),

        albu.OneOf(
            [
                albu.RandomContrast(p=1),
                albu.HueSaturationValue(p=1),
            ],
            p=0.9,
        ),
    ]
    return albu.Compose(train_transform)


def get_validation_augmentation():
    """Add paddings to make image shape divisible by 32"""
    test_transform = [
        albu.PadIfNeeded(384, 480)
    ]
    return albu.Compose(test_transform)


def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype('float32')


def get_preprocessing(preprocessing_fn):
    """Construct preprocessing transform

    Args:
        preprocessing_fn (callbale): data normalization function
            (can be specific for each pretrained neural network)
    Return:
        transform: albumentations.Compose

    """

    _transform = [
        albu.Lambda(image=preprocessing_fn),
        albu.Lambda(image=to_tensor, mask=to_tensor),
    ]
    return albu.Compose(_transform)


# helper function for data visualization
def visualize(**images):
    """PLot images in one row."""
    n = len(images)
    plt.figure(figsize=(16, 5))
    for i, (name, image) in enumerate(images.items()):
        plt.subplot(1, n, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(' '.join(name.split('_')).title())
        plt.imshow(image)
    plt.show()


# test 
def predict_image(net, img_path, show=False, save=False):
    # load data
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # preprocessing
    ENCODER = 'se_resnext50_32x4d'
    ENCODER_WEIGHTS = 'imagenet'
    preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)
    augmentation=get_validation_augmentation()
    preprocessing=get_preprocessing(preprocessing_fn)
    img_aug = preprocessing(image=augmentation(image=img)['image'])
    # Image.fromarray(img_aug['image'])
    image = img_aug['image']
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # processing
    x_tensor = torch.from_numpy(image).to(DEVICE).unsqueeze(0)
    
    # output
    pr_mask = net.predict(x_tensor)
    pr_mask = (pr_mask.squeeze().cpu().numpy().round())
    # type(pr_mask), pr_mask.shape
    if show:
        print(pr_mask.shape, type(pr_mask))
        plt.imshow(pr_mask)
        if save:
            plt.axes('off')
            plt.savefig(save, bbox_inches='tight')
            pass
        plt.show()
    
    return pr_mask

def main():
    # path of the datasets数据路径
    demo_data_path = {
        #     'train': r'D:\Projects\3DSegmentation\data\demo_data\train',
        #     'valid': r'D:\Projects\3DSegmentation\data\demo_data\train',
        'train': '../data/demo_data/train',
        'valid': '../data/demo_data/train',
        'test': '../data/demo_data/val'
    }
    train_path = Path(demo_data_path['train'])
    test_path = Path(demo_data_path['test'])

    # training image full path list 训练集数据完整路径列表
    train_image_list = list(train_path.glob("[0-9][0-9][0-9].png"))
    train_mask_list = list(train_path.glob("[0-9][0-9][0-9]_mask.png"))

    test_images_fps = list(test_path.glob("[0-9][0-9][0-9].png"))
    test_masks_fps = list(test_path.glob("[0-9][0-9][0-9]_mask.png"))

    # training set and valid set
    train_images_fps, valid_images_fps, train_masks_fps, valid_masks_fps = train_test_split(train_image_list,
                                                                                            train_mask_list,
                                                                                            random_state=30,
                                                                                            test_size=0.25)
    print("num of training set: {}".format(len(train_images_fps)))
    print("num of valid set: {}".format(len(valid_images_fps)))
    print("num of test set: {}".format(len(test_images_fps)))

    # Train
    CLASSES = ['liver']
    # models
    ENCODER = 'se_resnext50_32x4d'
    ENCODER_WEIGHTS = 'imagenet'

    ACTIVATION = 'sigmoid'  # could be None for logits or 'softmax2d' for multicalss segmentation
    DEVICE = 'cuda'

    # create segmentation model with pretrained encoder
    model = smp.FPN(
        encoder_name=ENCODER,
        encoder_weights=ENCODER_WEIGHTS,
        classes=len(CLASSES),
        activation=ACTIVATION,
    )

    preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)

    train_dataset = DemoLiverDataset(
        train_images_fps, train_masks_fps, classes=CLASSES,
        augmentation=get_training_augmentation(),
        preprocessing=get_preprocessing(preprocessing_fn),
    )
    valid_dataset = DemoLiverDataset(
        valid_images_fps, valid_masks_fps, classes=CLASSES,
        augmentation=get_validation_augmentation(),
        preprocessing=get_preprocessing(preprocessing_fn),
    )

    # Dataloader
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=0)
    valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=0)

    # Dice/F1 score - https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient
    # IoU/Jaccard score - https://en.wikipedia.org/wiki/Jaccard_index
    loss = smp.utils.losses.DiceLoss()
    metrics = [
        smp.utils.metrics.IoU(threshold=0.5),
    ]

    optimizer = torch.optim.Adam([
        dict(params=model.parameters(), lr=0.0001),
    ])

    # create epoch runners
    # it is a simple loop of iterating over dataloader`s samples
    train_epoch = smp.utils.train.TrainEpoch(
        model,
        loss=loss,
        metrics=metrics,
        optimizer=optimizer,
        device=DEVICE,
        verbose=True,
    )

    valid_epoch = smp.utils.train.ValidEpoch(
        model,
        loss=loss,
        metrics=metrics,
        device=DEVICE,
        verbose=True,
    )

    # train model for 40 epochs
    max_score = 0
    for i in range(0, 10):
        print('\nEpoch: {}'.format(i))
        train_logs = train_epoch.run(train_loader)
        valid_logs = valid_epoch.run(valid_loader)

        # do something (save model, change lr, etc.)
        if max_score < valid_logs['iou_score']:
            max_score = valid_logs['iou_score']
            torch.save(model, './liver_seg_best_model.pth')
            print('Model saved!')

        if i == 5:
            optimizer.param_groups[0]['lr'] = 1e-5
            print('Decrease decoder learning rate to 1e-5!')

    # Test
    # load best saved checkpoint
    best_model = torch.load('./liver_seg_best_model.pth')

    test_dataset = DemoLiverDataset(
        images_fps=test_images_fps,
        masks_fps=test_masks_fps,
        augmentation=get_validation_augmentation(),
        preprocessing=get_preprocessing(preprocessing_fn),
        classes=CLASSES,
    )

    test_dataset_vis = DemoLiverDataset(
        images_fps=test_images_fps,
        masks_fps=test_masks_fps,
        classes=CLASSES,
    )

    for i in range(5):
        n = np.random.choice(len(test_dataset_vis))

        image_vis = test_dataset_vis[n][0].astype('uint8')
        image, gt_mask = test_dataset[n]

        gt_mask = gt_mask.squeeze()

        x_tensor = torch.from_numpy(image).to(DEVICE).unsqueeze(0)
        pr_mask = best_model.predict(x_tensor)
        pr_mask = (pr_mask.squeeze().cpu().numpy().round())

        visualize(
            image=image_vis,
            ground_truth_mask=gt_mask,
            predicted_mask=pr_mask
        )


if __name__ == '__main__':
    # main()
    parser = argparse.ArgumentParser("Liver segmentation")
    parser.add_argument('-i', '--input', required=True, help='path of an image')
    parser.add_argument('-ckpt', '--checkpoint', required=True, help='path of a model checkpoint')
    parser.add_argument('-o', '--output', required=False, default=False, help='name of result')
    # ckpt = r'D:\home\intern_project\3d_segmentation_v2\examples\liver_seg_best_model.pth'
    args = parser.parse_args()
    print(args)
    time = time.time()
    net = torch.load(args.checkpoint)
    print(net)
    img_path = args.input
    save_to = args.output
    mask_pred = predict_image(net, img_path, show=True, save=save_to)


