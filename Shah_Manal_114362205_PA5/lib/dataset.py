'''
FSC147 Dataset Loader and Preprocess
By Yifeng Huang(yifehuang@cs.stonybrook.edu)
Based on Viresh and Minh's code
Last Modified 2021.9.2
'''
import os
import cv2
import json
import torch
import numpy as np
from PIL import Image
from lib.utils import gauss2D_density
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader


class FscBgDataset(Dataset):
    """Fewshot counting dataset with background FSC-147"""

    def __init__(self, root_dir, data_split, transform = None, use_negative_stroke = False, use_residual_learning = False):
        """
        Args:
            root_dir (string): Directory with all the images.
            data_split (string): either 'train', 'val', or 'test'
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        assert data_split in ['train', 'val', 'test']
        anno_file = os.path.join(root_dir, 'json_annotationCombined_384_VarV2.json')
        data_split_file = os.path.join(root_dir, 'Train_Test_Val_FSC_147.json')

        with open(anno_file) as f:
            self.annotations = json.load(f)

        with open(data_split_file) as f:
            data_split_ids = json.load(f)

        self.im_dir = os.path.join(root_dir, 'images_384_VarV2')
        self.gt_dir = os.path.join(root_dir, 'gt_density_map_adaptive_384_VarV2')
        if use_negative_stroke:
            self.bgmask_dir = os.path.join(root_dir, 'BgMask')

        self.im_ids = data_split_ids[data_split]
        self.transform = transform
        self.use_negative_stroke = use_negative_stroke
        self.use_residual_learning = use_residual_learning

    def __len__(self):
        return len(self.im_ids)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        im_id = self.im_ids[idx]
        anno = self.annotations[im_id]

        bboxes = anno['box_examples_coordinates']
        dots = np.array(anno['points'])

        rects = list()
        for bbox in bboxes:
            x1 = bbox[0][0]
            y1 = bbox[0][1]
            x2 = bbox[2][0]
            y2 = bbox[2][1]
            rects.append([y1, x1, y2, x2])

        image = Image.open('{}/{}'.format(self.im_dir, im_id))
        image.load()

        W, H = image.size
        dots = np.maximum(dots, 0)
        dots[:, 0] = np.minimum(dots[:, 0], W - 1)
        dots[:, 1] = np.minimum(dots[:, 1], H - 1)

        # Get GT Density
        density_path = os.path.join(self.gt_dir, im_id.split(".jpg")[0] + ".npy")
        density = np.load(density_path).astype('float32')

        if self.use_negative_stroke:
            bg_mask_img = cv2.imread('{}/{}_anno.jpg'.format(self.bgmask_dir, im_id[:-4]), cv2.IMREAD_GRAYSCALE)
            # Process the Background Mask to a Binary Image
            bg_mask_img = np.array(bg_mask_img)
            bg_mask_img = np.int32(bg_mask_img > 0)
            bg_mask_img *= 255
            bg_mask_img = bg_mask_img.astype(np.float32)
        else:
            bg_mask_img = np.zeros(density.shape).astype(np.float32)

        #Get Boxes
        boxes = np.array(rects)
        sample = {'im_id':im_id, 'image':image, 'dots':dots, 'boxes': boxes, 'bg_mask_img': bg_mask_img, 'gt_density':density}

        #Get Kernel-Resdiual-Learning Target and Mask
        if self.use_residual_learning:
            W, H = image.size
            bg_mask_img_res = Image.open('{}/{}_anno.jpg'.format(self.bgmask_dir, im_id[:-4]))
            bg_mask_img_res.load()
            num_bbox = boxes.shape[0]
            target = np.zeros(shape=(H, W))
            masks = np.zeros(shape=(H, W, num_bbox + 1))
            hws = np.zeros(shape=(num_bbox, 2))
            for i in range(num_bbox):
                bbox = boxes[i, :]
                y1, x1, y2, x2 = bbox[0], bbox[1], bbox[2], bbox[3]
                h = y2 - y1
                w = x2 - x1
                target[y1:y2, x1:x2] = gauss2D_density(shape=(h, w), sigmas=(h / 2, w / 2))
                masks[y1:y2, x1:x2, i] = 1
                hws[i, :] = [h, w]
            bg_mask = np.array(bg_mask_img_res)
            bg_mask = bg_mask.sum(axis=2) > 128
            bg_mask_im = np.zeros(shape=(H, W))
            bg_mask_im[bg_mask] = 1
            masks[:, :, num_bbox] = bg_mask_im
            sample["target_masks"] = masks
            sample["target"] = target
            sample["hws"] = hws

        if self.transform:
            sample = self.transform(sample)

        return sample
