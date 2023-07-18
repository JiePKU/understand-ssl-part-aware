# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import nntplib
import os
import random
import glob
import json
import torch
import numpy as np
from PIL import Image
from torchvision import transforms, get_image_backend



#### travel the image folder
# TODO: specify the return type
def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def pil_loader(path: str) -> Image.Image:
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def default_loader(path):
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)

class PatchLoader(torch.utils.data.Dataset):
    def __init__(self, image_root='/home/ssd6/cxk/CAE/imagenet-samples_10K/', patch_root='./patches/', split_ratio=4, input_size=224,
                 num_samples_per_cls=5, image_root_all='/home/ssd9/wangxiaodi03/workspace/wangxiaodi03/data/imagenet/val/',
                 mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        self.image_root = image_root
        self.patch_root = patch_root
        self.image_root_all = image_root_all

        self.num_samples_per_cls = num_samples_per_cls

        self.input_size = input_size
        self.total_size = input_size * split_ratio
        self.patch_stride = self.input_size // 2

        self.mean = mean
        self.std = std

        self.transforms1 = transforms.Compose([
            transforms.Resize(256, interpolation=3),
            transforms.CenterCrop(input_size),])
        self.transforms2 = transforms.Compose([
            transforms.Resize(self.total_size, interpolation=3),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)])

        image_path_list = []
        if num_samples_per_cls == 5:
            for path, dir_list, file_list in os.walk(image_root):
                for file_name in file_list:
                    image_name = os.path.join(path, file_name)
                    if 'JPEG' in image_name:
                        image_path_list.append(image_name)
        else:
            for dir_name in os.listdir(image_root_all):
                dir_path = os.path.join(image_root_all, dir_name)
                for image_name in os.listdir(dir_path):
                    if 'JPEG' in image_name and int(image_name.split('.')[0].split('_')[2]) < 25000 or \
                            int(image_name.split('.')[0].split('_')[2]) in (44160, 41979, 39553, 13660, 2144, 24863, 21756, 25793, 13660):
                        image_path_list.append(os.path.join(dir_path, image_name))
        self.image_path_list = image_path_list

        self.to_pil = transforms.ToPILImage()

    def __len__(self):
        return len(self.image_path_list)

    def torch_img_to_PIL(self, img):
        mean = torch.tensor(self.mean).unsqueeze(-1).unsqueeze(-1)
        std = torch.tensor(self.std).unsqueeze(-1).unsqueeze(-1)
        img = img * std + mean
        return self.to_pil(img).convert('RGB')

    def split_patches(self, img):
        patches = []
        patches_pil = []
        pos_params = []
        for t in range(0, self.total_size-self.input_size+1, self.patch_stride):
            for l in range(0, self.total_size-self.input_size+1, self.patch_stride):
                patch = img[:, t:t+self.input_size, l:l+self.input_size]
                patches.append(patch)
                patches_pil.append(self.torch_img_to_PIL(patch))
                pos_params.append((t, t+self.input_size, l, l+self.input_size))
        return patches, patches_pil, pos_params

    def __getitem__(self, i):
        img_path = self.image_path_list[i]
        img_name = os.path.basename(img_path).split('.')[0]

        img_pil = default_loader(img_path)
        img_pil = self.transforms1(img_pil)
        img = self.transforms2(img_pil)

        patches, patches_pil, pos_params = self.split_patches(img)

        img_root = os.path.join(self.patch_root, img_name)
        if not os.path.exists(os.path.join(img_root, 'origin.jpg')):
            os.makedirs(img_root, exist_ok=True)
            img_pil.save(os.path.join(img_root, 'origin.jpg'))
        patch_names = []
        for i in range(len(patches)):
            patch_name = '{}_{}_{}_{}.jpg'.format(*pos_params[i])
            patch_names.append(patch_name)
            if not os.path.exists(os.path.join(img_root, patch_name)):
                patches_pil[i].save(os.path.join(img_root, patch_name))

        patches = torch.stack(patches)
        return patches, img_name, patch_names

class PatchLoaderCUB200(torch.utils.data.Dataset):
    def __init__(self, patch_root='/home/ssd9/qjy/patch_embed_search/patches_cub200/', input_size=224,
                 data_root='/home/ssd9/qjy/patch_embed_search/CUB_200_2011/', mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225],
                 keypoint_cls_list=None, include_other_parts=False, training=False, use_retrieval_data=True):
        self.data_root = data_root
        self.part_name_list = [' '.join(line.strip().split()[1:]) for line in open(os.path.join(data_root, 'parts/parts.txt'))]
        self.image_list = [line.strip().split()[1] for line in open(os.path.join(data_root, 'images.txt'))]

        if not use_retrieval_data:
            image_id_list_split = set()
            for line in open(os.path.join(data_root, 'train_test_split.txt')):
                image_id, is_train = line.strip().split()
                if (is_train == '1') == training:
                    image_id_list_split.add(int(image_id))

        part_ann_file = os.path.join(data_root, 'parts/part_locs.txt')
        image_info_dict = {}
        image_id_list = []
        for line in open(part_ann_file):
            image_id, part_id, x, y, visible = line.strip().split()
            image_id, part_id, visible = int(image_id), int(part_id), int(visible)
            if not use_retrieval_data and image_id not in image_id_list_split:
                continue
            x, y = float(x), float(y)
            if visible:
                if part_id not in keypoint_cls_list:
                    continue
                if image_id not in image_info_dict:
                    image_info_dict[image_id] = []
                    image_id_list.append(image_id)
                image_info_dict[image_id].append([part_id, x, y])

        for image_id in image_info_dict:
            xy_list = [x[1:] for x in image_info_dict[image_id]]
            xy_list = np.array(xy_list)
            if not include_other_parts:
                dist = ((xy_list[:, None] - xy_list[None]) ** 2).sum(axis=2) ** 0.5
                dist[dist==0] = 999999
                dist = (dist.min(axis=1) / 2).tolist()
            else:
                dist = (xy_list[:, None] - xy_list[None]).max(axis=0).max(axis=0)  # the max dist along x/y axis
                dist = [dist.min(axis=0)] * len(xy_list)
            image_info_dict[image_id] = [image_info_dict[image_id][i] + [dist[i]] for i in range(len(dist)) if dist[i]>20 and dist[i]<1000]

        self.patch_info_list = []
        for image_id in image_id_list:
            for part_id, x, y, dist in image_info_dict[image_id]:
                self.patch_info_list.append({
                    'image_path': os.path.join(self.data_root, 'images', self.image_list[image_id-1]),
                    'xyd': (x, y, dist),
                    'part_id': part_id,
                })

        self.input_size = input_size
        if training:
            self.transforms = transforms.Compose([
                transforms.Resize(256, interpolation=3),
                transforms.CenterCrop(input_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ])
        else:
            self.transforms = transforms.Compose([
                transforms.Resize(input_size, interpolation=3),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std)])
        self.patch_root = patch_root
        self.part_name_map = {
            'right eye': 'eye',
            'right leg': 'leg',
            'left wing': 'wing',
            'tail': 'tail',
        }
        self.partid2clsid = {part_id: i for i, part_id in enumerate(keypoint_cls_list)}

    def __len__(self):
        return len(self.patch_info_list)

    def save(self, img, path):
        if not os.path.exists(path):
            img.save(path)

    def __getitem__(self, idx):
        patch_info = self.patch_info_list[idx]
        image_path = patch_info['image_path']
        x, y, dist = patch_info['xyd']
        part_name = self.part_name_list[patch_info['part_id']-1]
        dist = round(dist / 2)
        box = (x-dist, y-dist, x+dist, y+dist)
        img = default_loader(image_path)
        patch = img.crop(box)

        image_name = '/'.join(image_path.split('/')[-2:])[:-4]
        patch_root = os.path.join(self.patch_root, image_name)
        os.makedirs(patch_root, exist_ok=True)
        patch_path = os.path.join(patch_root, part_name.replace(' ', '_')+'.jpg')

        """
        save the part image if needed for part classification
        """
        # self.save(img, os.path.join(patch_root, 'origin.jpg'))
        # self.save(patch, patch_path)

        patch = self.transforms(patch)
        return patch, patch_path, self.partid2clsid[patch_info['part_id']]

class PatchLoaderCOCO(torch.utils.data.Dataset):
    def __init__(self, patch_root='/home/ssd9/qjy/patch_embed_search/patches_moco/', input_size=224,
                 data_root='/home/ssd2/wangxiaodi03/data/mscoco/', mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225],
                 keypoint_cls_list=None, include_other_parts=False, training=False, use_retrieval_data=True):
        if training or use_retrieval_data:
            self.img_dir = os.path.join(data_root, 'images/train2017')
            ann_file = os.path.join(data_root, 'annotations/person_keypoints_train2017.json')
        else:
            self.img_dir = os.path.join(data_root, 'images/val2017')
            ann_file = os.path.join(data_root, 'annotations/person_keypoints_val2017.json')
        ann = json.load(open(ann_file))

        image_info_dict = {x['id']: x for x in ann['images']}
        image_id_list = []
        for x in ann['annotations']:
            image_id = x['image_id']
            i_list = []
            for i in range(len(x['keypoints'])//3):
                if i+1 not in keypoint_cls_list:  # only use the target keypoints
                    continue
                if x['keypoints'][3*i+2] < 2:  # only use the visible keypoints
                    continue
                i_list.append(i)
            if len(i_list) < 2:  # more than one keypoint for each person, making sure the calculated distance is valid
                continue
            if 'ann' not in image_info_dict[image_id]:
                image_info_dict[image_id]['ann'] = []
                image_id_list.append(image_id)
            image_info_dict[image_id]['ann'].extend([([i+1] + x['keypoints'][3*i: 3*i+2] + [x['id']]) for i in i_list])

        for image_id in image_info_dict:
            if 'ann' not in image_info_dict[image_id]:
                continue
            xy_list = [x[1:] for x in image_info_dict[image_id]['ann']]
            xy_list = np.array(xy_list)
            if not include_other_parts:
                dist = ((xy_list[:, None] - xy_list[None]) ** 2).sum(axis=2) ** 0.5
                dist[dist==0] = 999999
                # dist = (dist.min(axis=1) / 1.6).tolist()
                dist = (dist.min(axis=1) / 2).tolist()
            else:
                dist = (xy_list[:, None] - xy_list[None]).max(axis=0).max(axis=0)  # the max dist along x/y axis
                dist = [dist.min(axis=0)] * len(xy_list)
            image_info_dict[image_id]['ann'] = [image_info_dict[image_id]['ann'][i] + [dist[i]] for i in range(len(dist)) if dist[i]>20 and dist[i]<1000]

        self.patch_info_list = []
        for image_id in image_id_list:
            for part_id, x, y, ann_id, dist in image_info_dict[image_id]['ann']:
                if part_id not in keypoint_cls_list:  # only use the target keypoints, part_id=idx+1
                    continue
                self.patch_info_list.append({
                    'image_path': os.path.join(self.img_dir, image_info_dict[image_id]['file_name']),
                    'xyd': (x, y, dist),
                    'part_id': part_id,
                    'ann_id': ann_id,
                })

        self.input_size = input_size
        if training:
            self.transforms = transforms.Compose([
                transforms.Resize(256, interpolation=3),
                transforms.CenterCrop(input_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ])
        else:
            self.transforms = transforms.Compose([
                transforms.Resize(input_size, interpolation=3),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std)])
        self.patch_root = patch_root
        self.part_name_map = {
            'right_eye': 'eye',
        }
        self.part_name_list = ["nose", "left_eye", "right_eye", "left_ear", "right_ear", "left_shoulder",
                               "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist",
                               "left_hip", "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle"]

        self.partid2clsid = {part_id: i for i, part_id in enumerate(keypoint_cls_list)}

    def __len__(self):
        return len(self.patch_info_list)

    def save(self, img, path):
        if not os.path.exists(path):
            img.save(path)

    def __getitem__(self, idx):
        patch_info = self.patch_info_list[idx]
        image_path = patch_info['image_path']
        x, y, dist = patch_info['xyd']
        ann_id = patch_info['ann_id']
        part_name = self.part_name_list[patch_info['part_id']-1]
        dist = round(dist / 2)
        box = (x-dist, y-dist, x+dist, y+dist)
        img = default_loader(image_path)
        patch = img.crop(box)

        image_name = '/'.join(image_path.split('/')[-2:])[:-4]
        patch_root = os.path.join(self.patch_root, image_name)
        os.makedirs(patch_root, exist_ok=True)
        patch_path = os.path.join(patch_root, part_name.replace(' ', '_') + f'_{ann_id}.jpg')
        
        """
        save the part image if needed for part classification
        """
        # self.save(img, os.path.join(patch_root, 'origin.jpg'))
        # self.save(patch, patch_path)

        patch = self.transforms(patch)
        return patch, patch_path, self.partid2clsid[patch_info['part_id']]

