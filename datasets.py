import json
from pathlib import Path
import os

import numpy as np
from PIL import Image, ImageDraw
import torch
from torch.utils import data
from torchvision import transforms


class VITONDataset(data.Dataset):
    def __init__(self, opt):
        super(VITONDataset, self).__init__()
        self.load_height = opt.load_height
        self.load_width = opt.load_width
        self.semantic_nc = opt.semantic_nc
        self.data_path = Path(opt.dataset_dir) / opt.dataset_mode
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        # Load data list
        self.img_names, self.c_names = self.load_data_list(opt.dataset_dir, opt.dataset_list)
        self.num_samples = len(self.img_names)

        # Check if num_samples is valid
        if not isinstance(self.num_samples, int) or self.num_samples <= 0:
            raise ValueError("num_samples must be a positive integer.")

    def load_data_list(self, dataset_dir, dataset_list):
        img_names = []
        c_names = []
        try:
            with open(Path(dataset_dir) / dataset_list, 'r') as f:
                for line in f:
                    img_name, c_name = line.strip().split()
                    img_names.append(img_name)
                    c_names.append(c_name)
        except FileNotFoundError:
            raise FileNotFoundError(f"Dataset list file '{dataset_list}' not found in '{dataset_dir}'")
        return img_names, c_names

    def __len__(self):
        return self.num_samples

    def get_parse_agnostic(self, parse, pose_data):
        parse_array = np.array(parse)
        parse_upper = ((parse_array == 5).astype(np.float32) +
                       (parse_array == 6).astype(np.float32) +
                       (parse_array == 7).astype(np.float32))
        parse_neck = (parse_array == 10).astype(np.float32)

        r = 10
        agnostic = parse.copy()

        # mask arms
        for parse_id, pose_ids in [(14, [2, 5, 6, 7]), (15, [5, 2, 3, 4])]:
            mask_arm = Image.new('L', (self.load_width, self.load_height), 'black')
            mask_arm_draw = ImageDraw.Draw(mask_arm)
            i_prev = pose_ids[0]
            for i in pose_ids[1:]:
                if (pose_data[i_prev, 0] == 0.0 and pose_data[i_prev, 1] == 0.0) or (pose_data[i, 0] == 0.0 and pose_data[i, 1] == 0.0):
                    continue
                mask_arm_draw.line([tuple(pose_data[j]) for j in [i_prev, i]], 'white', width=r*10)
                pointx, pointy = pose_data[i]
                radius = r*4 if i == pose_ids[-1] else r*15
                mask_arm_draw.ellipse((pointx-radius, pointy-radius, pointx+radius, pointy+radius), 'white', 'white')
                i_prev = i
            parse_arm = (np.array(mask_arm) / 255) * (parse_array == parse_id).astype(np.float32)
            agnostic.paste(0, None, Image.fromarray(np.uint8(parse_arm * 255), 'L'))

        # mask torso & neck
        agnostic.paste(0, None, Image.fromarray(np.uint8(parse_upper * 255), 'L'))
        agnostic.paste(0, None, Image.fromarray(np.uint8(parse_neck * 255), 'L'))

        return agnostic

    def get_img_agnostic(self, img, parse, pose_data):
        parse_array = np.array(parse)
        parse_head = ((parse_array == 4).astype(np.float32) +
                      (parse_array == 13).astype(np.float32))
        parse_lower = ((parse_array == 9).astype(np.float32) +
                       (parse_array == 12).astype(np.float32) +
                       (parse_array == 16).astype(np.float32) +
                       (parse_array == 17).astype(np.float32) +
                       (parse_array == 18).astype(np.float32) +
                       (parse_array == 19).astype(np.float32))

        r = 20
        agnostic = img.copy()
        agnostic_draw = ImageDraw.Draw(agnostic)

        length_a = np.linalg.norm(pose_data[5] - pose_data[2])
        length_b = np.linalg.norm(pose_data[12] - pose_data[9])
        point = (pose_data[9] + pose_data[12]) / 2
        pose_data[9] = point + (pose_data[9] - point) / length_b * length_a
        pose_data[12] = point + (pose_data[12] - point) / length_b * length_a

        # mask arms
        agnostic_draw.line([tuple(pose_data[i]) for i in [2, 5]], 'gray', width=r*10)
        for i in [2, 5]:
            pointx, pointy = pose_data[i]
            agnostic_draw.ellipse((pointx-r*5, pointy-r*5, pointx+r*5, pointy+r*5), 'gray', 'gray')
        for i in [3, 4, 6, 7]:
            if (pose_data[i - 1, 0] == 0.0 and pose_data[i - 1, 1] == 0.0) or (pose_data[i, 0] == 0.0 and pose_data[i, 1] == 0.0):
                continue
            agnostic_draw.line([tuple(pose_data[j]) for j in [i - 1, i]], 'gray', width=r*10)
            pointx, pointy = pose_data[i]
            agnostic_draw.ellipse((pointx-r*5, pointy-r*5, pointx+r*5, pointy+r*5), 'gray', 'gray')

        # mask torso
        for i in [9, 12]:
            pointx, pointy = pose_data[i]
            agnostic_draw.ellipse((pointx-r*3, pointy-r*6, pointx+r*3, pointy+r*6), 'gray', 'gray')
        agnostic_draw.line([tuple(pose_data[i]) for i in [2, 9]], 'gray', width=r*6)
        agnostic_draw.line([tuple(pose_data[i]) for i in [5, 12]], 'gray', width=r*6)
        agnostic_draw.line([tuple(pose_data[i]) for i in [9, 12]], 'gray', width=r*12)
        agnostic_draw.polygon([tuple(pose_data[i]) for i in [2, 5, 12, 9]], 'gray', 'gray')

        # mask neck
        pointx, pointy = pose_data[1]
        agnostic_draw.rectangle((pointx-r*7, pointy-r*7, pointx+r*7, pointy+r*7), 'gray', 'gray')
        agnostic.paste(img, None, Image.fromarray(np.uint8(parse_head * 255), 'L'))
        agnostic.paste(img, None, Image.fromarray(np.uint8(parse_lower * 255), 'L'))

        return agnostic

    def __getitem__(self, index):
        img_name = self.img_names[index]
        c_name = self.c_names['unpaired'][index]

        # Load cloth image
        cloth_path = self.data_path / 'cloth' / c_name
        cloth_image = self.load_image(cloth_path)

        # Load cloth mask
        cloth_mask_path = self.data_path / 'cloth-mask' / c_name
        cloth_mask = self.load_image(cloth_mask_path, is_mask=True)

        # Load pose image
        pose_name = img_name.replace('.jpg', '_rendered.png')
        pose_path = self.data_path / 'openpose-img' / pose_name
        pose_rgb = self.load_image(pose_path)

        # Load pose keypoints
        pose_keypoints_path = img_name.replace('.jpg', '_keypoints.json')
        pose_data = self.load_pose_data(pose_keypoints_path)

        # Load parsing image
        parse_name = img_name.replace('.jpg', '.png')
        parse_path = self.data_path / 'image-parse' / parse_name
        parse = self.load_image(parse_path)

        # Process and return the result
        return self.process_data(img_name, cloth_image, cloth_mask, pose_rgb, pose_data, parse)

    def load_image(self, path, is_mask=False):
        try:
            image = Image.open(path).convert('RGB')
            if is_mask:
                image = image.resize((self.load_width, self.load_height), Image.NEAREST)
            else:
                image = image.resize((self.load_width, self.load_height), Image.BILINEAR)
            return self.transform(image)
        except Exception as e:
            print(f"Error loading image {path}: {e}")
            return None

    def load_pose_data(self, pose_keypoints_path):
        try:
            with open(pose_keypoints_path, 'r') as f:
                pose_label = json.load(f)
                pose_data = pose_label['people'][0]['pose_keypoints_2d']
                return np.array(pose_data).reshape((-1, 3))[:, :2]
        except Exception as e:
            print(f"Error loading pose keypoints from {pose_keypoints_path}: {e}")
            return None

    def process_data(self, img_name, cloth_image, cloth_mask, pose_rgb, pose_data, parse):
        parse_agnostic = self.get_parse_agnostic(parse, pose_data)
        parse_agnostic = torch.from_numpy(np.array(parse_agnostic)[None]).long()

        labels = {
            0: ['background', [0, 10]],
            1: ['hair', [1, 2]],
            2: ['face', [4, 13]],
            3: ['upper', [5, 6, 7]],
            4: ['bottom', [9, 12]],
            5: ['left_arm', [14]],
            6: ['right_arm', [15]],
            7: ['left_leg', [16]],
            8: ['right_leg', [17]],
            9: ['left_shoe', [18]],
            10: ['right_shoe', [19]],
            11: ['socks', [8]],
            12: ['noise', [3, 11]]
        }
        parse_agnostic_map = torch.zeros(20, self.load_height, self.load_width, dtype=torch.float)
        parse_agnostic_map.scatter_(0, parse_agnostic, 1.0)
        new_parse_agnostic_map = torch.zeros(self.semantic_nc, self.load_height, self.load_width, dtype=torch.float)
        for i in range(len(labels)):
            for label in labels[i][1]:
                new_parse_agnostic_map[i] += parse_agnostic_map[label]

        # load person image
        img = Image.open(self.data_path / 'image' / img_name)
        img = img.resize((self.load_width, self.load_height), Image.BILINEAR)
        img_agnostic = self.get_img_agnostic(img, parse, pose_data)
        img = self.transform(img)
        img_agnostic = self.transform(img_agnostic)  # [-1,1]

        result = {
            'img_name': img_name,
            'c_name': c_name,
            'img': img,
            'img_agnostic': img_agnostic,
            'parse_agnostic': new_parse_agnostic_map,
            'pose': pose_rgb,
            'cloth': cloth_image,
            'cloth_mask': cloth_mask,
        }
        return result


class VITONDataLoader:
    def __init__(self, opt, dataset):
        super(VITONDataLoader, self).__init__()

        if opt.shuffle:
            train_sampler = data.sampler.RandomSampler(dataset)
        else:
            train_sampler = None

        self.data_loader = data.DataLoader(
                dataset, batch_size=opt.batch_size, shuffle=(train_sampler is None),
                num_workers=opt.workers, pin_memory=True, drop_last=True, sampler=train_sampler
        )
        self.dataset = dataset
        self.data_iter = self.data_loader.__iter__()

    def next_batch(self):
        try:
            batch = self.data_iter.__next__()
            return batch
        except StopIteration:
            # Reset the iterator and return None if there are no more batches
            self.data_iter = self.data_loader.__iter__()
            return None  # Return None when there are no more batches


class Config():
    load_height = 256
    load_width = 256
    semantic_nc = 20
    dataset_dir = 'C:\\Users\\swayam\\OneDrive\\Desktop\\Projects\\miniProject2'  # Base directory
    dataset_mode = 'train'
    dataset_list = 'dataset_list.txt'  # Ensure this file exists in the dataset_dir
    shuffle = True  # Set to True or False based on your requirement
    batch_size = 16  # Set your desired batch size here
    workers = 4  # Set the number of worker processes for data loading


if __name__ == '__main__':
    opt = Config()

    from datasets import VITONDataset, VITONDataLoader

    # Initialize the dataset
    dataset = VITONDataset(opt)

    # Initialize the data loader
    data_loader = VITONDataLoader(opt, dataset)

    # Example of getting a batch
    batch = data_loader.next_batch()
    if batch is not None:
        print("Batch loaded successfully:")
        print(batch)
    else:
        print("No more batches available.")

    # Correct path to your cloth images
    cloth_directory = 'C:\\Users\\swayam\\OneDrive\\Desktop\\Projects\\miniProject2\\cloth\\'

    try:
        for path in os.listdir(cloth_directory):
            print(path)  # or process the files as needed
    except FileNotFoundError as e:
        print(f"Error: {e}")
