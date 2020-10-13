import os
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import numpy as np

from PIL import Image

from utils.nyu_data_utils import train_set_list, test_set_list, val_set_list


class NYUDepthDataset(data.Dataset):
    CLASS_WEIGHTS = None

    def __init__(self, args, train=True):

        if (args.mode == 'train' and train):
            self.transform = transforms.Compose([
                transforms.Scale((224, 224)),
                transforms.ColorJitter(.4, .4, .4, .2),
                transforms.RandomGrayscale(p=0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

        else:
            self.transform = transforms.Compose([
                transforms.Scale((args.image_size, args.image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
            ])

        self.depth_transform = transforms.Compose([
            transforms.Scale((args.reconst_resolution, args.reconst_resolution)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                 std=[0.5, 0.5, 0.5]),
        ])
        self.root_dir = args.data
        self.img_dir = os.path.join(self.root_dir, 'nyu_images')
        self.depth_dir = os.path.join(self.root_dir, 'nyu_depths')

        if train:
            dataset_list = train_set_list
        elif args.use_test_for_val:
            dataset_list = test_set_list
        else:
            dataset_list = val_set_list

        self.data_set_list = ['%06d.png' % (x) for x in dataset_list]

        if args.manual_data_size:
            self.data_set_list = self.data_set_list[:args.manual_data_size]

        self.classification_weights = None

        if not train:
            torch.manual_seed(100)
            torch.cuda.manual_seed_all(100)
            perm = torch.randperm(len(self.data_set_list))
            self.data_set_list = [self.data_set_list[p] for p in perm]

        assert args.sequence_length == args.input_length == args.output_length == 1


    def __len__(self):
        return len(self.data_set_list)

    def load_and_resize(self, img_name):
        with open(img_name, 'rb') as fp:
            image = Image.open(fp).convert('RGB')
        return self.transform(image)

    def depth_load_and_resize(self, img_name):
        with open(img_name, 'rb') as fp:
            image = Image.open(fp).convert('RGB')
        return self.depth_transform(image)[0:1]

    def __getitem__(self, idx):

        fid = self.data_set_list[idx]

        image = self.load_and_resize(os.path.join(self.img_dir, fid))
        depth = self.depth_load_and_resize(os.path.join(self.depth_dir, fid))
        label = {
            'depth': depth.unsqueeze(0), #seqlen x c x w x h
        }
        input = {
            'rgb': image.unsqueeze(0),  #seqlen x c x w x h
        }

        return (input, label)

