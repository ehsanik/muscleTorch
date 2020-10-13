import json
import os
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision
from scipy.io import loadmat
from PIL import Image

SCENARIO_VIEW_INDEX = [(0, 1), (0, 4), (0, 0), (0, 3), (0, 6), (0, 7), (0, 2), (0, 5), (9, 4), (9, 6), (9, 5), (9, 2), (9, 7), (9, 0), (9, 3), (9, 1), (10, 0), (10, 1), (10, 2), (11, 2), (11, 0), (11, 1), (11, 3), (1, 1), (1, 0), (1, 3), (1, 2), (2, 2), (2, 1), (2, 7), (2, 4), (2, 5), (2, 6), (2, 3), (2, 0), (3, 5), (3, 7), (3, 0), (3, 6), (3, 4), (3, 3), (3, 1), (3, 2), (4, 0), (5, 0), (5, 1), (5, 2), (6, 0), (6, 1), (6, 2), (7, 2), (7, 0), (7, 7), (7, 3), (7, 1), (7, 6), (7, 4), (7, 5), (8, 1), (8, 4), (8, 0), (8, 6), (8, 5), (8, 7), (8, 2), (8, 3)]

class VindDataset(data.Dataset):
    CLASS_WEIGHTS = None

    def __init__(self, args, train=True):

        self.num_classes = args.num_classes

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
        self.mask_transform = transforms.Compose([
            transforms.Scale((args.image_size, args.image_size)),
            transforms.ToTensor(),
        ])

        self.root_dir = args.data
        if train:
            image_folder = os.path.join(self.root_dir, 'train', 'images')
        else:
            image_folder = os.path.join(self.root_dir, 'test', 'images')

        dataset = torchvision.datasets.ImageFolder(image_folder, self.transform)
        imagenames = [img_name for (img_name, _) in dataset.imgs]
        imagenames = self.prune_invalid_images(imagenames)
        self.data_set_list = imagenames
        if args.manual_data_size:
            self.data_set_list = self.data_set_list[:args.manual_data_size]

        self.classification_weights = None

        if not train:
            torch.manual_seed(100)
            torch.cuda.manual_seed_all(100)
            perm = torch.randperm(len(self.data_set_list))
            self.data_set_list = [self.data_set_list[p] for p in perm]
        assert args.sequence_length == args.input_length == args.output_length == 1

    def get_class_from_scenario_ge(self, scenario, ge):
        return SCENARIO_VIEW_INDEX.index((scenario, ge))

    def prune_invalid_images(self, image_list):
        valid_images = []
        print('total is', len(image_list))
        for img in image_list:
            mask_name = img.replace('/images/', '/objmask/')
            labels_dir = img.replace('/images/', '/labels/')
            mat_name = labels_dir.replace('.png', '_00_ge.mat')
            if not os.path.exists(mask_name) or not os.path.exists(mat_name):
                continue
            valid_images.append(img)
        print('after pruning', len(valid_images))
        return valid_images

    def __len__(self):
        return len(self.data_set_list)

    def load_and_resize(self, img_name):
        with open(img_name, 'rb') as fp:
            image = Image.open(fp).convert('RGB')
        return self.transform(image)

    def mask_load_and_resize(self, img_name):
        with open(img_name, 'rb') as fp:
            image = Image.open(fp).convert('RGB')
        return self.mask_transform(image)[0:1]

    def __getitem__(self, idx):
        assert self.num_classes == 66
        image_name = self.data_set_list[idx]
        mask_name = image_name.replace('/images/', '/objmask/')
        image = self.load_and_resize(image_name)

        labels_dir = image_name.replace('/images/', '/labels/')

        ge = loadmat(labels_dir.replace('.png', '_00_ge.mat'))['ge']
        viewpoint = ge.item() - 1
        scenario = image_name.split('/images/')[1].split('-')[0].replace('scenario', '')
        scenario = int(scenario) - 1
        vind_class = torch.tensor(self.get_class_from_scenario_ge(scenario, viewpoint))


        objmask = self.mask_load_and_resize(mask_name)
        label = {
            'vind_class': vind_class,
            'objmask': objmask.unsqueeze(0), #seqlen x c x w x h
        }
        input = {
            'rgb': image.unsqueeze(0),  #seqlen x c x w x h
            'image_names': image_name
        }

        return (input, label)

