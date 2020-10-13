import os
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image


def parse_file(dataset_adr, categories):
    dataset = []
    with open(dataset_adr) as f:
        for line in f:
            line = line[:-1].split('/')
            category = '/'.join(line[2:-1])
            file_name = '/'.join(line[2:])
            if not category in categories:
                continue
            dataset.append([file_name, category])
    return dataset


def get_class_names(path):
    classes = []
    with open(path) as f:
        for line in f:
            categ = '/'.join(line[:-1].split('/')[2:])
            classes.append(categ)
    class_dic = {classes[i]: i for i in range(len(classes))}
    return class_dic


class SunDataset(data.Dataset):
    SUN_SCENE_INDEX_TO_NAME = None

    CLASS_WEIGHTS = None

    def __init__(self, args, train=True):
        self.root_dir = args.data
        root_dir = self.root_dir
        if train:
            self.data_set_list = os.path.join(root_dir, 'train_test', 'Training_01.txt')
        else:
            self.data_set_list = os.path.join(root_dir, 'train_test', 'Testing_01.txt')

        self.categ_dict = get_class_names(
            os.path.join(root_dir, 'ClassName.txt'))

        SunDataset.SUN_SCENE_INDEX_TO_NAME = {scene_name:class_index for (class_index, scene_name) in self.categ_dict.items()}

        self.data_set_list = parse_file(self.data_set_list, self.categ_dict)

        self.args = args
        self.read_features = args.read_features

        self.classification_weights = None

        self.features_dir = args.features_dir
        if train:
            self.transform = transforms.Compose([
                transforms.RandomSizedCrop(args.image_size),
                transforms.RandomHorizontalFlip(),
                transforms.Scale((args.image_size, args.image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Scale((args.image_size, args.image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
            ])

        assert args.sequence_length == args.output_length == args.input_length == 1
        assert args.num_classes == 397

        if not train:
            torch.manual_seed(100)
            torch.cuda.manual_seed_all(100)
            perm = torch.randperm(len(self.data_set_list))
            self.data_set_list = [self.data_set_list[p] for p in perm]



    def get_relative_centroids(self):
        return None

    def __len__(self):
        return len(self.data_set_list)

    def load_and_resize(self, img_name):
        with open(img_name, 'rb') as fp:
            image = Image.open(fp).convert('RGB')
        return self.transform(image)

    def __getitem__(self, idx):
        file_name, categ = self.data_set_list[idx]
        try:
            image = self.load_and_resize(
                os.path.join(self.root_dir, 'all_data', file_name + '~'))
        except Exception:
            image = self.load_and_resize(
                os.path.join(self.root_dir, 'all_data', file_name))
        if not categ in self.categ_dict:
            raise Exception('category not found')
        label = self.categ_dict[categ]
        label = torch.Tensor([label]).long().squeeze()
        label = {
            'scene_class': label,
        }
        input = {
            'rgb': image.unsqueeze(0), #add sequence len
        }

        return input, label
