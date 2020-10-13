import os
import torch
import torch.utils.data as data
import torchvision.transforms as transforms

from PIL import Image

from utils.nyu_data_utils import train_set_list, test_set_list, val_set_list, UNAVAILBLE_WALK_TRAIN_SET, UNAVAILBLE_WALK_TEST_SET


def _category_weights():

    category_sizes = torch.Tensor([1.0 - 0.09, 0.09])
    weight = 1.0 / category_sizes
    return weight


class NYUWalkDataset(data.Dataset):
    CLASS_WEIGHTS = _category_weights() # do we need that?

    def __init__(self, args, train=True):
        self.root_dir = args.data

        if train:
            self.data_set_list = train_set_list
        elif args.use_test_for_val:
            self.data_set_list = test_set_list
        else:
            self.data_set_list = val_set_list

        self.data_set_list = ['%06d.png' % (x) for x in self.data_set_list]
        self.args = args
        self.read_features = args.read_features
        self.classification_weights = torch.Tensor([1 - 0.1070, 0.1070])
        # self.classification_weights = WALKABLE_WEIGHT

        self.features_dir = args.features_dir
        if train and args.mode =='train':

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
        self.transform_segmentation = transforms.Compose([
            transforms.Scale((args.image_size, args.image_size)),
            transforms.ToTensor(),
        ])

        if train:
            remove_list = UNAVAILBLE_WALK_TRAIN_SET
        else:
            remove_list = UNAVAILBLE_WALK_TEST_SET

        result_list = []
        for x in self.data_set_list:
            if not x in remove_list:
                result_list.append(x)
        self.data_set_list = result_list

        if not train:
            torch.manual_seed(100)
            torch.cuda.manual_seed_all(100)
            perm = torch.randperm(len(self.data_set_list))
            self.data_set_list = [self.data_set_list[p] for p in perm]

        assert args.sequence_length == 1



    def get_relative_centroids(self):
        return None

    def clean_mask(self, output):
        cleaned = output.clone()
        cleaned[output > 0.5] = 1.0
        cleaned[output < 0.5] = 0.0
        return cleaned

    def __len__(self):
        return len(self.data_set_list)

    def load_and_resize(self, img_name):
        with open(img_name, 'rb') as fp:
            image = Image.open(fp).convert('RGB')
        return self.transform(image)

    def load_and_resize_segmentation(self, img_name):
        with open(img_name, 'rb') as fp:
            image = Image.open(fp).convert('L')
        return self.clean_mask(self.transform_segmentation(image))

    def __getitem__(self, idx):
        fid = self.data_set_list[idx]

        image = self.load_and_resize(
            os.path.join(self.root_dir, 'images', fid))
        segment = self.load_and_resize_segmentation(
            os.path.join(self.root_dir, 'walkable', fid))

        labels = {
            'walk':segment.long().unsqueeze(0),
        }
        input = {
            'rgb': image.unsqueeze(0),
            'image_names': fid,
        }

        # The two 0s are just place holders. They can be replaced by any values
        return input, labels
