import os
import torch
import torchvision.transforms as transforms
import torchvision


def _read_file(file_name):
    result_dict = {}
    with open(file_name) as f:
        all_lines = [l for l in f]

    for line in all_lines:
        line = line.replace('\n', '')
        synset_id = line.split(' ')[0]
        synset_name = line.replace(synset_id, '').split(',')[0]
        result_dict[synset_id] = synset_name[1:] #remove space
    return result_dict

class ImagenetDataset(torchvision.datasets.ImageFolder):

    CLASS_WEIGHTS = None

    def __init__(self, args, train=True):

        self.root_dir = args.data
        self.classification_weights = None
        if train:
            imagenet_dir = os.path.join(self.root_dir, 'train')
        else:
            imagenet_dir = os.path.join(self.root_dir, 'val')


        if train:
            self.transform = transforms.Compose([
                transforms.RandomSizedCrop(args.image_size),
                transforms.RandomHorizontalFlip(),
                transforms.Scale((args.image_size, args.image_size)),
                transforms.ColorJitter(.4, .4, .4, .2),
                transforms.RandomGrayscale(p=0.2),
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

        super(ImagenetDataset, self).__init__(imagenet_dir, self.transform)

        synset_to_real_name = _read_file('utils/imagenet_synsets.txt')

        ImagenetDataset.IMAGENET_CLASS_TO_NAME = [synset_to_real_name[x] for x in self.classes]


        self.manual_size = args.manual_data_size
        if self.manual_size:
            self.samples = self.samples[:self.manual_size]

        if not train:
            torch.manual_seed(100)
            torch.cuda.manual_seed_all(100)
            perm = torch.randperm(len(self.samples))
            self.samples = [self.samples[p] for p in perm]


    def get_relative_centroids(self):
        return None

    def __getitem__(self, idx):
        data_item = super(ImagenetDataset, self).__getitem__(idx)
        image = data_item[0]
        label = data_item[1]
        image_name = self.samples[idx][0]

        label = torch.Tensor([label]).long().squeeze()
        label = {
            'class_label': label,
        }
        input = {
            'rgb': image.unsqueeze(0), #add sequence len
            'image_names': str(idx),
        }

        return input, label
