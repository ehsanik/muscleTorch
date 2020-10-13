import os
import torchvision.transforms as transforms
from PIL import Image
from .human_h5py_dataset import HumanH5pyDataset

class HumanContrastiveCombinedDataset(HumanH5pyDataset):
    CLASS_WEIGHTS = None

    def __init__(self, args, train=True):
        super(HumanContrastiveCombinedDataset, self).__init__(args, train)
        self.moco_transform = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.2, 1), ratio=(0.7, 1.4), interpolation=Image.BILINEAR),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.4)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomHorizontalFlip(),
            transforms.Scale((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])


    def load_and_resize_moco(self, img_name):
        if img_name[-4:] == '.png':
            img_name = img_name[:-4] + '.jpg'
        with open(img_name, 'rb') as fp:
            image = Image.open(fp).convert('RGB')
        return self.moco_transform(image)

    def __getitem__(self, idx):
        input, labels = super(HumanContrastiveCombinedDataset, self).__getitem__(idx)
        first_frame_adr = input['image_names'][0]
        feature_path = os.path.join(self.features_dir, first_frame_adr)
        feature_first = self.load_and_resize_moco(feature_path)
        feature_second = self.load_and_resize_moco(feature_path)

        input['first_augm'] = feature_first.unsqueeze(0)
        input['second_augm'] = feature_second.unsqueeze(0)

        return (input, labels)
