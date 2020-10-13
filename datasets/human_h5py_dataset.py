import os
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import numpy as np
import random
import h5py
from PIL import Image
from utils.data_reading_util import get_classification_weights, _read_h5py, _read_json, _read_labels, is_consecutive


class HumanH5pyDataset(data.Dataset):
    CLASS_WEIGHTS = None

    def __init__(self, args, train=True):

        root_dir = args.data
        annotation_dir = os.path.join(root_dir, 'annotation_h5')
        if train:
            annotation_dir = os.path.join(annotation_dir, 'train_')
        elif args.use_test_for_val:
            annotation_dir = os.path.join(annotation_dir, 'test_')
        else:
            annotation_dir = os.path.join(annotation_dir, 'val_')

        self.annotation_dir = annotation_dir

        if args.input_feature_type == ['classes']:
            self.classification_weights = get_classification_weights(os.path.join(root_dir, 'cluster_weights.json'))
        else:
            self.classification_weights = None
        
        assert not 'diff_quaternion' in args.input_feature_type, 'use diff_quaternion_fixed instead'


        self.num_classes = args.num_classes
        self.sequence_length = args.sequence_length
        self.input_length = args.input_length
        self.output_length = args.output_length
        self.imus = args.imus
        self.imu_names = args.imu_names
        self.input_feature_type = args.input_feature_type
        self.h5pyind_2_frameind = _read_json(annotation_dir + 'h5pyind_2_frameind.json')
        self.h5pyind_2_image_name = _read_json(annotation_dir + 'image_name.json')
        self.idx_to_fid = [k for k in self.h5pyind_2_frameind.keys()]
        self.idx_to_fid.sort()
        self.title = args.title
        self.skipping_frames = args.skipping_frames

        self.root_dir = root_dir
        self.features_dir = args.features_dir
        self.save_feature_mode = args.mode == 'save_feats'
        
        self.read_features = args.read_features
        self.features_save_dir = args.features_save_dir
        self.manual_size = None

        self.flip_sequence_prob = 0
        self.crop_sequence_prob = 0

        if len([feat for feat in self.input_feature_type if ('angle_axis' in feat or 'quaternion' in feat)]) > 0:
            print('Are you taking care of zero cases?', 'imus = imus > 0')

        if args.manual_data_size:
            self.manual_size = args.manual_data_size
        self.number_of_trained_resnet_blocks = args.number_of_trained_resnet_blocks
        pairs = {
            'llegu': 'rlegu',
            'llegd': 'rlegd',
            'larmu': 'rarmu',
            'larmd': 'rarmd',
        }
        pairs.update({b: a for (a, b) in pairs.items()})
        self.imu_flip_rules = [i for i in range(len(self.imu_names))]
        self.image_flip_rules = [i for i in range(224-1, -1, -1)]
        for i, imu_name in enumerate(self.imu_names):
            if imu_name in pairs:
                corresponding = pairs[imu_name]
                assert corresponding in self.imu_names, 'it has to exist otherwise it makes no sense to flip'
                index_corresponding = self.imu_names.index(corresponding)
                self.imu_flip_rules[i] = index_corresponding
        # assert feature in ['diff_quaternion', 'quaternion', 'acceleration', 'classes', 'pose_cluster_classes', 'pca_classes', 'sequence_classes', 'acc_unit_direction_classes', 'acc_mag_classes', 'angle_axis_mag_classes', 'angle_axis_direction_classes']

        self.all_possible_sequences = self.get_possible_sequences()
        
        if self.manual_size:
            perm = torch.randperm(len(self.all_possible_sequences))
            idx = perm[:self.manual_size]
            new_list = []
            for i in idx:
                new_list.append(self.all_possible_sequences[i])
            self.all_possible_sequences = new_list

        if not train:
            torch.manual_seed(100)
            torch.cuda.manual_seed_all(100)
            perm = torch.randperm(len(self.all_possible_sequences))
            self.all_possible_sequences = [self.all_possible_sequences[p] for p in perm]
        
        print('Train=', train, 'Size of dataset', len(self), 'File is', self.annotation_dir)
        self.all_features_h5py_dict = None
        # self.crop_sequence_ratio = 0.05, 0.5  are these values good?

        if (args.mode == 'train' and train):  # Only for train and during training we have data transform
            self.flip_sequence_prob = 0.3
            # self.crop_sequence_prob = 0

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

        assert self.flip_sequence_prob == 0 or (train and args.mode == 'train' and set(self.input_feature_type) <= set(['gaze_points', 'move_label'])), 'Flip sequence not supported for other features, or in test mode'


    
    def initialize_dataset(self):
        self.all_features_h5py_dict = {}
        for feature in self.input_feature_type:
            self.all_features_h5py_dict[feature] = _read_h5py(self.annotation_dir + feature + '.h5')
        
        
    def get_possible_sequences(self):
        possibilities = []
        total_lengths = {clip_name:len(self.h5pyind_2_frameind[clip_name]) for clip_name in self.idx_to_fid}
        total_size = sum([v for k,v in total_lengths.items()])
        if self.manual_size:
            ratio = self.manual_size / total_size
            total_lengths = {k: v * ratio for k, v in total_lengths.items()}


        all_removed_gaze = 0
        if 'gaze_points' in self.input_feature_type:
            with h5py.File(self.annotation_dir + 'gaze_points' + '.h5', 'r') as file:
                all_gazes = {key: file[key][:] for key in file.keys()}

        
        print('Calculating the possible sequences')
        for clip_ind in range(len(self.idx_to_fid)):
            clip_name = self.idx_to_fid[clip_ind]

            frame_numbers = self.h5pyind_2_frameind[clip_name]
            
            this_clip_poss = []
            for j in range(len(frame_numbers) - self.sequence_length * self.skipping_frames):
                if is_consecutive(frame_numbers[j:j + self.sequence_length * self.skipping_frames]):
                    list_of_indices = [i * self.skipping_frames for i in range(self.sequence_length)]

                    # not sure if we should keep this here
                    if 'gaze_points' in self.input_feature_type and 'just_one_images' not in self.title:
                        gaze_points = all_gazes[clip_name][j:j + self.sequence_length * self.skipping_frames][list_of_indices]
                        if not (True in np.all(gaze_points != -1, axis=-1)):
                            all_removed_gaze += 1
                            continue

                    this_clip_poss.append((clip_ind, j, list_of_indices))
                if len(this_clip_poss) >= (total_lengths[clip_name]):
                    break

            possibilities += this_clip_poss
        print('Done with calculating the possible sequences')
        print('Removed for GAZE', all_removed_gaze)
        return possibilities

    def get_relative_centroids(self):
        return None

    def __len__(self):
        return len(self.all_possible_sequences) 

    def load_and_resize(self, img_name, grayscale=False):
        if img_name[-4:] == '.png':
            img_name = img_name[:-4] + '.jpg'
        with open(img_name, 'rb') as fp:
            image = Image.open(fp).convert('RGB')
            if grayscale:
                image = transforms.RandomGrayscale(p=1)(image)
        return self.transform(image)

    def flip_the_sequence(self, input, labels): # this can make a lot of problem especially if we inherit this class and forget that we are sometimes flipping.
        for key, val in input.items():
            if key == 'rgb':
                val = val[:, :, :, self.image_flip_rules]
                input[key] = val
            elif key == 'image_names':
                pass
            else:
                raise Exception('Not implemented')
        for key, val in labels.items():
            if key == 'move_label':
                val = val[:, self.imu_flip_rules]
                labels[key] = val
            elif key == 'gaze_points':
                mask = val == -1
                val[:,1] = 1 - val[:, 1]
                val[mask] = -1
                labels[key] = val

            else:
                raise Exception('Not implemented')
        return input, labels


    def __getitem__(self, idx):
        
        if not self.all_features_h5py_dict:
            self.initialize_dataset() #to avoid workers reading from the same h5py file

        (file_index, start_index, list_of_sub_indices) = self.all_possible_sequences[idx] 
        
        fid = self.idx_to_fid[file_index]

        features = []
        current_image_names = self.h5pyind_2_image_name[fid][start_index:start_index + self.sequence_length * self.skipping_frames]
        current_image_names = [x for i, x in enumerate(current_image_names) if i in list_of_sub_indices]


        for i in range(self.sequence_length):
            image_name = current_image_names[i]
            feature_path = os.path.join(self.features_dir, image_name)
            feature = self.load_and_resize(feature_path)

            features.append(feature)


        input = {
            'rgb': torch.stack(features, 0),
            'image_names': current_image_names,
        }
        labels={feature_name:torch.Tensor(self.all_features_h5py_dict[feature_name][fid][start_index:start_index + self.sequence_length * self.skipping_frames][:,self.imus]) for feature_name in self.input_feature_type if not 'gaze' in feature_name}
        gaze_labels = {feature_name:torch.Tensor(self.all_features_h5py_dict[feature_name][fid][start_index:start_index + self.sequence_length * self.skipping_frames][:]) for feature_name in self.input_feature_type if 'gaze' in feature_name} #seq_len x 2
        labels.update(gaze_labels)


        labels = {feature_name: labels[feature_name][list_of_sub_indices] for feature_name in labels}


        for ind, feature in enumerate(self.input_feature_type):
            
            if feature in ['classes', 'pose_cluster_classes', 'pca_classes', 'sequence_classes', 'acc_unit_direction_classes', 'acc_mag_classes', 'angle_axis_mag_classes', 'angle_axis_direction_classes', 'angle_axis_classes']:
                labels[feature] = labels[feature].long()

            if feature == 'gaze_points':
                gaze_labels = labels[feature]
                mask = gaze_labels == -1
                gaze_labels = gaze_labels / torch.Tensor([299.,224.])
                gaze_labels[mask] = -1

                gaze_labels = gaze_labels[:, [1,0]]

                labels[feature] = gaze_labels



            if feature == 'angle_axis_mag_semiclass':
                raise Exception('Not implemented')

        if random.random() <= self.flip_sequence_prob:
            input, labels = self.flip_the_sequence(input, labels)

        return (input, labels)























# def crop_the_sequence(self, input, labels):
#         print('Gaze still not working for some reason')
#         pdb.set_trace()
#         left_crop_ratios = [random.random() * x for x in self.crop_sequence_ratio]
#         left_crop_number_pixels = [max(1, round(x * 224)) for x in left_crop_ratios]
#         right_crop_ratios = [random.random() * x for x in self.crop_sequence_ratio]
#         right_crop_number_pixels = [max(1, round(x * 224)) for x in right_crop_ratios]
#         for key, val in input.items():
#             if key == 'rgb':
#                 cropped_image = val[:, :, left_crop_number_pixels[0]:-right_crop_number_pixels[0], left_crop_number_pixels[1]:-right_crop_number_pixels[1]]
#                 cropped_image = F.interpolate(cropped_image, (224, 224))
#                 input[key] = cropped_image
#             elif key == 'image_names':
#                 pass
#             else:
#                 raise Exception('Not implemented')
#         for key, val in labels.items():
#             if key == 'move_label':
#                 val = val[:, self.imu_flip_rules]
#                 labels[key] = val
#             elif key == 'gaze_points':
#                 mask = val == -1
#
#                 print('TODO make sure min and max is 0 and 1')
#                 pdb.set_trace()
#                 val[:,0] = (val[:,0] - left_crop_ratios[0]) / 1 - (left_crop_ratios[0] + right_crop_ratios[0])
#                 val[:,1] = (val[:,1] - left_crop_ratios[1]) / 1 - (left_crop_ratios[1] + right_crop_ratios[1])
#                 val[mask] = -1
#                 labels[key] = val
#             else:
#                 raise Exception('Not implemented')
#         return input, labels
