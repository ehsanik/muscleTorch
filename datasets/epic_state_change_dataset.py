
import os
import torch
import random
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image
from utils.data_reading_util import _read_csv
from utils.epic_train_val_split import EPIC_VERB_OCCURENCE, EPIC_ORIGINAL_INDEX_TO_NAME, train_split, val_split

'''
based on this submission to EPIC challenged:
https://epic-kitchens.github.io/Reports/EPIC-Kitchens-Challenges-2019-Report.pdf
'''
STATE_CHANGING_VERBS = ['take', 'put', 'open', 'close', 'wash', 'cut', 'mix', 'pour', 'peel']

EPIC_SMALL_INDEX_TO_NAME = {i: verb for (i, verb) in enumerate(STATE_CHANGING_VERBS)}
EPIC_NAME_TO_SMALL_INDEX = {k: v for (v, k) in EPIC_SMALL_INDEX_TO_NAME.items()}

REMOVED_KEYS_EPIC = {k for (v, k) in EPIC_ORIGINAL_INDEX_TO_NAME.items() if (k not in EPIC_VERB_OCCURENCE) or (k not in STATE_CHANGING_VERBS)}

EPIC_WEIGHT_FREQ_INVERSE = torch.ones(len(STATE_CHANGING_VERBS))

class EpicStateChangingDataset(data.Dataset):
    CLASS_WEIGHTS = EPIC_WEIGHT_FREQ_INVERSE
    VERB_CLASS_TO_NAME = EPIC_SMALL_INDEX_TO_NAME
    VALID_NAMES = [verb_name for (class_index, verb_name) in VERB_CLASS_TO_NAME.items()]
    EPIC_ORIGINAL_CLASS_TO_NAME = EPIC_ORIGINAL_INDEX_TO_NAME

    def __init__(self, args, train=True):
        assert args.sequence_length <=6, 'Can not support more'

        self.root_dir = args.data
        self.skipping_frames = args.skipping_frames
        self.sequence_length = args.sequence_length
        self.num_classes = args.num_classes

        self.manual_size = None
        self.classification_weights = None
        if args.manual_data_size:
            self.manual_size = args.manual_data_size

        annotation_file = os.path.join(self.root_dir, 'EPIC_train_action_labels.csv')
        self.rgb_folder = 'images/train'
        if train:
            valid_files = train_split
        else:
            valid_files = val_split
        keys, all_action_labels = _read_csv(annotation_file)
        self.all_possible_sequences = self.get_possible_sequences(all_action_labels, valid_files)

        if not train:
            torch.manual_seed(100)
            torch.cuda.manual_seed_all(100)
            perm = torch.randperm(len(self.all_possible_sequences))
            self.all_possible_sequences = [self.all_possible_sequences[p] for p in perm]
        print('Train=', train, 'Size of dataset', len(self), 'File is', annotation_file)

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

        assert self.num_classes == len(self.CLASS_WEIGHTS)

        assert args.input_length == args.sequence_length and args.output_length == 1

        assert not args.manual_data_size

        assert self.num_classes == len(STATE_CHANGING_VERBS)
        if train:
            self.action_to_videos = self.get_balanced_data()
        else:
            self.action_to_videos = None

        if not train and args.mode == 'test':
            self.all_possible_sequences = self.get_complete_test_set(self.all_possible_sequences)


    def get_relative_centroids(self):
        return None

    def get_possible_sequences(self, all_action_labels, valid_files):
        result = []
        valid_files = set(valid_files)

        for element in all_action_labels:
            action_verb = int(element['verb_class'])
            action_verb = self.EPIC_ORIGINAL_CLASS_TO_NAME[action_verb]
            if action_verb not in self.VALID_NAMES:
                continue

            video_id = element['video_id']
            if video_id not in valid_files:
                continue


            result.append(element)

        return result

        # some videos overlap but they have different labels


    '''
    To get the full test data, rather than sampling
    '''
    def get_complete_test_set(self, videos):
        all_samples = []
        for selected_sequence in videos:
            # participant_id = selected_sequence['participant_id']
            # video_id = selected_sequence['video_id']
            # verb_class = int(selected_sequence['verb_class'])
            # verb_class = self.convert_class_index(verb_class) # convert verb_class
            # noun_class = int(selected_sequence['noun_class'])
            start_frame = int(selected_sequence['start_frame'])
            stop_frame = int(selected_sequence['stop_frame'])
            # random_start = random.randint(start_frame, stop_frame - self.skipping_frames * self.sequence_length)
            for random_start in range(start_frame, stop_frame - self.skipping_frames * self.sequence_length - 1):
                this_seq = selected_sequence.copy()
                this_seq['start_frame'] = random_start
                all_samples.append(this_seq) # this is not efficient. we could just keep another start index
        return all_samples

    def convert_class_index(self, verb_class):
        original_name = self.EPIC_ORIGINAL_CLASS_TO_NAME[verb_class]
        new_index = EPIC_NAME_TO_SMALL_INDEX[original_name]
        return new_index

    def get_balanced_data(self):

        action_to_videos_dict = {}
        for sequence in self.all_possible_sequences:
            verb_class = int(sequence['verb_class'])
            verb_class = self.convert_class_index(verb_class)
            action_to_videos_dict.setdefault(verb_class, [])
            action_to_videos_dict[verb_class].append(sequence)
        return action_to_videos_dict


    def __len__(self):
        return len(self.all_possible_sequences)

    def load_and_resize(self, img_name):
        with open(img_name, 'rb') as fp:
            image = Image.open(fp).convert('RGB')
        return self.transform(image)

    def __getitem__(self, idx):

        if self.action_to_videos:
            all_possible_keys = [k for k in self.action_to_videos.keys()]
            random_sequence_list = self.action_to_videos[random.choice(all_possible_keys)] #randomly pick a value from this dictionary
            selected_sequence = random.choice(random_sequence_list)
        else:
            selected_sequence = self.all_possible_sequences[idx]

        participant_id = selected_sequence['participant_id']
        video_id = selected_sequence['video_id']
        verb_class = int(selected_sequence['verb_class'])
        verb_class = self.convert_class_index(verb_class) # convert verb_class
        noun_class = int(selected_sequence['noun_class'])
        start_frame = int(selected_sequence['start_frame'])
        stop_frame = int(selected_sequence['stop_frame'])


        image_dir = os.path.join(self.root_dir, self.rgb_folder, participant_id, video_id)

        random_start = random.randint(start_frame, stop_frame - self.skipping_frames * self.sequence_length) #This is fine because of the get all data set thing. but it is only okay when you specifically call test

        # assert random_start + self.skipping_frames * self.sequence_length <= stop_frame
        image_names = ['frame_{}.jpg'.format(str(offset * self.skipping_frames + random_start).zfill(10)) for offset in range(self.sequence_length)]

        features = [self.load_and_resize(os.path.join(image_dir, image_name)) for image_name in image_names]

        input = {
            'rgb': torch.stack(features, 0),
        }

        labels = {
            'verb_class': verb_class,
            'noun_class': noun_class,
        }

        return (input, labels)
