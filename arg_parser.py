import argparse
import datasets
import datetime
import logging
import models
import os
import sys
import torch
import pprint
from utils.logging_util import LoggingModule
import utils.loss_util as loss
import time
from utils.constants import IMU_NAME_TO_INDEX



def loss_class(class_name):
    if class_name not in loss.__all__:
       raise argparse.ArgumentTypeError("Invalid Loss {}; choices: {}".format(
           class_name, loss.__all__))
    return getattr(loss, class_name)

def model_class(class_name):
    if class_name not in models.__all__:
        raise argparse.ArgumentTypeError("Invalid model {}; choices: {}".format(
            class_name, models.__all__))
    return getattr(models, class_name)


def dataset_class(class_name):
    if class_name not in datasets.__all__:
        raise argparse.ArgumentTypeError(
            "Invalid dataset {}; choices: {}".format(class_name,
                                                     datasets.__all__))
    return getattr(datasets, class_name)


def setup_logging(filepath, verbose):
    logFormatter = logging.Formatter(
        '%(levelname)s %(asctime)-20s:\t %(message)s')
    rootLogger = logging.getLogger()
    if verbose:
        rootLogger.setLevel(logging.DEBUG)
    else:
        rootLogger.setLevel(logging.INFO)
    logging.getLogger('PIL').setLevel(logging.WARNING)

    # Setup the logger to write into file
    fileHandler = logging.FileHandler(filepath)
    fileHandler.setFormatter(logFormatter)
    rootLogger.addHandler(fileHandler)

    # Setup the logger to write into stdout
    consoleHandler = logging.StreamHandler(sys.stdout)
    consoleHandler.setFormatter(logFormatter)
    rootLogger.addHandler(consoleHandler)


def get_non_default_flags_str(args, parser, *ignore):
    flags = []
    counter = 0
    for key, val in sorted(vars(args).items()):
        if key in ignore:
            continue
        if isinstance(val, type):
            val = val.__name__
        if val != parser.get_default(key):
            flags.append(key + '-' + str(val).replace(' ', '#'))
            counter += 1
        if counter > 5:
            break
    return '+'.join(flags)






def parse_args():
    parser = argparse.ArgumentParser(description='Dog project training script')
    parser.add_argument(
        'mode', default='train', nargs='?',
        choices=('train', 'test', 'testtrain'))
    parser.add_argument('--data', metavar='DIR', default='data',
                        help='path to dataset')
    parser.add_argument('--save', metavar='DIR', default='cache',
                        help='path to cache directory')
    parser.add_argument('--dataset', default='MultipleDogClipDataset',
                        help='Dataset to '
                        'use for training/test.', type=dataset_class)
    parser.add_argument(
        '--arch', '-a', metavar='ARCH', default='AlexNetImage2IMU',
        help='model to use for training/test.', type=model_class)
    parser.add_argument('-j', '--workers', default=5, type=int, metavar='N',
                        help='number of data loading workers')
    parser.add_argument('--verbose', action='store_true',
                        help='Level of logging the outputs')
    parser.add_argument('--epochs', default=90000, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-b', '--batch-size', default=64, type=int,
                        metavar='N', help='mini-batch size (default: 256)')
    parser.add_argument('--break-batch', default=1, type=int,
                        help='break batches with this factor to fit to memory.')
    parser.add_argument('--lrm', default=0.1, type=float, help='learning rate '
                        'multiplier.')    
    parser.add_argument('--base-lr', default=0.0001, type=float, help='base learning rate ')
    parser.add_argument('--reload', default=None, type=str, metavar='PATH',
                        help='path to latest checkpoint')
    parser.add_argument('--reload_dir', default=None, type=str, metavar='PATH')
    parser.add_argument('--reload_from_title', default=None, type=str, metavar='PATH')
    parser.add_argument('--no-strict', action='store_false', dest='strict',
                        help='Loading the weights from another model.')
    parser.add_argument('--pretrain', action='store_true', dest='pretrain',
                        help='Initialize the model with random intialization')
    parser.add_argument('--not-fixed-weights', action='store_false', dest='fixed_feature_weights')
    parser.add_argument('--input_feature_type', nargs='+', default=[])
    parser.add_argument('--reconst_resolution', default=224, type=int)
    parser.add_argument('--image_size', default=224, type=int,
                        help='Input image size')
    parser.add_argument('--seed', default=1, type=int)
    parser.add_argument('--log_results', action='store_true')
    parser.add_argument('--trainset_image_list', default='train_w_axis_angle.json',
                        help='Train dataset annotation file')
    parser.add_argument('--testset_image_list', default='test_w_axis_angle.json',
                        help='Test dataset annotation file')
    parser.add_argument('--valset_image_list', default='val.json_w_diff.json',
                        help='Validation dataset annotation file')
    parser.add_argument('--num_classes', default=8, type=int,
                        help='Number of classes per IMU')
    parser.add_argument('--num_imus', default=0, type=int,
                        help='Number of classes per IMU')
    parser.add_argument('--imu_names', nargs='+', default=[], type=str,
                        help='List of IMUs to train') #default=LIST_OF_IMUS
    parser.add_argument('--features_save_dir', default='feature_resnet')
    parser.add_argument('--features_dir', default='images', 
                        help='Address to read image features for LSTM networks')
    parser.add_argument(
        '--read_features', action='store_true',
        help='Indicate whether read features or use original image')
    parser.add_argument('--use_test_for_val', action='store_true',
                        help='Use this option to do final evaluation')
    parser.add_argument('--input_length', default=0, type=int,
                        help='Length of the input sequence')
    parser.add_argument('--output_length', default=1, type=int,
                        help='Length of the output sequence')
    parser.add_argument('--sequence_length', default=1, type=int,
                        help='Length of the sequence involved in training')
    parser.add_argument('--image_feature', default=512, type=int,
                        help='Size of Image features')
    parser.add_argument('--hidden_size', default=512, type=int,
                        help='Size of hidden layers in LSTM')
    parser.add_argument('--num_layers', default=1, type=int,
                        help='Number of layers of LSTM')
    parser.add_argument('--step_size', default=200, type=int,
                        help='Step size for reducing the learning rate')
    parser.add_argument('--tensorboard_log_freq', default=100, type=int, help='Frequency of logging to tensorboard')
    parser.add_argument('--dropout', default=0.0, type=float)
    parser.add_argument('--use_leakyrelu', default = False, action='store_true')
    parser.add_argument('--no_group_norm', dest='use_group_norm', action='store_false')
    parser.add_argument(
        '--planning_distance', default=3, type=int,
        help='Indicates the length of the predicting sequence in Planning network'
    )
    parser.add_argument('--save_frequency', default=1, type=int,
                        help='Frequency of saving the model, per epoch')
    parser.add_argument('--title', default='')
    parser.add_argument('--visualize', action='store_true')
    parser.add_argument('--double_feature', action='store_true')
    parser.add_argument('--manual_collate_fn', action='store_true')
    parser.add_argument('--data_parallel', action='store_true')
    parser.add_argument('--manual_data_size', default = None, type=int)
    parser.add_argument('--manual_epoch', default = None, type=int)
    parser.add_argument('--reload_from_title_epoch', default = -1, type=int)
    parser.add_argument('--tensorboard_viz_freq_epoch', default = 3, type=int)
    parser.add_argument('--manual_test_size', default = None, type=float)
    parser.add_argument('--number_of_trained_resnet_blocks', default = 5, type=int)
    parser.add_argument('--detach_level', default = 1000, type=float)
    parser.add_argument('--bert_mask_prob', default = 0.2, type=float)
    parser.add_argument('--reload_from_title_dir', default=None, type=str)
    parser.add_argument('--logdir', default = None, type=str)

    parser.add_argument(
        '--gpu-ids',
        type=int,
        default=-1,
        nargs='+',
        help='GPUs to use [-1 CPU only] (default: -1)')

    parser.add_argument('--loss', default='MSELoss', type=loss_class)
    parser.add_argument('--reorder_imu_classes', default=0, type=int)
    parser.add_argument('--transformer_do', default=0, type=float)
    parser.add_argument('--transformer_h', default=8, type=int)
    parser.add_argument('--transformer_n', default=6, type=int)
    parser.add_argument('--transformer_dff', default=1024, type=int) #: 2048 in original
    parser.add_argument('--bert_pe_size', default=32, type=int)
    parser.add_argument('--skipping_frames', default=1, type=int)
    parser.add_argument('--no_masking', action='store_true', default=False)
    parser.add_argument('--simple_feature_extractor', action='store_true', default=False)
    parser.add_argument('--replace_target_w_input', default=False, action='store_true') 
    args = parser.parse_args()


    if not args.logdir:
        args.logdir = args.data

    args.save = os.path.join(args.logdir, args.save)
    args.features_save_dir = os.path.join(args.data, args.features_save_dir)
    args.features_dir = os.path.join(args.data, args.features_dir)
    if args.reload_from_title_dir is None:
        args.reload_from_title_dir = args.data

    if args.log_results:
        assert args.mode != 'train'

    if args.gpu_ids == [-1]:
        args.gpu_ids = -1
        
    args.imus = [IMU_NAME_TO_INDEX[imu_name] for imu_name in args.imu_names]
    if args.num_imus is None:
        args.num_imus = len(args.imus)
    else:
        assert args.num_imus == len(args.imus)

    if args.gpu_ids != -1:
        torch.cuda.manual_seed(args.seed)

    args.imu_feature = len(args.imus) * args.num_classes
    assert args.break_batch == 1, 'We should resolve the followings before we continue'

    # Make log directory
    logging_path = os.path.join(args.logdir, 'runs/')
    local_start_time_str = time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime(time.time()))
    args.log_title = args.title + '_' + local_start_time_str
    log_dir = os.path.join(logging_path,  args.log_title)
    args.logging_module = LoggingModule(args, log_dir)

    timestamp = str(datetime.datetime.now()).replace(' ', '#').replace(':', '.')
    args.timestamp = timestamp
    args.save = os.path.join(
        args.save, args.arch.__name__, args.log_title,
        get_non_default_flags_str(args, parser, 'data', 'save', 'arch',
                                  'reload', 'title', 'workers', 'save_frequency', 'imu_names', 'batch-size', 'gpu-ids'), timestamp)
    os.makedirs(args.save, exist_ok=True)
    setup_logging(os.path.join(args.save, 'log.txt'), args.verbose)

    logging.info('Command: {}'.format(' '.join(sys.argv)))
    logging.info('Command line arguments parsed: {}'.format(
        pprint.pformat(vars(args))))

    assert not args.reload_from_title or not args.reload

    return args
