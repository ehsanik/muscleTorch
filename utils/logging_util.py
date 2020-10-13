from tensorboardX import SummaryWriter
import torch
import os
import random
import numpy as np
from utils.constants import DIRECTION_CLUSTER_CENTERS
import matplotlib as mpl
mpl.use('Agg')
from utils import drawing
from utils.visualization_util import save_image_list_to_gif, visualize_gaze, channel_last, channel_first, combine_image_table, put_epic_class_text_on_images, put_text_on_images, put_circles_on_image, make_full_label_text_image, get_human_pose_fig

from utils.visualization_util import normalize, depth_normalize, identity, put_labels_on_image

try:
    from StringIO import StringIO  # Python 2.7
except ImportError:
    from io import BytesIO as StringIO

def kernel_to_image(data, padsize=1):
    """Turns a convolutional kernel into an image of nicely tiled filters.
    :param data: numpy array in format N x C x H x W.
    :param padsize: optional int to indicate visual padding between the filters.
    :return: image of the filters in a tiled/mosaic layout
    """
    if len(data.shape) > 4:
        data = np.squeeze(data)
    data = np.transpose(data, (0, 2, 3, 1))
    data_shape = tuple(data.shape)
    min_val = np.min(np.reshape(data, (data_shape[0], -1)), axis=1)
    data = np.transpose((np.transpose(data, (1, 2, 3, 0)) - min_val), (3, 0, 1, 2))
    max_val = np.max(np.reshape(data, (data_shape[0], -1)), axis=1)
    data = np.transpose((np.transpose(data, (1, 2, 3, 0)) / max_val), (3, 0, 1, 2))

    n = int(np.ceil(np.sqrt(data_shape[0])))
    ndim = len(data.shape)
    padding = ((0, n ** 2 - data_shape[0]), (0, padsize), (0, padsize)) + ((0, 0),) * (ndim - 3)
    data = np.pad(data, padding, mode="constant", constant_values=0)
    # tile the filters into an image
    data_shape = data.shape
    data = np.transpose(np.reshape(data, ((n, n) + data_shape[1:])), ((0, 2, 1, 3) + tuple(range(4, ndim + 1))))
    data_shape = data.shape
    data = np.reshape(data, ((n * data_shape[1], n * data_shape[3]) + data_shape[4:]))
    return np.transpose((data * 225).astype(np.uint8), (2, 0, 1))



class ScalarMeanTracker(object):
    def __init__(self) -> None:
        self._sums = {}
        self._counts = {}

    def add_scalars(self, scalars):
        for k in scalars:
            if k not in self._sums:
                self._sums[k] = scalars[k]
                self._counts[k] = 1
            else:
                self._sums[k] += scalars[k]
                self._counts[k] += 1

    def pop_and_reset(self):
        means = {k: self._sums[k] / self._counts[k] for k in self._sums}
        self._sums = {}
        self._counts = {}
        return means

class LoggingModule(object):
    def __init__(self, args, log_dir):   
        print('initializing logger', log_dir)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        self.log_writer = SummaryWriter(log_dir=log_dir)
        self.output_length = args.output_length
        self.input_length = args.input_length
        self.sequence_length = args.sequence_length
        self.imus = args.imus
        self.imu_names = args.imu_names
        self.num_imus = args.num_imus
        self.number_of_items_to_visualize = 10
        self.DIRECTION_CLUSTER_CENTERS = DIRECTION_CLUSTER_CENTERS
        self.visualize = args.visualize
        self.dataset = args.dataset
        self.title = args.title
        if self.visualize:
            assert '/runs/' in log_dir
            self.gif_adr = log_dir.replace('/runs/', '/plot_gifs/')
            os.makedirs(self.gif_adr, exist_ok=True)

        if args.gpu_ids != -1:
            self.DIRECTION_CLUSTER_CENTERS = self.DIRECTION_CLUSTER_CENTERS.cuda()

    def recursive_write(self, item_to_write, epoch_number, add_to_keys=''):
        if type(item_to_write) == dict:
            for res in item_to_write:
                sub_item = item_to_write[res]
                new_translated_key = add_to_keys + '/' + res
                self.recursive_write(sub_item, epoch_number, add_to_keys=new_translated_key)
        elif type(item_to_write) == torch.Tensor and  item_to_write.dim() > 0 and item_to_write.numel() > 1:
            if (item_to_write.shape[0] == self.output_length) and (item_to_write.shape[1] == len(self.imus)):
                averaged_sequence = item_to_write.mean(1)
                for i in range(self.output_length):
                    self.log_writer.add_scalar(
                        add_to_keys + '/' + 'Time_' + str(i), averaged_sequence[i], epoch_number
                    )    
                averaged_imu = item_to_write.mean(0)
                for i in range(len(self.imus)):
                    self.log_writer.add_scalar(
                        add_to_keys + '/' + 'IMU_' + str(i), averaged_imu[i], epoch_number
                    )    
                self.log_writer.add_scalar(
                    add_to_keys + '/Overall', item_to_write.mean(), epoch_number
                )
            elif item_to_write.dim() == 1:
                for i in range(len(item_to_write)):
                    self.log_writer.add_scalar(
                        add_to_keys + '/' + 'Item_' + str(i), item_to_write[i], epoch_number
                    ) 
                self.log_writer.add_scalar(
                    add_to_keys + '/Overall', item_to_write.mean(), epoch_number
                )

            else:
                AssertionError('Not implemented')
        else:
            self.log_writer.add_scalar(
                add_to_keys, item_to_write, epoch_number
            )

    def subplot_summary(self, subplot_image_sequence, step, add_to_keys):
        sequence_length = len(subplot_image_sequence)
        for seq_index in range(sequence_length):
            output = subplot_image_sequence[seq_index].transpose(2, 0, 1)
            self.log_writer.add_image(tag="%s/time_%d_output" % (add_to_keys, step), img_tensor=output, global_step=step + seq_index)


    def visualize_feature(self, input, output, target, step, add_to_keys, feature_name, normalize_output_method):
        output_image = output[feature_name].repeat(1,1,3,1,1).float()
        target_image = target[feature_name].repeat(1,1,3,1,1).float()
        input_images = input['rgb']

        batch_size, seq_len, c, w, h = target_image.shape

        items_to_visualize = min(batch_size, self.number_of_items_to_visualize)

        output_image = output_image[-items_to_visualize:]
        target_image = target_image[-items_to_visualize:]

        target_image = normalize_output_method(target_image)
        output_image = normalize_output_method(output_image)

        input_images = normalize(input_images[-items_to_visualize:])
        all_images = [input_images, target_image, output_image]
        combined_images_before_append = torch.cat(all_images, dim=1)
        combined_images_before_append = channel_last(combined_images_before_append)
        combined_images = combine_image_table(combined_images_before_append)

        combined_images = channel_first(combined_images)
        self.log_writer.add_image(tag=('feature_' + feature_name + '/' + add_to_keys), img_tensor=combined_images, global_step=step)


    def save_gifs(self, input, target, output):

        if 'gaze_points' in output and 'gaze_points' in target:
            batch_to_visualize = 0
            rgb_images = channel_last(normalize(target['rgb'][batch_to_visualize]))
            target_gaze_images = put_circles_on_image(rgb_images, target['gaze_points'][batch_to_visualize])
            output_gaze_images = put_circles_on_image(rgb_images, output['gaze_points'][batch_to_visualize])

            combined_images = torch.stack([target_gaze_images, output_gaze_images], dim=1)
            image_names = input['image_names'][0]
            assert len(image_names) == 1
            start_image_name = image_names[0].split('/')[-1]

            save_image_list_to_gif(combined_images, 'gaze', self.gif_adr, start_image_name) # remove this


        if 'move_label' in output and 'move_label' in target:
            batch_to_visualize = 0
            rgb_images = channel_last(normalize(target['rgb'][batch_to_visualize]))
            target_move_labels = target['move_label'][batch_to_visualize]
            output_move_labels = output['cleaned_move_label'][batch_to_visualize]
            masked_moves = target_move_labels == -1
            correctness = target_move_labels == output_move_labels
            translate = {
                'larmu': 'LArm',
                'rarmu': 'RArm',
                'llegu': 'LLeg',
                'rlegu': 'RLeg',
                'neck': 'Neck',
                'body': 'Body',
            }
            imu_names = [translate[imu] for imu in self.imu_names]

            target_move_images = put_labels_on_image(rgb_images, target_move_labels, mask=masked_moves, imu_names=imu_names, is_gt=True)
            output_move_images = put_labels_on_image(rgb_images, output_move_labels, mask=masked_moves,  imu_names=imu_names, correctness=correctness, is_gt=False)
            combined_images = torch.stack([target_move_images, output_move_images], dim=1)
            image_names = input['image_names'][0]
            assert len(image_names) == 1
            start_image_name = image_names[0].split('/')[-1]
            save_image_list_to_gif(combined_images, 'move_label', self.gif_adr, start_image_name) # remove this

        if 'move_label' in output and 'move_label' in target and 'input_gaze_points' in target:
            batch_to_visualize = 0

            rgb_images = channel_last(normalize(target['rgb'][batch_to_visualize]))

            input_gaze_points = target['input_gaze_points'][batch_to_visualize]
            target_gaze_images = visualize_gaze(rgb_images, input_gaze_points)

            target_move_labels = target['move_label'][batch_to_visualize]
            output_move_labels = output['cleaned_move_label'][batch_to_visualize]
            masked_moves = target_move_labels == -1
            correctness = target_move_labels == output_move_labels
            target_move_images = put_labels_on_image(target_gaze_images, target_move_labels, mask=masked_moves, imu_names=self.imu_names)
            output_move_images = put_labels_on_image(rgb_images, output_move_labels, mask=masked_moves,  imu_names=self.imu_names, correctness=correctness)
            combined_images = torch.stack([target_move_images, output_move_images], dim=1)
            save_image_list_to_gif(combined_images, 'move_label', self.gif_adr) # remove this

        if 'gaze_points' in output and 'gaze_points' in target and 'move_label' in output and 'move_label' in target:
            batch_to_visualize = 0
            rgb_images = channel_last(normalize(target['rgb'][batch_to_visualize]))
            image_with_gaze = put_circles_on_image(rgb_images, target['gaze_points'][batch_to_visualize], color=(0,1,0))
            image_with_gaze = put_circles_on_image(image_with_gaze, output['gaze_points'][batch_to_visualize], color=(1,0,0))

            target_move_labels = target['move_label'][batch_to_visualize]
            output_move_labels = output['cleaned_move_label'][batch_to_visualize]
            masked_moves = target_move_labels == -1
            correctness = target_move_labels == output_move_labels
            translate = {
                'larmu': 'LArm',
                'rarmu': 'RArm',
                'llegu': 'LLeg',
                'rlegu': 'RLeg',
                'neck': 'Neck',
                'body': 'Body',
            }
            imu_names = [translate[imu] for imu in self.imu_names]

            gt_label = get_human_pose_fig(target_move_labels, masked_moves, imu_names, is_gt=True)
            output_label = get_human_pose_fig(output_move_labels, masked_moves, imu_names, is_gt=False)

            combined_images = torch.stack([image_with_gaze, gt_label, output_label], dim=1)
            image_names = input['image_names'][0]
            assert len(image_names) == 1
            start_image_name = image_names[0].split('/')[-1]
            save_image_list_to_gif(combined_images, 'no_text_move_gaze', self.gif_adr, start_image_name, duration=2./5.) # remove this




    def visualize_results(self, input, output, target, step, add_to_keys):



        if 'reconstructed_rgb' in output and 'reconstructed_rgb' in target:
            batch_to_visualize = 0
            # pdb.set_trace()
            output_reconstruct = output['reconstructed_rgb'][batch_to_visualize]
            target_reconstruct = target['reconstructed_rgb'][batch_to_visualize]

            all_images = [channel_last(normalize(img)) for img in [target_reconstruct, output_reconstruct]]

            combined_images_before_append = torch.stack(all_images, dim=1)
            combined_images = combine_image_table(combined_images_before_append)

            combined_images = channel_first(combined_images)
            self.log_writer.add_image(tag=('reconstruct_rgb_viz' + '/' + add_to_keys), img_tensor=combined_images, global_step=step)

        if 'verb_class' in output and 'verb_class' in target:
            batch_to_visualize = 10
            output_verb_class = output['verb_actual_class'][:batch_to_visualize]
            target_verb_class = target['verb_class'][:batch_to_visualize]

            rgb_images = (input['rgb'][:batch_to_visualize])
            batch_size, seq_len, c, w, h = rgb_images.shape
            # half_way = int(seq_len / 2)
            # rgb_images = rgb_images[:,[0,half_way, -1]]
            rgb_images = channel_last(normalize(rgb_images))

            combined_images_before_append = put_epic_class_text_on_images(rgb_images, target_verb_class, output_verb_class, self.dataset.VERB_CLASS_TO_NAME)
            combined_images = combine_image_table(combined_images_before_append)

            combined_images = channel_first(combined_images)
            self.log_writer.add_image(tag=('verb_class_rgb_viz' + '/' + add_to_keys), img_tensor=combined_images, global_step=step)

        if 'class_probs' in output and 'class_probs' in target:
            batch_to_visualize = 10
            output_verb_class = output['class_probs'][:batch_to_visualize]
            target_verb_class = target['class_probs'][:batch_to_visualize]

            rgb_images = (input['rgb'][:batch_to_visualize])
            batch_size, seq_len, c, w, h = rgb_images.shape
            # half_way = int(seq_len / 2)
            # rgb_images = rgb_images[:,[0,half_way, -1]]
            rgb_images = channel_last(normalize(rgb_images))


            class_names = self.dataset.VERB_CLASS_TO_NAME
            _, output_top_k = torch.topk(output_verb_class, k=5, dim=-1)
            _, target_top_k = torch.topk(target_verb_class, k=5, dim=-1)
            output_top_k = output_top_k.squeeze(1)
            target_top_k = target_top_k.squeeze(1)
            output_text_list = []
            target_text_list = []

            for b_ind in range(len(output_top_k)):
                output_verbs = '/'.join([class_names[cls.item()].split(' ')[0].split('/')[0] for cls in output_top_k[b_ind]])
                target_verbs = '/'.join([class_names[cls.item()].split(' ')[0].split('/')[0] for cls in target_top_k[b_ind]])
                output_text_list.append(output_verbs)
                target_text_list.append(target_verbs)
            combined_images_before_append = put_text_on_images(rgb_images, output_text_list, target_text_list, color_list=None, font_scale=0.3, line_type=1)

            combined_images = combine_image_table(combined_images_before_append)

            combined_images = channel_first(combined_images)
            self.log_writer.add_image(tag=('verb_class_rgb_viz' + '/' + add_to_keys), img_tensor=combined_images, global_step=step)

        if 'vind_class' in output and 'vind_class' in target:
            batch_to_visualize = 10

            output_pose_class = output['vind_actual_class'][:batch_to_visualize]
            target_pose_class = target['vind_class'][:batch_to_visualize]

            rgb_images = (input['rgb'][:batch_to_visualize])
            mask_images = (target['combined_mask'][:batch_to_visualize])

            batch_size, seq_len, c, w, h = rgb_images.shape

            rgb_images = channel_last(normalize(rgb_images))
            mask_images = channel_last(normalize(mask_images))
            combined_images = torch.cat([rgb_images, mask_images], dim=1)

            combined_images_before_append = put_epic_class_text_on_images(combined_images, target_pose_class, output_pose_class, {i:str(i) for i in range(100)})
            combined_images = combine_image_table(combined_images_before_append)

            combined_images = channel_first(combined_images)
            self.log_writer.add_image(tag=('vind_class' + '/' + add_to_keys), img_tensor=combined_images, global_step=step)



        if 'scene_class' in output and 'scene_class' in target:
            batch_to_visualize = 10

            output_scene_class = output['scene_actual_class'][:batch_to_visualize]
            target_scene_class = target['scene_class'][:batch_to_visualize]

            rgb_images = (input['rgb'][:batch_to_visualize])
            batch_size, seq_len, c, w, h = rgb_images.shape
            rgb_images = channel_last(normalize(rgb_images))

            combined_images_before_append = put_epic_class_text_on_images(rgb_images, target_scene_class, output_scene_class, self.dataset.SUN_SCENE_INDEX_TO_NAME)
            combined_images = combine_image_table(combined_images_before_append)

            combined_images = channel_first(combined_images)
            self.log_writer.add_image(tag=('scene_class_rgb_viz' + '/' + add_to_keys), img_tensor=combined_images, global_step=step)

        if 'move_label' in output and 'move_label' in target:

            from utils.constants import IMU_INDEX_TO_NAME
            def get_one_set_str(imus, imu_indices):
                result = ''
                for imu_ind in range(len(imu_indices)):
                    imu_name = IMU_INDEX_TO_NAME[imu_indices[imu_ind]]
                    move_label = imus[imu_ind]
                    if move_label == 0:
                        result += imu_name + '0-'
                    elif move_label == 1:
                        result += imu_name + '1-'
                return result


            def translate_move_label(move_labels, list_of_imus):
                seq_len, num_imus = move_labels.shape
                result = []
                for seq_ind in range(seq_len):
                    this_item = get_one_set_str(move_labels[seq_ind], list_of_imus)
                    result.append(this_item)
                return result


            batch_to_visualize = 0

            # output_move_label = output['move_label'][batch_to_visualize]
            output_move_label = output['cleaned_move_label'][batch_to_visualize]


            target_move_label = target['move_label'][batch_to_visualize]

            rgb_images = (target['rgb'][batch_to_visualize])
            seq_len = rgb_images.shape[0]
            output_images = channel_last(normalize(rgb_images)).cpu().numpy()
            target_images = channel_last(normalize(rgb_images)).cpu().numpy()

            list_of_images = [target_images[i] for i in range(seq_len)]
            list_of_images += [output_images[i] for i in range(seq_len)]



            target_titles = translate_move_label(target_move_label, self.imus)
            output_titles = translate_move_label(output_move_label, self.imus)
            target_titles = ['gt-'+x for x in target_titles]

            titles = target_titles + output_titles

            combined_images = drawing.subplot(list_of_images, 2, seq_len, 224, 224, 5, titles)

            combined_images = channel_first(combined_images)
            self.log_writer.add_image(tag=('move_label_viz' + '/' + add_to_keys), img_tensor=combined_images, global_step=step)


        if 'gaze_points' in output and 'gaze_points' in target:
            batch_to_visualize = 0
            rgb_images = channel_last(normalize(target['rgb'][batch_to_visualize]))
            target_gaze_images = visualize_gaze(rgb_images, target['gaze_points'][batch_to_visualize])
            output_gaze_images = visualize_gaze(rgb_images, output['gaze_points'][batch_to_visualize])
            combined_images_before_append = torch.stack([target_gaze_images, output_gaze_images], dim=1)
            combined_images = combine_image_table(combined_images_before_append)

            combined_images = channel_first(combined_images)
            self.log_writer.add_image(tag=('gaze_viz' + '/' + add_to_keys), img_tensor=combined_images, global_step=step)


        if 'depth' in output:
            self.visualize_feature(input, output, target, step, add_to_keys, 'depth', depth_normalize) 
        if 'walk' in output:
            output['real_walk'] = torch.argmax(output['walk'], dim=2).unsqueeze(2)
            target['real_walk'] = target['walk']
            self.visualize_feature(input, output, target, step, add_to_keys, 'real_walk', identity)


    
    def image_summary(self, output_images, target_images, step, add_to_keys):

        batch_size, sequence_length, _, _, _ = output_images.shape

        # img_summaries = []
        batch_to_visualize = random.randint(0, batch_size - 1)
        for seq_index in range(sequence_length):

            output = output_images[batch_to_visualize, seq_index].cpu()
            target = target_images[batch_to_visualize, seq_index].cpu()

            output = normalize(output)
            target = normalize(target)

            self.log_writer.add_image(tag="%s/time_%d_output" % (add_to_keys, seq_index), img_tensor=output, global_step=step)
            self.log_writer.add_image(tag="%s/time_%d_target" % (add_to_keys, seq_index), img_tensor=target, global_step=step)


    
    def log(self, dict_res, epoch_number, add_to_keys=''):
        for k in dict_res:
            if add_to_keys != '':
                translated_k = k + '/' + add_to_keys
            else:
                translated_k = k
            self.recursive_write(dict_res[k], epoch_number, add_to_keys=translated_k)


    def conv_variable_summaries(self, var, step, add_to_keys=""):
        # Useful stats for variables and the kernel images.
        scope = "conv_summaries/" + add_to_keys + "/filters"
        var_shape = var.shape
        if not (var_shape[0] == 1 and var_shape[1] == 1):
            if var_shape[2] < 3:
                var = np.tile(var, [1, 1, 3, 1])
                var_shape = var.shape
            summary_image = kernel_to_image(var[:, :3, :, :])#[np.newaxis, ...]
            #how is summary image
            self.log_writer.add_image(tag=scope, img_tensor=summary_image, global_step=step)

    def network_conv_summary(self, network, step, add_to_keys=''):
        for ii, (name, val) in enumerate(network.state_dict().items()):
            val = val.detach().cpu().numpy()
            name = "layer_%03d/" % ii + name
            if len(val.squeeze().shape) == 4:
                self.conv_variable_summaries(val, step, add_to_keys + '/' + name)             
