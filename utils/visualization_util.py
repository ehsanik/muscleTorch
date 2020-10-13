import pdb
import pybullet as p
import torch
import os
import imageio
import cv2
import numpy as np
from datetime import datetime
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw
import time

def get_moving_not_moving(move_labels, mask, imu_names, correctness):
    assert len(imu_names) == len(move_labels)
    moving_correct = []
    not_moving_correct = []
    moving_wrong = []
    not_moving_wrong = []
    for i in range(len(move_labels)):
        if not mask[i]:
            if move_labels[i] == 0:
                if correctness[i]:
                    not_moving_correct.append(imu_names[i])
                else:
                    not_moving_wrong.append(imu_names[i])
            elif move_labels[i] == 1:
                if correctness[i]:
                    moving_correct.append(imu_names[i])
                else:
                    moving_wrong.append(imu_names[i])
    return moving_correct, not_moving_correct, moving_wrong, not_moving_wrong

def get_human_pose_fig(move_labels, masked_moves, imu_names, is_gt=True):
    seq_len = move_labels.shape[0]
    human_image = cv2.imread('utils/human_fig.jpg')
    human_image = cv2.resize(human_image, (224,224)).astype(float) / 255.


    if is_gt:
        circle_color = (0,0.6,0)
    else:
        circle_color = (1,0,0)

    #Add corresponding texts
    font = ImageFont.truetype('/Library/Fonts/Arial Bold.ttf', 13)
    this_image = Image.fromarray(np.uint8(human_image * 255))
    draw = ImageDraw.Draw(this_image)
    if is_gt:
        text_to_write = 'Ground-Truth'
    else:
        text_to_write = 'Predictions'
    start = 5
    draw.text((5, start), text_to_write, (int(circle_color[0] * 255), int(circle_color[1] * 255), int(circle_color[2] * 255)), font)
    human_image = np.array(this_image).astype(float) / 255.


    body_joint_coord = {
        'Neck':(254,70),
        'Body':(254,161),
        'LLeg':(289,360),
        'RLeg':(219,360),
        'LArm':(359,117),
        'RArm':(149,117),
    }

    body_joint_coord = {key:(int(val[0] / 512. * 224.), int(val[1] / 512. * 224.)) for (key, val) in body_joint_coord.items()}

    all_images = []
    for seq_ind in range(seq_len):

        this_image = human_image.copy()
        # draw = ImageDraw.Draw(this_image)


        for imu_ind in range(len(imu_names)):
            name = imu_names[imu_ind]
            if not masked_moves[seq_ind][imu_ind]:
                if move_labels[seq_ind][imu_ind] == 1:
                    color = circle_color
                    this_image = cv2.circle(this_image, (body_joint_coord[name][0], body_joint_coord[name][1]), radius=10, color=color, thickness=2)


        all_images.append(this_image)

    return torch.Tensor(all_images)


def make_full_label_text_image(target_move_labels, output_move_labels, masked_moves, correctness, imu_names):

    total_gt = []
    total_output = []
    font = ImageFont.truetype('/Library/Fonts/Arial Bold.ttf', 13)
    # gt_color = (3, 252, 236)

    seq_len = target_move_labels.shape[0]
    for seq_ind in range(seq_len):

        #GT making
        gt_label_img = np.zeros((224,224,3))
        labels = get_moving_not_moving(target_move_labels[seq_ind], masked_moves[seq_ind], imu_names, correctness=target_move_labels[seq_ind] == target_move_labels[seq_ind])
        moving, not_moving, _, _ = [label for label in labels]
        gray_area = set(imu_names) - set(moving + not_moving)
        moving, not_moving, gray_area = [','.join(label) for label in [moving, not_moving, gray_area]]

        this_image = Image.fromarray(np.uint8(gt_label_img * 255))
        draw = ImageDraw.Draw(this_image)

        # draw.text((int(w / 2), 5), 'GT', (0,0,0), font)
        start = 5
        draw.text((5, start), 'Moving:', (255,51,255), font)
        draw.text((5, start + 20), moving, (255,51,255), font)

        draw.text((5, start + 60), 'Not Moving:', (102, 178,255), font)
        draw.text((5, start + 80), not_moving, (102, 178,255), font)

        draw.text((5, start + 120), 'No Label:', (192,192,192), font)
        draw.text((5, start + 140), gray_area, (192,192,192), font)

        draw.text((5, 224 - 40), 'Ground-truth Gaze in Green', (0,255,0), font)

        gt_label_img = np.array(this_image).astype(float) / 255.
        total_gt.append(gt_label_img)


        #Output making
        output_label_img = np.zeros((224,224,3))
        labels = get_moving_not_moving(output_move_labels[seq_ind], masked_moves[seq_ind], imu_names, correctness=correctness[seq_ind])
        moving_correct, not_moving_correct, moving_wrong, not_moving_wrong = [label for label in labels]
        all_corrects = moving_correct + not_moving_correct
        all_wrongs = moving_wrong + not_moving_wrong
        all_corrects, all_wrongs = [','.join(label) for label in [all_corrects, all_wrongs]]

        this_image = Image.fromarray(np.uint8(output_label_img * 255))
        draw = ImageDraw.Draw(this_image)

        start = 5
        draw.text((5, start), 'Correct Predictions:', (0,255,0), font)
        draw.text((5, start + 20), all_corrects, (0,255,0), font)

        draw.text((5, start + 70), 'Wrong Predictions:', (255,0,0), font)
        draw.text((5, start + 90), all_wrongs, (255,0,0), font)

        draw.text((5, 224 - 40), 'Predicted Gaze in Red', (255,0,0), font)


        output_label_img = np.array(this_image).astype(float) / 255.
        total_output.append(output_label_img)

    return torch.Tensor(total_gt), torch.Tensor(total_output)
def put_labels_on_image(images, move_labels, mask, imu_names, correctness=None, is_gt=False):
    images = images.detach().cpu().numpy().copy() # not efficient
    seq_len, w, h, c = images.shape
    if correctness is None:
        correctness = move_labels == move_labels
    for seq_ind in range(seq_len):
        labels = get_moving_not_moving(move_labels[seq_ind], mask[seq_ind], imu_names, correctness[seq_ind])
        moving_correct_list, not_moving_correct_list, moving_wrong_list, not_moving_wrong_list = [label for label in labels]
        moving_correct, not_moving_correct, moving_wrong, not_moving_wrong = [','.join(label) for label in labels]

        # font = cv2.FONT_HERSHEY_SIMPLEX
        # font_scale = 0.5
        # line_type = 1 #2
        font = ImageFont.truetype('/Library/Fonts/Arial Bold.ttf', 13)


        if is_gt:

            gt_color = (3, 252, 236)
            this_image = Image.fromarray(np.uint8(images[seq_ind] * 255))
            draw = ImageDraw.Draw(this_image)

            # draw.text((int(w / 2), 5), 'GT', (0,0,0), font)
            draw.text((5, 5), 'Moving:', gt_color, font)
            draw.text((5, 25), moving_correct, gt_color, font)

            draw.text((5, h - 40), 'Not Moving:', gt_color, font)
            draw.text((5, h - 20), not_moving_correct, gt_color, font)


            images[seq_ind] = np.array(this_image).astype(float) / 255.


        else:
            this_image = Image.fromarray(np.uint8(images[seq_ind] * 255))
            draw = ImageDraw.Draw(this_image)

            correct_list = ','.join(moving_correct_list + not_moving_correct_list)
            wrong_list = ','.join(moving_wrong_list + not_moving_wrong_list)

            draw.text((5, 5), 'Correct:', (0,255,0), font)
            draw.text((5, 25), correct_list, (0,255,0), font)

            draw.text((5, h - 40), 'Wrong:', (255,99,71), font)
            draw.text((5, h - 20), wrong_list, (255,99,71), font)

            images[seq_ind] = np.array(this_image).astype(float) / 255.

    return torch.Tensor(images) # not efficient

def put_circles_on_image(images, gaze_points, color=(1,0.1,0.1)):
    mask = torch.any(gaze_points == -1, dim=-1)
    images = images.detach().cpu().numpy().copy() # not efficient
    seq_len, w, h, c = images.shape
    for seq_ind in range(seq_len):

        if mask[seq_ind]:
            continue
        this_gaze = (gaze_points[seq_ind] * 224).long()
        images[seq_ind] = cv2.circle(images[seq_ind], (this_gaze[1].item(), this_gaze[0].item()), radius=10, color=color, thickness=2)

    return torch.Tensor(images) # not efficient


def put_epic_class_text_on_images(rgb_images, target_verb_class, output_verb_class, verb_class_to_name):
    batch_size, seq_len, w, h, c = rgb_images.shape
    color_list = []
    output_text_list = []
    target_text_list = []
    for batch_ind in range(batch_size):
        if target_verb_class[batch_ind] == output_verb_class[batch_ind]:
            color = (0, 1, 0)
        else:
            color = (1, 0, 0)
        color_list.append(color)
        output_verb = 'output ' + verb_class_to_name[output_verb_class[batch_ind].item()]
        target_verb = 'target ' + verb_class_to_name[target_verb_class[batch_ind].item()]
        output_text_list.append(output_verb)
        target_text_list.append(target_verb)
    updated_rgb_images = put_text_on_images(rgb_images, output_text_list, target_text_list, color_list)
    return updated_rgb_images

def put_text_on_images(rgb_images, output_text_list, target_text_list, color_list=None, font_scale=0.5, line_type=2):

    rgb_images = rgb_images.detach().cpu().numpy().copy()
    batch_size, seq_len, w, h, c = rgb_images.shape
    font = cv2.FONT_HERSHEY_SIMPLEX
    if color_list is None:
        color_list = [(0,0,1) for _ in range(batch_size)]
    for batch_ind in range(batch_size):
        color = color_list[batch_ind]
        output_verb = output_text_list[batch_ind]
        target_verb = target_text_list[batch_ind]
        rgb_images[batch_ind, 0] = cv2.putText(rgb_images[batch_ind, 0], output_verb, (10, h - 30), font, font_scale, color, line_type)
        rgb_images[batch_ind, 0] = cv2.putText(rgb_images[batch_ind, 0], target_verb, (10, 10), font, font_scale, color, line_type)
    return torch.Tensor(rgb_images)


def combine_image_table(image_list):
    if len(image_list.shape) == 5:

        seq_len, cols, w, h, c = image_list.shape

        pallet = torch.zeros((w * seq_len, h * cols, c))
        for row_ind in range(seq_len):
            for col_ind in range(cols):
                pallet[row_ind * w: (row_ind + 1) * w, col_ind * h: (col_ind + 1) * h, :] = image_list[row_ind, col_ind]

    else:
        pdb.set_trace()

    return pallet


def combine_image_columns(image_list):
    seq_len, cols, w, h, c = image_list.shape

    pallet = torch.zeros((seq_len, w, h * cols, c))

    for col_ind in range(cols):
        pallet[:, :, col_ind * h: (col_ind + 1) * h, :] = image_list[:, col_ind]

    return pallet

def save_image_list_to_gif(image_list, gif_name, gif_dir, image_name=None, duration=1/5):
    # if gif_name is None:
    if image_name is None:
        pdb.set_trace()
        gif_name = datetime.now().strftime('%m_%d_%Y_%H_%M_%S') + '_' + gif_name + '.gif'
    else:
        gif_name = '{}_{}.gif'.format(image_name, gif_name)
    gif_adr = os.path.join(gif_dir, gif_name)

    pallet = combine_image_columns(image_list)


    if not os.path.exists(gif_dir):
        os.makedirs(gif_dir)
    imageio.mimsave(gif_adr, (pallet * 255.).type(torch.uint8),format='GIF', duration=duration)

    if 'just_one_' in gif_adr:
        gif_image_folder = gif_adr.replace('.gif', '_folder')
        os.makedirs(gif_image_folder)
        number_repeats = 1
        for repeat_ind in range(number_repeats):
            for seq_ind in range(pallet.shape[0]):
                imageio.imsave(os.path.join(gif_image_folder, '{}_repeat{}.png'.format(seq_ind, repeat_ind)), (pallet[seq_ind] * 255.).type(torch.uint8),format='PNG')
            time.sleep(1)

        #Save video
        video_adr = gif_adr.replace('.gif', '.avi')
        height, width, layers = pallet[seq_ind].shape
        video = cv2.VideoWriter(video_adr, cv2.VideoWriter_fourcc(*'MJPG'), 2, (width,height))
        repeat_video = 3
        numpy_images = pallet.cpu().detach().numpy()[:,:,:,[2,1,0]]
        for repeat_ind in range(repeat_video):
            for seq_ind in range(numpy_images.shape[0]):
                video.write((numpy_images[seq_ind] * 255.).astype(np.uint8))
        cv2.destroyAllWindows()
        video.release()

    print('Saved result in ', gif_adr)

def put_gaze(rgb_img, gaze, length=5, color=[1,0,0]):
    color=torch.tensor(color)
    rgb_img[gaze[0] - length: gaze[0] + length,gaze[1] - length: gaze[1] + length] = color
    #Do not switch 0 and 1, this is checked
    #maybe make sure -len , + len not out of bound it seems allright


def visualize_gaze(rgb_images_original, gaze_points):
    seq_len, w, h, c = rgb_images_original.shape
    rgb_images = rgb_images_original.clone().detach().cpu()
    size = torch.Tensor([w,h])
    size = size.to(gaze_points.device)
    mask_gaze = torch.any(gaze_points == -1, dim=1)
    gaze_points = gaze_points * size
    gaze_points = gaze_points.long()

    for seq_ind in range(seq_len):
        if not mask_gaze[seq_ind]:
            put_gaze(rgb_images[seq_ind], gaze_points[seq_ind])
    return rgb_images

def get_rgb():
    output_w, output_h = 500,500
    viewMat = (0.7309443354606628, 0.25749075412750244, -0.6319959759712219, 0.0, -0.6824371218681335, 0.27579304575920105, -0.6769178509712219, 0.0, 0.0, 0.9260867834091187, 0.3773106038570404, 0.0, -0.0, -0.0, -4.999999523162842, 1.0)
    projMat = (0.7499999403953552, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, -1.0000200271606445, -1.0, 0.0, 0.0, -0.02000020071864128, 0.0)
    w, h, rgb, depth, segmmask = p.getCameraImage(output_w, output_h, viewMatrix=viewMat, projectionMatrix=projMat)

    return rgb[:, :, :3]


def depth_normalize(tensor):
    return torch.clamp((tensor + 1) / 2, 0, 1)
def image_normalize(tensor):
    return normalize(tensor)
def identity(tensor):
    return tensor
def segmentation_mask(tensor):
    tensor[tensor >= 0.5] = 1
    tensor[tensor <= 0.5] = 0
    return tensor

def channel_first(img):
    if type(img)==torch.Tensor:
        return img.transpose(-1, -2).transpose(-2, -3)
    if type(img)==np.ndarray:
        return img.swapaxes(-1, -2).swapaxes(-2, -3)
def channel_last(img):
    if type(img)==torch.Tensor:
        return img.transpose(-3, -2).transpose(-2, -1)
    if type(img)==np.ndarray:
        return img.swapaxes(-3, -2).swapaxes(-2, -1)

def normalize(img):
    mean=torch.Tensor([0.485, 0.456, 0.406]).to(img.device)
    std=torch.Tensor([0.229, 0.224, 0.225]).to(img.device)
    img = channel_last(img)
    img = (img * std + mean)
    img = torch.clamp(img, 0, 1)
    return channel_first(img)

