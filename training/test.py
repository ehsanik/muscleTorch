from training import metrics
import time
import tqdm
import torch
import logging
import os
import random
import json

def log_results_in_test(target, output, target_heatmap, output_heatmap):
    if 'gaze_points' in output and 'gaze_points' in target:
        if output_heatmap is None:
            heatmap_size = 224
            output_heatmap = torch.zeros((heatmap_size,heatmap_size))
            target_heatmap = torch.zeros((heatmap_size,heatmap_size))
        def update_heatmap(heatmap, gaze_points):
            assert gaze_points.shape[-1] == 2
            gaze = gaze_points.view(-1, 2)
            gaze = gaze * heatmap_size
            gaze = torch.round(gaze).long()
            masked_gaze = torch.any((gaze < 0) | (gaze >= heatmap_size), dim=-1)
            not_masked_gaze = ~masked_gaze
            gaze = gaze[not_masked_gaze]
            for gaze_ind in range(len(gaze)):
                heatmap[gaze[gaze_ind][0], gaze[gaze_ind][1]] += 1
        update_heatmap(output_heatmap, output['gaze_points'])
        update_heatmap(target_heatmap, target['gaze_points'])
        return target_heatmap, output_heatmap


def test_one_epoch(model, loss, data_loader, epoch, args):

    add_to_keys='Test'
    # Prepare model and loss
    model.eval()
    loss.eval()

    output_gaze_heatmap = None
    target_gaze_heatmap = None

    # Setup average meters
    data_time_meter = metrics.AverageMeter()
    batch_time_meter = metrics.AverageMeter()
    loss_meter = metrics.AverageMeter()
    accuracy_metric = [m(args) for m in args.arch.metric]
    loss_detail_meter = {loss_name:metrics.AverageMeter() for loss_name in loss.local_loss_dict}

    # Iterate over data
    timestamp = time.time()
    length_of_test_set = len(data_loader) + 0.0
    with torch.no_grad():
        for i, (input, target) in enumerate(tqdm.tqdm(data_loader)):

            # Move data to gpu
            batch_size = input['rgb'].size(0)
            if args.gpu_ids != -1:
                image_names = None
                if 'image_names' in input:
                    image_names = input['image_names']
                input = {key: input[key].cuda(async=True) for key in input.keys() if key != 'image_names'}
                input['image_names'] = image_names

                target = {k: target[k].cuda(async=True) for k in target}
            data_time_meter.update(time.time() - timestamp, batch_size)

            # Forward pass
            output, target = model(input, target)
            loss_output = loss(output, target)

            if args.log_results:
                target_gaze_heatmap, output_gaze_heatmap = log_results_in_test(target, output, target_gaze_heatmap, output_gaze_heatmap)


            loss_values = loss.local_loss_dict
            for loss_name in loss_detail_meter:
                (loss_val, data_size) = loss_values[loss_name]
                loss_detail_meter[loss_name].update(loss_val.item(), data_size)

            # Bookkeeping on loss, accuracy, and batch time
            loss_meter.update(loss_output, batch_size)
            for ac in accuracy_metric:
                ac.record_output(output, target, batch_size)

            batch_time_meter.update(time.time() - timestamp)
            # Log report
            dataset_length = len(data_loader.dataset)
            real_index = (epoch - 1) * dataset_length + (i * args.batch_size)
            if i % args.tensorboard_log_freq == 0:
                result_log_dict = {
                    'Time/Batch': batch_time_meter.avg,
                    'Time/Data': data_time_meter.avg,
                    'Loss': loss_meter.avg,
                }

                for loss_name in loss_detail_meter:
                    result_log_dict['Loss/' + loss_name] = loss_detail_meter[loss_name].avg

                for ac in accuracy_metric:
                    result_log_dict[type(ac).__name__] = ac.average()
                args.logging_module.log(result_log_dict, real_index + 1, add_to_keys=add_to_keys)

                if random.random() > 0.9:
                    if 'reconstructed_images' in output:
                        output_image = output['reconstructed_images']
                        target_image = target['reconstructed_images']
                        args.logging_module.image_summary(output_image, target_image, real_index + 1, add_to_keys=add_to_keys)


            timestamp = time.time()

            if args.manual_test_size and i / length_of_test_set > args.manual_test_size:
                break
            if i % (args.tensorboard_log_freq * 50) == 0 or i == len(data_loader) - 1:
                args.logging_module.visualize_results(input, output, target, real_index + 1, add_to_keys=add_to_keys)
            if args.visualize:
                args.logging_module.save_gifs(input, target, output)

    dataset_length = len(data_loader.dataset)
    real_index = (epoch) * dataset_length
    result_log_dict = {
        'Time/Batch': batch_time_meter.avg,
        'Time/Data': data_time_meter.avg,
        'Loss': loss_meter.avg,
    }
    if 'reconstructed_images' in output:
        output_image = output['reconstructed_images']
        target_image = target['reconstructed_images']
        args.logging_module.image_summary(output_image, target_image, real_index + 1, add_to_keys=add_to_keys)


    for loss_name in loss_detail_meter:
        result_log_dict['Loss/' + loss_name] = loss_detail_meter[loss_name].avg
    for ac in accuracy_metric:
        result_log_dict[type(ac).__name__] = ac.average()
    args.logging_module.log(result_log_dict, epoch, add_to_keys=add_to_keys+'_Summary')
    args.logging_module.log(result_log_dict, epoch, add_to_keys=add_to_keys+'_Summary')

    testing_summary = (
        'Epoch: [{}] -- TESTING SUMMARY\t'.format(epoch) +
        'Time {batch_time.sum:.2f}   Data {data_time.sum:.2f}   '
        'Loss {loss:.6f}   {accuracy_report}'.format(
            batch_time=batch_time_meter, data_time=data_time_meter,
            loss=loss_meter.avg.item(), accuracy_report='\n'.join(
                [ac.final_report() for ac in accuracy_metric])))
    logging.info(testing_summary)
    logging.info('Full test result is at {}'.format(
        os.path.join(args.save, 'test.log')))
    with open(os.path.join(args.save, 'test.log'), 'a') as fp:
        fp.write('{}\n'.format(testing_summary))

    if args.log_results:
        os.makedirs(os.path.join(args.data, 'log_results'), exist_ok=True)
        log_json_name = os.path.join(args.data, 'log_results', args.log_title + '.json')
        print('log in ', log_json_name)
        with open(log_json_name, 'w') as log_result_file:
            result_dict = {
                'output_gaze_heatmap': output_gaze_heatmap.cpu().numpy().tolist(),
                'target_gaze_heatmap': target_gaze_heatmap.cpu().numpy().tolist(),
            }
            json.dump(result_dict, log_result_file)

