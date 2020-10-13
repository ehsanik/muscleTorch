from training import metrics
import time
import tqdm
import torch
import logging
import os

def learning_rate(epoch, base_lr, lrm, step_size):
    base_lr = base_lr
    decay_rate = lrm
    step = step_size
    assert 1 <= epoch
    if 1 <= epoch <= step:
        return base_lr
    elif step <= epoch <= step * 2:
        return base_lr * decay_rate
    elif step * 2 <= epoch <= step * 3:
        return base_lr * decay_rate * decay_rate
    else:
        return base_lr * decay_rate * decay_rate * decay_rate

def train_one_epoch(model, loss, optimizer, data_loader, epoch, args):

    add_to_keys = 'Train'

    # Prepare model and optimizer
    model.train()
    loss.train()
    lr = learning_rate(epoch, args.base_lr, args.lrm, args.step_size)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    # Setup average meters
    data_time_meter = metrics.AverageMeter()
    batch_time_meter = metrics.AverageMeter()
    metrics_time_meter = metrics.AverageMeter()
    backward_time_meter = metrics.AverageMeter()
    forward_pass_time_meter = metrics.AverageMeter()
    loss_time_meter = metrics.AverageMeter()
    loss_meter = metrics.AverageMeter()
    accuracy_metric = [m(args) for m in args.arch.metric]
    loss_detail_meter = {loss_name:metrics.AverageMeter() for loss_name in loss.local_loss_dict}

    # Iterate over data
    timestamp = time.time()

    args.logging_module.network_conv_summary(model, (epoch - 1) * len(data_loader.dataset),
                                             add_to_keys=add_to_keys)

    for i, (input, target) in enumerate(tqdm.tqdm(data_loader)):

        # Move data to gpu
        batch_size = input['rgb'].size(0)
        if args.gpu_ids != -1:
            image_names = None
            if 'image_names' in input:
                image_names = input['image_names']
            input = {key: input[key].cuda(async=True) for key in input.keys() if key != 'image_names'}
            input['image_names'] = image_names

            target = {feature: target[feature].cuda(async=True) for feature in target.keys()}
        data_time_meter.update((time.time() - timestamp) / batch_size, batch_size)

        before_forward_pass_time = time.time()
        # Forward pass
        output, target = model(input, target)
        forward_pass_time_meter.update((time.time() - before_forward_pass_time) / batch_size, batch_size)


        before_loss_time = time.time()
        loss_output = loss(output, target)
        loss_time_meter.update((time.time() - before_loss_time)/batch_size, batch_size)

        # Backward pass and update weights
        before_backward_time = time.time()
        loss_output.backward()
        backward_time_meter.update((time.time() - before_backward_time)/batch_size, batch_size)

        output = {f: output[f].detach() for f in output.keys()}

        if i % args.break_batch == 0:
            optimizer.step()
            optimizer.zero_grad()

        # Bookkeeping on loss, accuracy, and batch time
        loss_meter.update(loss_output, batch_size)
        before_metrics_time = time.time()
        with torch.no_grad():
            for acc in accuracy_metric:
                acc.record_output(output, target, batch_size)
        metrics_time_meter.update((time.time() - before_metrics_time) / batch_size, batch_size)
        batch_time_meter.update((time.time() - timestamp), batch_size)


        # Log report
        dataset_length = len(data_loader.dataset)
        real_index = (epoch - 1) * dataset_length + (i * args.batch_size)

        loss_values = loss.local_loss_dict
        for loss_name in loss_detail_meter:
            (loss_val, data_size) = loss_values[loss_name]
            loss_detail_meter[loss_name].update(loss_val.item(), data_size)



        if i % args.tensorboard_log_freq == 0:
            result_log_dict = {
                'Time/Batch': batch_time_meter.avg,
                'Time/Data': data_time_meter.avg,
                'Time/Metrics': metrics_time_meter.avg,
                'Time/backward': backward_time_meter.avg,
                'Time/forward': forward_pass_time_meter.avg,
                'Time/loss': loss_time_meter.avg,
                'Loss': loss_meter.avg,
            }


            for loss_name in loss_detail_meter:
                result_log_dict['Loss/' + loss_name] = loss_detail_meter[loss_name].avg

            for ac in accuracy_metric:
                result_log_dict[type(ac).__name__] = ac.average()
            args.logging_module.log(result_log_dict, real_index + 1, add_to_keys=add_to_keys)
        if i % (args.tensorboard_log_freq * 50) == 0:
            if 'reconstructed_images' in output:
                output_image = output['reconstructed_images']
                target_image = target['reconstructed_images']
                args.logging_module.image_summary(output_image, target_image, real_index + 1, add_to_keys=add_to_keys)

        if i % (args.tensorboard_log_freq * 50) == 0 or i == len(data_loader) - 1:
            args.logging_module.visualize_results(input, output, target, real_index + 1, add_to_keys=add_to_keys)

        timestamp = time.time()

    dataset_length = len(data_loader.dataset)
    real_index = (epoch) * dataset_length
    result_log_dict = {
        'Time/Batch': batch_time_meter.avg,
        'Time/Data': data_time_meter.avg,
        'Time/Metrics': metrics_time_meter.avg,
        'Time/backward': backward_time_meter.avg,
        'Time/forward': forward_pass_time_meter.avg,
        'Time/loss': loss_time_meter.avg,
        'Loss': loss_meter.avg,
    }


    for loss_name in loss_detail_meter:
        result_log_dict['Loss/' + loss_name] = loss_detail_meter[loss_name].avg


    with torch.no_grad():
        for ac in accuracy_metric:
            result_log_dict[type(ac).__name__] = ac.average()
    args.logging_module.log(result_log_dict, epoch, add_to_keys=add_to_keys+'_Summary')

    if 'reconstructed_images' in output:
        output_image = output['reconstructed_images']
        target_image = target['reconstructed_images']
        args.logging_module.image_summary(output_image, target_image, real_index + 1, add_to_keys=add_to_keys)

    args.logging_module.visualize_results(input, output, target, real_index + 1, add_to_keys=add_to_keys)

    training_summary = ('Epoch: [{}] -- TRAINING SUMMARY\t'.format(epoch) + 'Time {batch_time.avg:.2f}   Data {data_time.avg:.2f}  Loss {loss:.6f}   {accuracy_report}'.format(batch_time=batch_time_meter, data_time=data_time_meter, loss=loss_meter.avg.item(), accuracy_report='\n'.join([ac.final_report() for ac in accuracy_metric])))
    logging.info(training_summary)
    logging.info('Full train result is at {}'.format(
        os.path.join(args.save, 'train.log')))
    with open(os.path.join(args.save, 'train.log'), 'a') as fp:
        fp.write('{}\n'.format(training_summary))
