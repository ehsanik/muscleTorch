"""
=================
Command line argument parser and loading the models.
=================
"""
from pathlib import Path
import logging
import os
import random
import torch
from arg_parser import parse_args
from torch.utils.data.dataloader import default_collate

from training.train import train_one_epoch
from training.test import test_one_epoch
import numpy as np



def get_data_loaders(args):
    train_dataset = args.dataset(args, train=True)
    val_dataset = args.dataset(args, train=False)
    # Do not shuffle dataset in save_feats mode to get consistent order of
    # inputs for saving features.
    train_shuffle = (args.mode != 'save_feats' and args.mode != 'pseudo_supervision' and args.mode != 'save_scene_label')
    test_shuffle = False
    if args.visualize or args.mode == 'nearest_neighbor':
        train_shuffle = True
        test_shuffle = True
    if not args.manual_collate_fn:
        collate_fn = default_collate
    else:
        raise Exception('Not Implemented')
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size,
        shuffle=train_shuffle, num_workers=args.workers, pin_memory=True, collate_fn=collate_fn)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size,
        shuffle=test_shuffle, num_workers=args.workers, pin_memory=True, collate_fn=collate_fn)
    args.train_loader = train_loader
    return train_loader, val_loader


def get_model_and_loss(args):
    model = args.arch(args)
    restarting_epoch = 0
    if args.gpu_ids != -1:
        model = model.cuda()
    reload_adr = None
    if args.reload is not None:
        reload_adr = args.reload
    elif args.reload_from_title is not None:
        file = [f for f in Path(os.path.join(args.reload_from_title_dir, 'cache')).glob('**/' + args.reload_from_title)]
        assert len(file) == 1
        file = file[0]
        all_saved_models = [str(f) for f in file.glob('**/*.pytar')]
        epoch_indices = [int(mod.split('_')[-1].replace('.pytar', '')) for mod in all_saved_models]
        if args.reload_from_title_epoch > 0:
            latest_index = epoch_indices.index(args.reload_from_title_epoch)
        else:
            latest_index = np.argmax(np.array(epoch_indices))
        reload_adr = all_saved_models[latest_index]

    if reload_adr is not None:
        if args.gpu_ids == -1:
            loaded_weights = torch.load(reload_adr, map_location='cpu')
        else: 
            loaded_weights = torch.load(reload_adr)
        print('Exact address', reload_adr)
        model.load_state_dict(loaded_weights, strict=args.strict)
        epoch_index = reload_adr.split('_')[-1].replace('.pytar', '')
        try:
            epoch_index = int(epoch_index)
        except Exception:
            epoch_index = 0
        restarting_epoch = epoch_index
        print('Restarting from epoch', restarting_epoch)

    if not args.strict:
        restarting_epoch = 0

    if args.manual_epoch is not None:
        restarting_epoch = args.manual_epoch
        print('Manually setting the epoch', restarting_epoch)
    
    loss = model.loss(args)
    if args.gpu_ids != -1:
        loss = loss.cuda()
    logging.info('Model: {}'.format(model))
    logging.info('Loss: {}'.format(loss))

    if args.data_parallel:
        model = torch.nn.DataParallel(model, device_ids=args.gpu_ids)

    return model, loss, restarting_epoch


def main():
    args = parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    logging.info('Reading dataset metadata')
    train_loader, val_loader = get_data_loaders(args)
    args.classification_weights = train_loader.dataset.classification_weights

    logging.info('Constructing model')
    model, loss, restarting_epoch = get_model_and_loss(args)



    if args.mode == 'train':
        if not args.data_parallel:
            optimizer = model.optimizer()
        else:
            optimizer = torch.optim.Adam(model.parameters(), lr=args.base_lr)

        for i in range(restarting_epoch, args.epochs):
            train_one_epoch(model, loss, optimizer, train_loader, i + 1,
                                   args)
            if i % args.save_frequency == 0:
                torch.save(
                    model.state_dict(),
                    os.path.join(args.save,
                                 'model_state_{:02d}.pytar'.format(i + 1)))

    elif args.mode == 'test' or args.mode == 'testtrain':
        if args.mode == 'testtrain':
            val_loader = train_loader
        if args.reload_dir is not None:
            all_saved_models = [f for f in os.listdir(args.reload_dir) if f.endswith('.pytar')]
            all_indices = [f.split('_')[-1].replace('.pytar', '') for f in all_saved_models]
            int_indices = [int(f) for f in all_indices]
            int_indices.sort()
            for epoch in int_indices:
                args.reload = os.path.join(args.reload_dir, 'model_state_{:02d}.pytar'.format(epoch))
                model, loss, restarting_epoch = get_model_and_loss(args)
                test_one_epoch(model, loss, val_loader, epoch, args)
        else:
            test_one_epoch(model, loss, val_loader, 0, args)
    else:
        raise NotImplementedError("Unsupported mode {}".format(args.mode))


if __name__ == '__main__':
    main()
