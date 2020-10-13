"""
=================
This is the basic model containing place holder/implementation for necessary functions.

All the models in this project inherit from this class
=================
"""

import torch.nn as nn
import logging


class BaseModel(nn.Module):

    def __init__(self, args):
        super(BaseModel, self).__init__()
        self.base_lr = args.base_lr
        self.lrm = args.lrm
        self.step_size = args.step_size
        self.detach_level = args.detach_level

    def loss(self, args):
        return nn.CrossEntropyLoss()

    def optimizer(self):
        # Learning rate here is just a place holder. This will be overwritten
        # at training time.
        raise NotImplementedError




    def evaluation_report(self, output, target):
        raise NotImplementedError

    def load_state_dict(self, state_dict, strict=True):
        """Copies parameters and buffers from :attr:`state_dict` into
        this module and its descendants. If :attr:`strict` is ``True`` then
        the keys of :attr:`state_dict` must exactly match the keys returned
        by this module's :func:`state_dict()` function.
        Arguments:
            state_dict (dict): A dict containing parameters and
                persistent buffers.
            strict (bool): Strictly enforce that the keys in :attr:`state_dict`
                match the keys returned by this module's `:func:`state_dict()`
                function.
        """

        own_state = self.state_dict()
        copied = []
        extra_in_state_dict = []
        extra_in_own_dict = []
        for name, param in state_dict.items():
            original_name = name
            if not strict and not name in own_state and name[:len('module.')] == 'module.':
                name = name.replace('module.', '')
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    # backwards compatibility for serialized parameters
                    param = param.data
                try:

                    if own_state[name].size() != param.size() and not strict:
                        print('Size not matched', name)
                        extra_in_own_dict.append(name)
                        extra_in_state_dict.append(original_name)
                        continue
                    own_state[name].copy_(param)
                    copied.append(name)
                    print('Copied {} to {}'.format(original_name, name))
                except Exception:
                    raise RuntimeError(
                        'While copying the parameter named {}, '
                        'whose dimensions in the model are {} and '
                        'whose dimensions in the checkpoint are {}.'.format(
                            name, own_state[name].size(), param.size()))
            elif strict:
                raise KeyError('unexpected key "{}" in state_dict'.format(name))

            else:
                logging.warning(
                    'Parameter {} not found in own state'.format(original_name))
                extra_in_state_dict.append(original_name)
        if strict:
            missing = set(own_state.keys()) - set(state_dict.keys())
            if len(missing) > 0:
                raise KeyError(
                    'missing keys in state_dict: "{}"'.format(missing))
        else:
            missing = set(own_state.keys()) - set(copied)
            extra_in_own_dict += list(missing)
            # missing = set(own_state.keys()) - set(state_dict.keys())
            if len(missing) > 0:
                logging.warning(
                    'missing keys in state_dict: "{}"'.format(missing))

        print('Copied', copied)
        print('Exist only in this models dict', extra_in_own_dict)
        print('Exist only in loaded weights dict', extra_in_state_dict)
