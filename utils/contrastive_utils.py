'''
Borrowed from https://github.com/bl0/moco/
'''

import torch
import torch.nn as nn
import math
import torch.nn.functional as F

def moment_update(model, model_ema, m):
    """ model_ema = m * model_ema + (1 - m) model
    https://github.com/bl0/moco/blob/2830605b9abe543e9ec879a8c62f84a70f48b7f8/lib/util.py#L133
    """
    with torch.no_grad():
        for p1, p2 in zip(model.parameters(), model_ema.parameters()):
            p2.data.mul_(m).add_(1 - m, p1.detach().data)

def set_bn_train(model):
    def set_bn_train_helper(m):
        classname = m.__class__.__name__
        if classname.find('BatchNorm') != -1:
            m.train()

    model.eval()
    model.apply(set_bn_train_helper)

def get_moco_labels(model, first_image, augmented_image):

    with torch.no_grad():
        moment_update(model.feature_extractor, model.moco_feature_extractor, model.alpha)
        moment_update(model.feature_linear, model.moco_feature_linear, model.alpha)

    query_features = model.feature_extractor(first_image.unsqueeze(1)).squeeze(1)
    query_features = model.feature_linear(query_features)
    query_features = F.normalize(query_features, dim=-1)


    with torch.no_grad():

        shuffle_order = torch.randperm(augmented_image.shape[0]).to(augmented_image.device)
        unshuffle_order = torch.zeros(augmented_image.shape[0], dtype=torch.int64, device=augmented_image.device)
        unshuffle_order.index_copy_(0, shuffle_order, torch.arange(augmented_image.shape[0], device=augmented_image.device))
        augmented_image = augmented_image[shuffle_order].contiguous()


        augmented_features = model.moco_feature_extractor(augmented_image.unsqueeze(1)).squeeze(1)
        augmented_features = model.moco_feature_linear(augmented_features)
        augmented_features = F.normalize(augmented_features, dim=-1)

        augmented_features = augmented_features[unshuffle_order].contiguous()

        augmented_features_all = augmented_features.detach()
    out = model.contrast(query_features, augmented_features, augmented_features_all)
    return out

class MemoryMoCo(nn.Module):
    """Fixed-size queue with momentum encoder
    https://github.com/bl0/moco/blob/2830605b9abe543e9ec879a8c62f84a70f48b7f8/lib/NCE/Contrast.py#L6
    """
    def __init__(self, feature_dim, queue_size, temperature=0.07):
        super(MemoryMoCo, self).__init__()
        self.queue_size = queue_size
        self.temperature = temperature
        self.index = 0

        # noinspection PyCallingNonCallable
        self.register_buffer('params', torch.tensor([-1]))
        stdv = 1. / math.sqrt(feature_dim / 3)
        memory = torch.rand(self.queue_size, feature_dim, requires_grad=False).mul_(2 * stdv).add_(-stdv)
        self.register_buffer('memory', memory)



    def forward(self, q, k, k_all):
        k = k.detach()
        l_pos = (q * k).sum(dim=-1, keepdim=True)  # shape: (batchSize, 1)
        # : remove clone. need update memory in backwards
        l_neg = torch.mm(q, self.memory.clone().detach().t())
        out = torch.cat((l_pos, l_neg), dim=1)
        out = torch.div(out, self.temperature).contiguous()
        # update memory
        with torch.no_grad():
            all_size = k_all.shape[0]
            range_mat = torch.arange(all_size, dtype=torch.long).to(q.device)
            out_ids = torch.fmod(range_mat + self.index, self.queue_size)
            self.memory.index_copy_(0, out_ids, k_all)
            self.index = (self.index + all_size) % self.queue_size

        return out