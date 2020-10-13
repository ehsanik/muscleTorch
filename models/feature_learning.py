"""
=================
This is the basic model containing place holder/implementation for necessary functions.

All the models in this project inherit from this class
=================
"""

import torch.nn as nn
from torchvision.models import resnet18 as torchvision_resnet18


class FeatureLearnerModule(nn.Module):

    def __init__(self, args):
        super(FeatureLearnerModule, self).__init__()
        self.base_lr = args.base_lr
        self.lrm = args.lrm
        self.read_features = args.read_features
        self.number_of_trained_resnet_blocks = args.number_of_trained_resnet_blocks
        resnet_model = torchvision_resnet18(pretrained=args.pretrain)
        del resnet_model.fc
        self.resnet = resnet_model
        self.detach_level = args.detach_level

        self.fixed_feature_weights = args.fixed_feature_weights
        self.intermediate_features = None

        imagenet_feature_testing = args.pretrain and self.fixed_feature_weights and 'imagenet' in args.title and self.detach_level == 0
        imagenet_feature_training = args.pretrain and 'imagenet_train_all_the_way' in args.title

        assert imagenet_feature_testing or (not args.pretrain) or imagenet_feature_training

            
    def resnet_features(self, x):

        result = []

        if not self.read_features or self.number_of_trained_resnet_blocks >= 5 :
            x = self.resnet.conv1(x)
            x = self.resnet.bn1(x)
            x = self.resnet.relu(x)
            if self.detach_level <= 5:
                x = x.detach()
            result.append(x)
            x = self.resnet.maxpool(x)

            

        if not self.read_features or self.number_of_trained_resnet_blocks >= 4:
            x = self.resnet.layer1(x)
            if self.detach_level <= 4:
                x = x.detach()
            result.append(x)

        if not self.read_features or self.number_of_trained_resnet_blocks >= 3:
            x = self.resnet.layer2(x)
            if self.detach_level <= 3:
                x = x.detach()
            result.append(x)
        

        if not self.read_features or self.number_of_trained_resnet_blocks >= 2:
            x = self.resnet.layer3(x)
            if self.detach_level <= 2:
                x = x.detach()
            result.append(x)
        

        if not self.read_features or self.number_of_trained_resnet_blocks >= 1:
            x = self.resnet.layer4[0](x)
            x = self.resnet.layer4[1](x)
            if self.detach_level <= 1:
                x = x.detach()
            result.append(x)
            x = self.resnet.avgpool(x)
            
        

        if not self.read_features or self.number_of_trained_resnet_blocks >= 0:

            x = x.view(x.shape[0], 512)

        self.intermediate_features = result
        if self.detach_level <= 0:
            x = x.detach()

        return x

    def get_feature(self, images):
        shape = list(images.shape)
        batchsize = shape[0]
        sequence_length = shape[1]
        images = images.contiguous().view([batchsize * sequence_length] + shape[2:]) 
        features = self.resnet_features(images)
        return features.view(batchsize, sequence_length, 512)
        
    
    def forward(self, images):
        self.intermediate_features = None #just a sanity check that they are reinitialized each time
        return self.get_feature(images)
