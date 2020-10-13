

import torch.nn as nn
import torch.optim as optim
from .basemodel import BaseModel
from .feature_learning import FeatureLearnerModule
from utils.net_util import input_embedding_net, combine_block_w_do, upshufflenorelu, upshuffle

class AutoEncoderModel(BaseModel):

    metric = []

    def __init__(self, args):
        super(AutoEncoderModel, self).__init__(args)

        self.image_size = args.image_size
        self.imus = args.imus

        self.input_length = args.input_length
        self.output_length = args.output_length
        self.sequence_length = args.sequence_length
        self.num_classes = args.num_classes
        self.gpu_ids = args.gpu_ids

        self.base_lr = args.base_lr
        self.image_feature = args.image_feature
        self.hidden_size = args.hidden_size
        self.imu_embedding_size = 30

        self.loss_function = args.loss

        self.relu = nn.LeakyReLU()
        self.num_imus = args.num_imus
        self.feature_extractor = FeatureLearnerModule(args)

        self.pointwise_conv = combine_block_w_do(512, 64, args.dropout)

        self.reconst_resolution = args.reconst_resolution
        assert self.reconst_resolution == 224
        self.feature_sizes = [7, 14, 28, 56, 112, self.reconst_resolution]
        self.upscale_factor = [int(self.feature_sizes[i + 1]/ self.feature_sizes[i]) for i in range(len(self.feature_sizes) - 1)]


        self.up1 = upshuffle(64, 256, self.upscale_factor[0], kernel_size=3, stride=1, padding=1)
        self.up2 = upshuffle(256, 128, self.upscale_factor[1], kernel_size=3, stride=1, padding=1)
        self.up3 = upshuffle(128, 64, self.upscale_factor[2], kernel_size=3, stride=1, padding=1)
        self.up4 = upshuffle(64, 64, self.upscale_factor[3], kernel_size=3, stride=1, padding=1)
        self.up5 = upshufflenorelu(64, 3, self.upscale_factor[4])


        assert self.input_length == self.sequence_length and self.input_length == self.output_length and self.sequence_length == 1


    def forward(self, input, target):
        input_images = input['rgb']
        batch_size, seq_len, _, _, _ = input_images.shape
        features = self.feature_extractor(input_images)
        intermediate_features = self.feature_extractor.intermediate_features
        spatial_features = intermediate_features[-1]


        spatial_features = self.pointwise_conv(spatial_features)

        spatial_features = self.up1(spatial_features)
        spatial_features = self.up2(spatial_features)
        spatial_features = self.up3(spatial_features)
        spatial_features = self.up4(spatial_features)
        spatial_features = self.up5(spatial_features)

        output = {
            'reconstructed_rgb': spatial_features.unsqueeze(1), # to put back the sequence length
        }
        target['reconstructed_rgb'] = input['rgb']
        return output, target

    def loss(self, args):
        return self.loss_function(args)

    def optimizer(self):
        return optim.Adam(self.parameters(), lr=self.base_lr)
