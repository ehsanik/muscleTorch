

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from .basemodel import BaseModel
from .feature_learning import FeatureLearnerModule
from utils.net_util import input_embedding_net, combine_block_w_do, upshufflenorelu, upshuffle
from training import metrics

class ComplexAEGazeImuModel(BaseModel):

    metric = [
        metrics.MoveDetectorMetric,
    ]

    def __init__(self, args):
        super(ComplexAEGazeImuModel, self).__init__(args)

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

        self.pointwise_conv = combine_block_w_do(512, 64, dropout=0) #Very important
        self.imu_pointwise_conv = combine_block_w_do(512, 64, args.dropout)
        self.gaze_pointwise_conv = combine_block_w_do(512, 64, args.dropout)

        self.reconst_resolution = args.reconst_resolution
        assert self.reconst_resolution == 56
        self.feature_sizes = [7, 14, 28, 56, 56, self.reconst_resolution]
        self.upscale_factor = [int(self.feature_sizes[i + 1]/ self.feature_sizes[i]) for i in range(len(self.feature_sizes) - 1)]


        self.up1 = upshuffle(64, 256, self.upscale_factor[0], kernel_size=3, stride=1, padding=1)
        self.up2 = upshuffle(256, 128, self.upscale_factor[1], kernel_size=3, stride=1, padding=1)
        self.up3 = upshuffle(128, 64, self.upscale_factor[2], kernel_size=3, stride=1, padding=1)
        self.up4 = upshuffle(64, 64, self.upscale_factor[3], kernel_size=3, stride=1, padding=1)
        self.up5 = upshufflenorelu(64, 3, self.upscale_factor[4])

        gaze_unembed_size = torch.Tensor([self.hidden_size, 100, 2])
        self.gaze_unembed = input_embedding_net(gaze_unembed_size.long().tolist(), dropout=args.dropout)

        imu_unembed_size = torch.Tensor([self.hidden_size, 100, self.num_imus * 1])
        self.imu_unembed = input_embedding_net(imu_unembed_size.long().tolist(), dropout=args.dropout)

        self.imu_embed_lstm = nn.LSTM(64 * 7 * 7, self.hidden_size, batch_first=True, num_layers=3)
        self.gaze_embed_lstm = nn.LSTM(64 * 7 * 7, self.hidden_size, batch_first=True, num_layers=3)

        self.gaze_decoder_lstm = nn.LSTM(64 * 7 * 7, self.hidden_size, batch_first=True, num_layers=3)
        self.imu_decoder_lstm = nn.LSTM(64 * 7 * 7, self.hidden_size, batch_first=True, num_layers=3)

        assert self.input_length == self.sequence_length == self.sequence_length

    def forward(self, input, target):
        input_images = input['rgb']
        batch_size, seq_len, c, w, h = input_images.shape
        features = self.feature_extractor(input_images)
        intermediate_features = self.feature_extractor.intermediate_features
        spatial_features = intermediate_features[-1]

        spatial_features = self.pointwise_conv(spatial_features)

        reconstructed_image = self.up1(spatial_features)
        reconstructed_image = self.up2(reconstructed_image)
        reconstructed_image = self.up3(reconstructed_image)
        reconstructed_image = self.up4(reconstructed_image)
        reconstructed_image = self.up5(reconstructed_image)
        reconstructed_image = reconstructed_image.view(batch_size, seq_len, 3, self.reconst_resolution, self.reconst_resolution)

        imu_spatial_features = intermediate_features[-1]
        imu_spatial_features = self.imu_pointwise_conv(imu_spatial_features)
        reshaped_imu_spatial_features = imu_spatial_features.view(batch_size, seq_len, 64, 7, 7)
        reshaped_imu_spatial_features = reshaped_imu_spatial_features.view(batch_size, seq_len, 64 * 7 * 7)
        output_imu, (hidden_imu, cell_imu) = self.imu_embed_lstm(reshaped_imu_spatial_features) #To see the whole sequence and embed it into a hidden vector
        output_imu, (hidden_imu, cell_imu) = self.imu_decoder_lstm(reshaped_imu_spatial_features, (hidden_imu, cell_imu))

        gaze_spatial_features = intermediate_features[-1]
        gaze_spatial_features = self.gaze_pointwise_conv(gaze_spatial_features)
        reshaped_gaze_spatial_features = gaze_spatial_features.view(batch_size, seq_len, 64, 7, 7)
        reshaped_gaze_spatial_features = reshaped_gaze_spatial_features.view(batch_size, seq_len, 64 * 7 * 7)
        output_gaze, (hidden_gaze, cell_gaze) = self.gaze_embed_lstm(reshaped_gaze_spatial_features) #To see the whole sequence and embed it into a hidden vector
        output_gaze, (hidden_gaze, cell_gaze) = self.gaze_decoder_lstm(reshaped_gaze_spatial_features, (hidden_gaze, cell_gaze))


        predicted_gaze = self.gaze_unembed(output_gaze)
        predicted_imu = self.imu_unembed(output_imu)


        output = {
            'reconstructed_rgb': reconstructed_image,
            'gaze_points': predicted_gaze,
            'move_label': predicted_imu,
            'cleaned_move_label': (torch.sigmoid(predicted_imu) > 0.5).float().detach(),
        }

        flatten_images = input_images.view(batch_size * seq_len, c, w, h)
        reshaped_images = F.interpolate(flatten_images, (self.reconst_resolution, self.reconst_resolution))
        target['reconstructed_rgb'] = reshaped_images.view(batch_size, seq_len, c, self.reconst_resolution, self.reconst_resolution)
        target['rgb'] = input_images
        return output, target

    def loss(self, args):
        return self.loss_function(args)

    def optimizer(self):
        return optim.Adam(self.parameters(), lr=self.base_lr)
