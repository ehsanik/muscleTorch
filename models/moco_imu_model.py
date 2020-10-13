import torch
import torch.nn as nn
import torch.optim as optim
from .basemodel import BaseModel
from .feature_learning import FeatureLearnerModule
from utils.net_util import input_embedding_net, combine_block_w_do
from training import metrics
import math
from utils.contrastive_utils import moment_update, MemoryMoCo, get_moco_labels


class MoCoIMUModel(BaseModel):
    metric = [
              metrics.MoveDetectorMetric]

    def __init__(self, args):
        super(MoCoIMUModel, self).__init__(args)

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
        self.moco_feature_extractor = FeatureLearnerModule(args)


        self.feature_linear = nn.Linear(512, 128)
        self.moco_feature_linear = nn.Linear(512, 128)

        moment_update(self.feature_extractor, self.moco_feature_extractor, 0.0) #Copy feature extractor to moco_feature
        moment_update(self.feature_linear, self.moco_feature_linear, 0.0) #Copy feature extractor to moco_feature

        self.alpha = 0.999
        queue_size = 16384

        assert self.input_length == self.sequence_length == self.output_length

        imu_unembed_size = torch.Tensor([self.hidden_size, 100, self.num_imus * 1])
        self.imu_unembed = input_embedding_net(imu_unembed_size.long().tolist(), dropout=args.dropout)

        self.pointwise_conv = combine_block_w_do(512, 64, args.dropout)

        self.imu_embed_lstm = nn.LSTM(64 * 7 * 7, self.hidden_size, batch_first=True, num_layers=3)

        self.imu_decoder_lstm = nn.LSTM(64 * 7 * 7, self.hidden_size, batch_first=True, num_layers=3)


        self.contrast = MemoryMoCo(128, queue_size, 0.07)
        if args.gpu_ids != -1:
            self.contrast = self.contrast.cuda()




    def forward(self, input, target):
        first_augm = input['first_augm'].squeeze(1)
        second_augm = input['second_augm'].squeeze(1)

        self.imu_embed_lstm.flatten_parameters()
        self.imu_decoder_lstm.flatten_parameters()

        out = get_moco_labels(self, first_augm, second_augm)

        input_images = input['rgb']
        batch_size, seq_len, _, _, _ = input_images.shape
        _ = self.feature_extractor(input_images)
        intermediate_features = self.feature_extractor.intermediate_features
        spatial_features = intermediate_features[-1]
        spatial_features = self.pointwise_conv(spatial_features)
        spatial_features = spatial_features.view(batch_size, seq_len, 64, 7, 7)
        spatial_features = spatial_features.view(batch_size, seq_len, 64 * 7 * 7)


        output, (hidden, cell) = self.imu_embed_lstm(spatial_features) #To see the whole sequence and embed it into a hidden vector

        output, (hidden, cell) = self.imu_decoder_lstm(spatial_features, (hidden, cell))

        predicted_imu = self.imu_unembed(output)


        output = {
            'moco_label': out,
            'actual_moco_label': torch.argmax(out, dim=-1),
            'move_label': predicted_imu,
            'cleaned_move_label': (torch.sigmoid(predicted_imu) > 0.5).float().detach(),

        }
        target['rgb'] = input['rgb']

        return output, target

    def loss(self, args):
        return self.loss_function(args)

    def optimizer(self):
        return optim.Adam(self.parameters(), lr=self.base_lr)