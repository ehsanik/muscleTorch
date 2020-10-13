

import torch
import torch.nn as nn
import torch.optim as optim
from .basemodel import BaseModel
from .feature_learning import FeatureLearnerModule
from utils.net_util import input_embedding_net, combine_block_w_do
from training import metrics

class CurrentMoveFromGazeImgModel(BaseModel):

    metric = [metrics.MoveDetectorMetric]

    def __init__(self, args):
        super(CurrentMoveFromGazeImgModel, self).__init__(args)

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
        self.embedding_input = nn.Linear(args.image_feature, args.hidden_size)

        self.input_feature_type = args.input_feature_type[0]

        imu_moves_unembed_size = torch.Tensor([self.hidden_size, 100, self.num_imus * 1])
        self.imu_moves_unembed = input_embedding_net(imu_moves_unembed_size.long().tolist(), dropout=args.dropout)

        gaze_embed_size = torch.Tensor([2, 100, self.hidden_size])
        self.gaze_embed = input_embedding_net(gaze_embed_size.long().tolist(), dropout=args.dropout)

        self.pointwise_conv = combine_block_w_do(512, 64, args.dropout)

        self.lstm = nn.LSTM(64 * 7 * 7 + 512, self.hidden_size, batch_first=True, num_layers=3)

        self.decoder_lstm = nn.LSTM(64 * 7 * 7 + 512, self.hidden_size, batch_first=True, num_layers=3)

        assert self.input_length == self.sequence_length and self.input_length == self.output_length


    def forward(self, input, target):
        input_images = input['rgb']
        input_gaze = target['gaze_points']
        batch_size, seq_len, _, _, _ = input_images.shape
        features = self.feature_extractor(input_images)
        intermediate_features = self.feature_extractor.intermediate_features
        spatial_features = intermediate_features[-1]
        spatial_features = self.pointwise_conv(spatial_features)
        spatial_features = spatial_features.view(batch_size, seq_len, 64, 7, 7)
        spatial_features = spatial_features.view(batch_size, seq_len, 64 * 7 * 7)

        embedded_gaze = self.gaze_embed(input_gaze)
        input_features = torch.cat([spatial_features, embedded_gaze], dim=-1)

        self.lstm.flatten_parameters()
        self.decoder_lstm.flatten_parameters()


        output, (hidden, cell) = self.lstm(input_features) #To see the whole sequence and embed it into a hidden vector


        output, (hidden, cell) = self.decoder_lstm(input_features, (hidden, cell))


        predicted_imu_moves = self.imu_moves_unembed(output)


        output = {
            'move_label': predicted_imu_moves,
            'cleaned_move_label': (torch.sigmoid(predicted_imu_moves) > 0.5).float().detach(),
        }

        target['rgb'] = input['rgb']
        target['input_gaze_points'] = input_gaze

        return output, target

    def loss(self, args):
        return self.loss_function(args)

    def optimizer(self):
        return optim.Adam(self.parameters(), lr=self.base_lr)
