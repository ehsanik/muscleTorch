import torch
import torch.nn as nn
import torch.optim as optim
from .basemodel import BaseModel
from .feature_learning import FeatureLearnerModule
from utils.net_util import input_embedding_net, combine_block_w_do
from training import metrics


class SceneClassModel(BaseModel):
    metric = [metrics.SceneMetric]

    def __init__(self, args):
        super(SceneClassModel, self).__init__(args)

        assert args.detach_level == 0

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

        self.pointwise_conv = combine_block_w_do(512, 64, args.dropout)


        scene_unembed_size = torch.Tensor([64 * 7 * 7, self.hidden_size, self.num_classes])
        self.scene_unembed = input_embedding_net(scene_unembed_size.long().tolist(), dropout=args.dropout)

        assert self.input_length == self.sequence_length == 1 == self.output_length

    def forward(self, input, target):
        input_images = input['rgb']
        batch_size, seq_len, _, _, _ = input_images.shape
        features = self.feature_extractor(input_images)
        intermediate_features = self.feature_extractor.intermediate_features
        spatial_features = intermediate_features[-1]
        spatial_features = self.pointwise_conv(spatial_features)
        spatial_features = spatial_features.view(batch_size, seq_len, 64, 7, 7)
        assert seq_len == 1
        spatial_features = spatial_features.view(batch_size * 1, 64 * 7 * 7)

        scene_class = self.scene_unembed(spatial_features)
        scene_actual_class = torch.argmax(scene_class, dim=-1)

        output = {
            'scene_class': scene_class,
            'scene_actual_class': scene_actual_class,
        }

        return output, target

    def loss(self, args):
        return self.loss_function(args)

    def optimizer(self):
        return optim.Adam(self.parameters(), lr=self.base_lr)
