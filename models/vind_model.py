import torch
import torch.nn as nn
import torch.optim as optim
from .basemodel import BaseModel
from .feature_learning import FeatureLearnerModule
from utils.net_util import input_embedding_net, combine_block_w_do
from training import metrics


class VindModel(BaseModel):
    metric = [metrics.VindMetric]

    def __init__(self, args):
        super(VindModel, self).__init__(args)

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


        self.pointwise_conv = combine_block_w_do(512, 64, args.dropout)

        self.conv1 = nn.Conv2d(1,64,7,stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool1 = nn.MaxPool2d(3,stride=2,padding=1)
        self.conv2 = nn.Conv2d(64,256,7,stride=2, padding=3)
        self.bn2 = nn.BatchNorm2d(256)
        self.maxpool2 = nn.MaxPool2d(3,stride=2,padding=1)
        self.conv3 = nn.Conv2d(256,64,7,stride=2, padding=3)
        self.bn3 = nn.BatchNorm2d(64)


        vind_unembed_size = torch.Tensor([64 * 7 * 7 * 2, 64 * 7 * 7, self.hidden_size, self.num_classes])
        self.vind_unembed = input_embedding_net(vind_unembed_size.long().tolist(), dropout=args.dropout)

        assert self.input_length == self.sequence_length == 1 == self.output_length

    def forward(self, input, target):
        input_images = input['rgb']
        obj_mask = target['objmask'].squeeze(1) #because no seq len

        x = self.relu(self.bn1(self.conv1(obj_mask)))
        x = self.maxpool1(x)
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.maxpool2(x)
        x = self.bn3(self.conv3(x))


        batch_size, seq_len, _, _, _ = input_images.shape
        features = self.feature_extractor(input_images)
        intermediate_features = self.feature_extractor.intermediate_features
        spatial_features = intermediate_features[-1]
        spatial_features = self.pointwise_conv(spatial_features)
        spatial_features = spatial_features.view(batch_size, seq_len, 64, 7, 7)
        assert seq_len == 1
        spatial_features = spatial_features.view(batch_size, 64 * 7 * 7)
        x = x.view(batch_size, 64 * 7 * 7)
        spatial_features = torch.cat([spatial_features, x], dim=-1)

        vind_class = self.vind_unembed(spatial_features)
        vind_actual_class = torch.argmax(vind_class, dim=-1)

        output = {
            'vind_class': vind_class,
            'vind_actual_class': vind_actual_class,
        }

        target['combined_mask']= target['objmask']

        return output, target

    def loss(self, args):
        return self.loss_function(args)

    def optimizer(self):
        return optim.Adam(self.parameters(), lr=self.base_lr)
