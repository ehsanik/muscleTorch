### This is borrowed from
### https://raw.githubusercontent.com/xanderchf/MonoDepth-FPN-PyTorch/master/model_fpn.py



import torch
from .basemodel import BaseModel
from .feature_learning import FeatureLearnerModule
from utils.net_util import _upsample_add, upshuffle, upshufflenorelu, combine_block_w_do
from training import metrics

class DepthEstimationModel(BaseModel):
    metric = [
        metrics.DepthMetric,
        metrics.DepthMSE,
        metrics.DepthLogMSE,
        metrics.DepthLog,
    ]
    def __init__(self, args):
        super(DepthEstimationModel, self).__init__(args)
        assert args.dropout == 0

        self.loss_function = args.loss

        self.fixed_feature_weights = args.fixed_feature_weights

        self.pointwise_conv = combine_block_w_do(512, 64, args.dropout)

        self.depth_up1 = upshuffle(64, 256, 2, kernel_size=3, stride=1, padding=1)
        self.depth_up2 = upshuffle(256, 128, 2, kernel_size=3, stride=1, padding=1)
        self.depth_up3 = upshuffle(128, 64, 2, kernel_size=3, stride=1, padding=1)
        self.depth_up4 = upshuffle(64, 64, 2, kernel_size=3, stride=1, padding=1)
        self.depth_up5 = upshufflenorelu(64, 1, 2)

        self.feature_extractor = FeatureLearnerModule(args)

    def loss(self, args):
        return self.loss_function(args)

    def forward(self, input, target):
        images = input['rgb']
        batch, seqlen, c ,H,W = images.size()


        features = self.feature_extractor(images)
        intermediate_features = self.feature_extractor.intermediate_features
        spatial_features = intermediate_features[-1]
        spatial_features = self.pointwise_conv(spatial_features)

        assert seqlen == 1
        spatial_features = spatial_features.view(batch * 1, 64, 7, 7)


        c1, c2, c3, c4, _ = intermediate_features
        d5 = self.depth_up1(spatial_features)
        d5_ = _upsample_add(d5, c4)
        d4 = self.depth_up2(d5_)
        d4_ = _upsample_add(d4, c3)
        d3 = self.depth_up3(d4_)
        d3_ = _upsample_add(d3, c2)
        d2 = self.depth_up4(d3_)
        d2_ = _upsample_add(d2, c1)
        depth = self.depth_up5(d2_)


        output = {
            'depth': depth.unsqueeze(1), #Put sequence length back
        }

        return output, target


    def optimizer(self):
        return torch.optim.Adam(self.parameters(), lr=self.base_lr)
