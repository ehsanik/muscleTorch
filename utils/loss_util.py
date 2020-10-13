import torch
import torch.nn as nn

__all__ = [
    'GazeRegressionLoss',
    'MoveLabelLoss',
    'ActionRecognitionLoss',
    'DepthLoss',
    'SceneClassLoss',
    'ReconstructionLoss',
    'AEGazeIMULoss',
    'ClassificationLoss',
    'NCESoftmaxLoss',
    'MocoGazeLoss',
    'MocoImuLoss',
    'MoCoGazeIMULoss',
    'VindClassLoss',
    'WalkableLoss',
]

variables = locals()


class BasicLossFunction(nn.Module):
    def __init__(self):
        super(BasicLossFunction, self).__init__()

    @property
    def local_loss_dict(self):

        module_attributes = self._modules.keys()
        result = self._local_loss_dict
        for mod in module_attributes:
            attr = self.__getattr__(mod)
            if issubclass(type(attr), BasicLossFunction):
                result.update(attr.local_loss_dict)
        return result

    def calc_and_update_total_loss(self, loss_dict, batch_size):
        total = 0
        for k in loss_dict:
            self._local_loss_dict[k] = (loss_dict[k], batch_size)
            total += loss_dict[k] * self.weights_for_each_loss[k]
        return total


class ReconstructionLoss(BasicLossFunction):
    def __init__(self, args):
        super(ReconstructionLoss, self).__init__()
        self._local_loss_dict = {
            'reconstructed_rgb': None,
        }

        self.weights_for_each_loss = {
            'reconstructed_rgb': 1,
        }
        self.mse_loss = nn.MSELoss()

    def forward(self, output, target):
        output_reconstructed_rgb = output['reconstructed_rgb']
        target_reconstructed_rgb = target['reconstructed_rgb']
        assert output_reconstructed_rgb.shape == target_reconstructed_rgb.shape
        loss = self.mse_loss(output_reconstructed_rgb, target_reconstructed_rgb)
        batch_size = output_reconstructed_rgb.shape[0]

        loss_dict = {
            'reconstructed_rgb': loss,
        }

        total_loss = self.calc_and_update_total_loss(loss_dict, batch_size)

        return total_loss



class MoCoGazeIMULoss(BasicLossFunction):
    def __init__(self, args):
        super(MoCoGazeIMULoss, self).__init__()
        self._local_loss_dict = {
            'moco_loss': None,
            'gaze_loss': None,
            'imu_loss': None,
        }

        self.weights_for_each_loss = {
            'moco_loss': 0.09,
            'gaze_loss': 0.90,
            'imu_loss': 0.009,
        }
        self.moco_loss = NCESoftmaxLoss(args)
        self.gaze_loss = GazeRegressionLoss(args)
        self.imu_loss = MoveLabelLoss(args)

    def forward(self, output, target):
        moco_loss = self.moco_loss(output, target)
        gaze_loss = self.gaze_loss(output, target)
        imu_loss = self.imu_loss(output, target)

        loss_dict = {
            'moco_loss': moco_loss,
            'gaze_loss': gaze_loss,
            'imu_loss': imu_loss,
        }

        batch_size = 1

        total_loss = self.calc_and_update_total_loss(loss_dict, batch_size)

        return total_loss

class MocoImuLoss(MoCoGazeIMULoss):
    def __init__(self, args):
        super(MocoImuLoss, self).__init__(args)
        self.weights_for_each_loss = {
            'moco_loss': 0.95,
            'imu_loss': 0.5,
            'gaze_loss': 0.0,
        }


class MocoGazeLoss(MoCoGazeIMULoss):
    def __init__(self, args):
        super(MocoGazeLoss, self).__init__(args)
        self.weights_for_each_loss = {
            'moco_loss': 0.10,
            'gaze_loss': 0.90,
            'imu_loss': 0.0,

        }

class ClassificationLoss(BasicLossFunction):
    def __init__(self, args):
        super(ClassificationLoss, self).__init__()
        self._local_loss_dict = {
            'classification_loss': None,
        }

        self.weights_for_each_loss = {
            'classification_loss': 1,
        }
        self.loss = nn.CrossEntropyLoss()

    def forward(self, output, target):
        output_class = output['class_label']
        target_class = target['class_label']

        loss = self.loss(output_class, target_class)

        loss_dict = {
            'classification_loss': loss,
        }

        batch_size = output_class.shape[0]

        total_loss = self.calc_and_update_total_loss(loss_dict, batch_size)

        return total_loss


class AEGazeIMULoss(BasicLossFunction):
    def __init__(self, args):
        super(AEGazeIMULoss, self).__init__()
        self._local_loss_dict = {
            'gaze_loss': None,
            'imu_loss': None,
            'reconstruction_loss': None,
        }

        self.weights_for_each_loss = {
            'gaze_loss': 0.48, #This is around 0.01
            'imu_loss': 0.04, #This is around 0.45
            'reconstruction_loss': 0.48, #This is around 0.05
        }
        self.gaze_loss_class = GazeRegressionLoss(args)
        self.imu_loss_class = MoveLabelLoss(args)
        self.reconstruction_loss = ReconstructionLoss(args)

    def forward(self, output, target):
        gaze_loss = self.gaze_loss_class(output, target)
        imu_loss = self.imu_loss_class(output, target)
        reconstruction_loss = self.reconstruction_loss(output, target)

        loss_dict = {
            'gaze_loss': gaze_loss,
            'imu_loss': imu_loss,
            'reconstruction_loss': reconstruction_loss,
        }

        batch_size = 1 #This is not absolutely correct but it is just an approx and only changes the plotting

        total_loss = self.calc_and_update_total_loss(loss_dict, batch_size)

        return total_loss


class DepthLoss(BasicLossFunction):
    def __init__(self, args):
        super(DepthLoss, self).__init__()
        self._local_loss_dict = {
            'depth': None,
        }

        self.weights_for_each_loss = {
            'depth': 1,
        }
        self.loss = nn.SmoothL1Loss()

    def forward(self, output, target):
        output_depth = output['depth']
        target_depth = target['depth']
        assert output_depth.shape == target_depth.shape


        loss = self.loss(output_depth, target_depth)

        loss_dict = {
            'depth': loss,
        }

        batch_size = output_depth.shape[0]

        total_loss = self.calc_and_update_total_loss(loss_dict, batch_size)

        return total_loss

class WalkableLoss(BasicLossFunction):
    def __init__(self, args, weights):
        super(WalkableLoss, self).__init__()
        self._local_loss_dict = {
            'walk': None,
        }

        self.weights_for_each_loss = {
            'walk': 1,
        }
        weights = 1 / weights
        weights[1] *= 2
        if args.gpu_ids != -1:
            weights = weights.cuda()
        self.bceloss = nn.CrossEntropyLoss(weights)


    def forward(self, output, target):
        output_walk = output['walk'].squeeze(1)
        target_walk = target['walk'].squeeze(1)

        target_walk = target_walk.permute(0, 2, 3, 1)
        output_walk = output_walk.permute(0, 2, 3, 1)

        flat_output = output_walk.contiguous().view(-1, 2)
        flat_target = target_walk.contiguous().view(-1)


        loss = self.bceloss(flat_output, flat_target)

        loss_dict = {
            'walk': loss,
        }

        batch_size = output_walk.shape[0]

        total_loss = self.calc_and_update_total_loss(loss_dict, batch_size)

        return total_loss


class ActionRecognitionLoss(BasicLossFunction):

    def __init__(self, args):
        super(ActionRecognitionLoss, self).__init__()
        self._local_loss_dict = {
            'action_recognition': None,
        }
        self.weights_for_each_loss = {
            'action_recognition': 1,
        }
        self.loss = nn.CrossEntropyLoss(weight=args.dataset.CLASS_WEIGHTS)


    def forward(self, output, target):
        output_action = output['verb_class']
        target_action = target['verb_class']
        loss = self.loss(output_action, target_action)
        loss_dict = {
            'action_recognition': loss,
        }
        batch_size = target_action.shape[0]
        total_loss = self.calc_and_update_total_loss(loss_dict, batch_size)
        return total_loss

class NCESoftmaxLoss(BasicLossFunction):
    """Softmax cross-entropy loss (a.k.a., info-NCE loss in CPC paper)
    https://github.com/bl0/moco/blob/master/lib/NCE/NCECriterion.py
    """
    def __init__(self, args):
        super(NCESoftmaxLoss, self).__init__()
        self.criterion = nn.CrossEntropyLoss()
        self._local_loss_dict = {
            'nce_softmax': None,
        }
        self.weights_for_each_loss = {
            'nce_softmax': 1,
        }

    def forward(self, output, target):
        output_labels = output['moco_label']
        label = torch.zeros([output_labels.shape[0]]).long().to(output_labels.device)
        loss = self.criterion(output_labels, label)
        loss_dict = {
            'nce_softmax': loss,
        }
        batch_size = output_labels.shape[0]
        total_loss = self.calc_and_update_total_loss(loss_dict, batch_size)

        return total_loss

class SceneClassLoss(BasicLossFunction):

    def __init__(self, args):
        super(SceneClassLoss, self).__init__()
        self._local_loss_dict = {
            'scene_class': None,
        }
        self.weights_for_each_loss = {
            'scene_class': 1,
        }
        self.loss = nn.CrossEntropyLoss()

    def forward(self, output, target):
        output_scene_label = output['scene_class']
        target_scene_label = target['scene_class']
        loss = self.loss(output_scene_label, target_scene_label)
        loss_dict = {
            'scene_class': loss,
        }
        batch_size = target_scene_label.shape[0]
        total_loss = self.calc_and_update_total_loss(loss_dict, batch_size)
        return total_loss


class VindClassLoss(BasicLossFunction):

    def __init__(self, args):
        super(VindClassLoss, self).__init__()
        self._local_loss_dict = {
            'vind_class': None,
        }
        self.weights_for_each_loss = {
            'vind_class': 1,
        }
        self.loss = nn.CrossEntropyLoss()

    def forward(self, output, target):
        output_vind_label = output['vind_class']
        target_vind_label = target['vind_class']
        loss = self.loss(output_vind_label, target_vind_label)
        loss_dict = {
            'vind_class': loss,
        }
        batch_size = target_vind_label.shape[0]
        total_loss = self.calc_and_update_total_loss(loss_dict, batch_size)
        return total_loss

class GazeRegressionLoss(BasicLossFunction):
    def __init__(self, args):
        super(GazeRegressionLoss, self).__init__()
        self._local_loss_dict = {
            'gaze': None,
        }

        self.weights_for_each_loss = {
            'gaze': 1,
        }
        self.smooth_loss = nn.SmoothL1Loss()

    def forward(self, output, target):
        output_gaze = output['gaze_points']
        target_gaze = target['gaze_points']
        assert output_gaze.shape == target_gaze.shape

        mask = torch.any(target_gaze == -1, dim=-1)
        not_masked = ~mask
        batch_size = (not_masked.sum()).item()
        if batch_size == 0:
            loss = torch.tensor(0., requires_grad=True, device=output_gaze.device)

        else:
            loss = self.smooth_loss(output_gaze[not_masked], target_gaze[not_masked])

        loss_dict = {
            'gaze': loss,
        }

        total_loss = self.calc_and_update_total_loss(loss_dict, batch_size)

        return total_loss


class MoveLabelLoss(BasicLossFunction):
    def __init__(self, args):
        super(MoveLabelLoss, self).__init__()
        self._local_loss_dict = {
            'move_label': None,
        }

        self.weights_for_each_loss = {
            'move_label': 1,
        }
        self.bce_loss = nn.BCEWithLogitsLoss()

    def forward(self, output, target):
        output_move_label = output['move_label']
        target_move_label = target['move_label']
        assert output_move_label.shape == target_move_label.shape

        mask = (target_move_label == -1)
        not_masked = ~mask
        batch_size = (not_masked.sum()).item()
        if batch_size == 0:
            loss = torch.tensor(0., requires_grad=True, device=output_move_label.device)

        else:
            loss = self.bce_loss(output_move_label[not_masked], target_move_label[not_masked])

        loss_dict = {
            'move_label': loss,
        }

        total_loss = self.calc_and_update_total_loss(loss_dict, batch_size)

        return total_loss
