from .basemodel import BaseModel
from .action_pred_model import ActionReprModel
from .auto_encoder_model import AutoEncoderModel
from .current_imu_from_image_gaze import CurrentMoveFromGazeImgModel
from .current_imu_no_gaze_model import CurrentMoveWithoutGazeImgModel
from .current_move_label_predict import CurrentMoveLabelModel
from .depth_estimation_model import DepthEstimationModel
from .scene_class_model import SceneClassModel
from .complex_ae_gaze_imu_model import ComplexAEGazeImuModel
from .vind_model import VindModel
from .walkable_model import WalkableModel
from .moco_gaze_model import MoCoGazeModel
from .moco_imu_model import MoCoIMUModel
from .moco_gaze_imu_model import MoCoGazeIMUModel


__all__ = [
    'VindModel',
    'WalkableModel',
    'AutoEncoderModel',
    'ComplexAEGazeImuModel',
    'MoCoGazeIMUModel',
    'ActionReprModel',
    'DepthEstimationModel',
    'SceneClassModel',
    'CurrentMoveWithoutGazeImgModel',
    'CurrentMoveLabelModel',
    'CurrentMoveFromGazeImgModel',
]

# All models should inherit from BaseModel
variables = locals()
for model in __all__:
    assert issubclass(variables[model], BaseModel),\
             "All model classes should inherit from %s.%s. Model %s does not."\
                % (BaseModel.__module__, BaseModel.__name__, model)
