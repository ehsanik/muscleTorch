#!/bin/zsh

# For using the pretrained models to reproduce the numbers in the paper, Add the following to the commands
# --use_test_for_val test --reload PATHTOWEIGHTDIR/trained_end_tasks/TASKNAME/MODELNAME.pytar

# Scene Classification
python3 main.py --title scene --use_test_for_val --gpu-ids 0 --arch SceneClassModel --input_length 1 --sequence_length 1 --output_length 1 --dataset SunDataset --num_classes 397 --loss SceneClassLoss --step_size 50 --save_frequency 1 --imu_feature_type scene_class --dropout .5 --data PATHTODATA --detach_level 0

# Action Recognition
python3 main.py --title action --use_test_for_val --gpu-ids 0 --arch ActionReprModel --input_length 5 --sequence_length 5 --output_length 1 --dataset EpicStateChangingDataset --num_classes 9 --loss ActionRecognitionLoss --step_size 50 --save_frequency 1 --input_feature_type verb_class --base-lr 0.001 --dropout .5 --data PATHTODATA --skipping_frames 6 --detach_level 0

# Depth Estimation
python3 main.py --title depth --use_test_for_val --gpu-ids 0 --arch DepthEstimationModel --input_length 1 --sequence_length 1 --output_length 1 --dataset NYUDepthDataset --loss DepthLoss --step_size 500 --save_frequency 100 --input_feature_type depth --base-lr 0.001 --data PATHTODATA --detach_level 0

# Walkable Surface Estimation
python3 main.py --title walk --use_test_for_val --gpu-ids 0 --arch WalkableModel --input_length 1 --sequence_length 1 --output_length 1 --dataset NYUWalkDataset --loss WalkableLoss --step_size 500 --save_frequency 100 --base-lr 0.001 --data PATHTODATA --detach_level 0

#Dynamic Prediction
python3 main.py --title dynamic --use_test_for_val --gpu-ids 0 --arch VindModel --input_length 1 --sequence_length 1 --output_length 1 --dataset VindDataset --num_classes 66 --loss VindClassLoss --step_size 50 --save_frequency 1 --dropout .5 --data PATHTODATA --skipping_frames 1 --detach_level 0

