#!/bin/zsh

# For training with Visual and Attention Objective
python3 main.py --gpu-ids 0 1 2 3 --arch MoCoGazeIMUModel --input_length 5 --sequence_length 5 --output_length 5 --dataset HumanContrastiveCombinedDataset --workers 20 --num_classes -1 --batch-size 256 --break-batch 1 --loss MocoGazeLoss --num_imus 6 --imu_names neck body llegu rlegu larmu rarmu --step_size 50 --save_frequency 1 --input_feature_type gaze_points move_label --base-lr 0.0005 --dropout 0.5 --data PATHTODATA/human_data --skipping_frames 1 --data_parallel

# Trained weights: trained_representations/ours_gaze.pytar

# For training with Visual and Movement Objective
python3 main.py --gpu-ids 0 1 2 3 --arch MoCoGazeIMUModel --input_length 5 --sequence_length 5 --output_length 5 --dataset HumanContrastiveCombinedDataset --workers 20 --num_classes -1 --batch-size 256 --break-batch 1 --loss MocoImuLoss --num_imus 6 --imu_names neck body llegu rlegu larmu rarmu --step_size 50 --save_frequency 1 --input_feature_type gaze_points move_label --base-lr 0.0005 --dropout 0.5 --data PATHTODATA/human_data --skipping_frames 1 --data_parallel

# Trained weights: trained_representations/ours_imu.pytar

# For training with Visual, Attention and Movement objectives
python3 main.py --gpu-ids 0 1 2 3 --arch MoCoGazeIMUModel --input_length 5 --sequence_length 5 --output_length 5 --dataset HumanContrastiveCombinedDataset --workers 20 --num_classes -1 --batch-size 256 --break-batch 1 --loss MoCoGazeIMULoss --num_imus 6 --imu_names neck body llegu rlegu larmu rarmu --step_size 50 --save_frequency 1 --input_feature_type gaze_points move_label --base-lr 0.0005 --dropout 0.5 --data PATHTODATA/human_data --skipping_frames 1 --data_parallel

# Trained weights: trained_representations/ours_gaze_imu.pytar

# MoCo baseline trained on our data: trained_representations/moco_on_ours.pytar


#For training our auto encoder ablations
python3 main.py --title ae_gaze_imu --use_test_for_val --gpu-ids 0 --arch ComplexAEGazeImuModel --input_length 5 --sequence_length 5 --output_length 5 --dataset HumanH5pyDataset --num_classes -1 --loss AEGazeIMULoss --num_imus 6 --imu_names neck body llegu rlegu larmu rarmu --step_size 50 --save_frequency 1 --input_feature_type gaze_points move_label --base-lr 0.0005 --dropout 0.5 --data PATHTODATA/HumanDataset --reconst_resolution 56
python3 main.py --title ae_baseline --use_test_for_val --gpu-ids 0 --arch AutoEncoderModel --input_length 1 --sequence_length 1 --output_length 1 --dataset HumanH5pyDataset --loss ReconstructionLoss --step_size 50 --save_frequency 1 --base-lr 0.001 --data PATHTODATA/HumanDataset


#For training separate joint experiments
#python3 main.py --title no_arm --use_test_for_val --gpu-ids 0 --arch MoCoGazeIMUModel --input_length 5 --sequence_length 5 --output_length 5 --dataset HumanExcludeDataset --workers 1 --num_classes -1 --batch-size 128  --loss MocoImuLoss --num_imus 6 --imu_names neck body llegu rlegu larmu rarmu --step_size 50 --save_frequency 1 --input_feature_type move_label gaze_points --base-lr 0.001 --dropout 0.5 --data PATHTODATA/HumanDataset --skipping_frames 1
