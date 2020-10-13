import json
import torch
import h5py
import csv

def _read_labels(json_file, imus, sequence_length):
    """Returns a list of all frames, and a list of where each data point (whose
    length is sequence_length) in the list of frames."""
    with open(json_file, 'r') as fp:
        dataset_meta = json.load(fp)

    idx_to_fid = []
    clips = dataset_meta.keys()

    for clip in clips:
        removed = False
        if removed:
            continue
        idx_to_fid.append(clip)
    return dataset_meta, idx_to_fid

def is_consecutive(sequence):
    for i in range(len(sequence) - 1):
        if sequence[i] + 1 != sequence[i+1]:
            return False
    return True

def get_classification_weights(json_file):

    with open(json_file, 'r') as f:
        cluster_weights = json.load(f)

    return torch.Tensor(cluster_weights['weights'])


def _read_json(json_file):
    with open(json_file, 'r') as f:
        result_dict = json.load(f)
    return result_dict


def _read_h5py(h5adr):
    return h5py.File(h5adr, 'r')

def _read_csv(csv_file):
    with open(csv_file) as f:
        reader = csv.DictReader(f)
        all_rows = [dict(row) for row in reader]
        keys = reader.fieldnames
    return keys, all_rows

