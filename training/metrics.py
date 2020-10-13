"""
=================
This file contains all the metrics that is used for evaluation. 
=================
"""

import torch

class AverageMeter(object):

    def __init__(self):
        self.val = None
        self.sum = None
        self.count = 0

    def update(self, val, n=1):
        with torch.no_grad():
            if self.val is None:
                self.val = val
                self.sum = self.val * n
            else:
                self.val = val
                self.sum += val * n
            self.count += n

    @property
    def avg(self):
        with torch.no_grad():
            return self.sum / self.count if self.count > 0 else 0

    def __str__(self):
        return 'Avg({}),Ct({}),Sum({})'.format(self.avg, self.count, self.sum)


class BaseMetric:

    def record_output(self, output, target, batch_size=1):
        raise Exception('record_output is not implemented')

    def report(self):
        raise Exception('report is not implemented')

    def final_report(self):
        return self.report()
    
    def average(self): 
        raise Exception('avearge is not implemented')


class WalkMetric(BaseMetric):

    def __init__(self, args):
        self.args = args
        self.num_classes = args.num_classes
        self.meter = {'walk':AverageMeter(), 'no_walk':AverageMeter()}

    def calc_iou(self, walk_output, walk_target):
        assert walk_output.shape == walk_target.shape
        sum_target_pred = walk_output + walk_target
        intersection = sum_target_pred == 2
        union = sum_target_pred >= 1

        batch_intersection = intersection.sum(-1).sum(-1).float()
        batch_union = union.sum(-1).sum(-1).float()

        total_iou = batch_intersection / (batch_union)
        batch_size , seq_len, c = total_iou.shape
        assert seq_len == c == 1
        ratio = total_iou[batch_union > 0].sum() / (batch_union > 0).sum()
        return ratio

    def record_output(self, output, target, batch_size=1):

        walk_output = output['walk']
        walk_target = target['walk']

        walk_output = torch.argmax(walk_output, dim=2).unsqueeze(2)
        ratio = self.calc_iou(walk_output, walk_target)

        self.meter['walk'].update(ratio.item(), batch_size)


        no_walk_output = 1 - walk_output
        no_walk_target = 1 - walk_target

        no_walk_ratio = self.calc_iou(no_walk_output, no_walk_target)

        self.meter['no_walk'].update(no_walk_ratio.item(), batch_size)


    def report(self):
        all_averages = [v for (k,v) in self.average().items()]
        return ' WalkMetric {:.5f} {:.5f} '.format(sum(all_averages) / len(all_averages), self.meter['walk'].avg)

    def average(self):
        average_report = {str(verb_class): averagemet.avg for (verb_class, averagemet) in self.meter.items()}
        all_averages = [val for val in average_report.values()]
        average_report['all'] = sum(all_averages) / len(all_averages)
        return average_report

class ActionMetric(BaseMetric):

    def __init__(self, args):
        self.args = args
        weight_freq_inv = args.dataset.CLASS_WEIGHTS
        self.meter = {action_class_label: AverageMeter() for action_class_label in range(len(weight_freq_inv)) if weight_freq_inv[action_class_label] != 0}

    def record_output(self, output, target, batch_size=1):

        verb_class_output = output['verb_actual_class']
        verb_class_target = target['verb_class']

        assert verb_class_output.shape == verb_class_target.shape

        corrects = verb_class_output == verb_class_target #batchsize

        for verb_index in self.meter:
            this_verb = verb_class_target == verb_index
            this_verb_correct = corrects[this_verb]
            count_of_this_verb = this_verb.sum()
            if count_of_this_verb == 0:
                continue
            self.meter[verb_index].update((this_verb_correct.sum().float() / count_of_this_verb).item())

    def report(self):
            return ' ActionMetric {:.5f}'.format(self.average()['avg_per_class'])

    def average(self):
        all_averages = [val.avg for val  in self.meter.values()]
        average_report = {'avg_per_class': sum(all_averages) / len(all_averages)}
        return average_report

class VindMetric(BaseMetric):

    def __init__(self, args):
        self.args = args
        self.meter = {i:AverageMeter() for i in range(args.num_classes)}


    def record_output(self, output, target, batch_size=1):

        vind_class_output = output['vind_actual_class']
        vind_class_target = target['vind_class']

        assert vind_class_output.shape == vind_class_target.shape

        corrects = vind_class_output == vind_class_target #batchsize

        for vind_index in self.meter:
            this_vind = vind_class_target == vind_index
            this_vind_correct = corrects[this_vind]
            count_of_this_vind = this_vind.sum()
            if count_of_this_vind == 0:
                continue
            self.meter[vind_index].update((this_vind_correct.sum().float() / count_of_this_vind).item())

    def report(self):
        return ' VindMetric {:.5f}'.format(self.average()['avg_per_class'])

    def average(self):
        all_averages = [val.avg for val  in self.meter.values()]
        average_report = {'avg_per_class': sum(all_averages) / len(all_averages)}
        return average_report

class SceneMetric(BaseMetric):

    def __init__(self, args):
        self.args = args
        self.meter = AverageMeter()

    def record_output(self, output, target, batch_size=1):

        scene_class_output = output['scene_actual_class']
        scene_class_target = target['scene_class']

        assert scene_class_output.shape == scene_class_target.shape

        corrects = scene_class_output == scene_class_target #batchsize
        batch_size = len(corrects)
        self.meter.update((corrects.sum().float() / batch_size).item(), batch_size)


    def report(self):
        return ' SceneMetric {:.5f}'.format(self.meter.avg)

    def average(self):
        average_report = {
            'scene_classification': self.meter.avg
        }
        return average_report


class DepthMetric(BaseMetric):

    def __init__(self, args):
        self.args = args
        self.meter = AverageMeter()

    def record_output(self, output, target, batch_size=1):

        depth_output = output['depth']
        depth_target = target['depth']
        depth_output = (depth_output + 1) / 2
        depth_target = (depth_target + 1) / 2

        assert depth_output.shape == depth_target.shape

        batch_size = depth_target.shape[0]

        all_result = torch.abs(depth_output - depth_target) / torch.abs(depth_target)

        self.meter.update((all_result.mean()).item(), batch_size)


    def report(self):
        return ' DepthMetric {:.5f}'.format(self.meter.avg)

    def average(self):
        average_report = {
            'scene_classification': self.meter.avg
        }
        return average_report

class DepthMSE(BaseMetric):

    def __init__(self, args):
        self.args = args
        self.meter = AverageMeter()

    def record_output(self, output, target, batch_size=1):

        depth_output = output['depth']
        depth_target = target['depth']
        depth_output = (depth_output + 1) / 2
        depth_target = (depth_target + 1) / 2

        assert depth_output.shape == depth_target.shape

        batch_size = depth_target.shape[0]

        all_result = torch.pow(depth_output - depth_target, 2)

        self.meter.update((all_result.mean() ** .5).item(), batch_size)


    def report(self):
        return ' DepthMSE {:.5f}'.format(self.meter.avg)

    def average(self):
        average_report = {
            'scene_classification': self.meter.avg
        }
        return average_report


class DepthLogMSE(BaseMetric):

    def __init__(self, args):
        self.args = args
        self.meter = AverageMeter()

    def record_output(self, output, target, batch_size=1):

        depth_output = output['depth']
        depth_target = target['depth']
        depth_output = (depth_output + 1) / 2
        depth_target = (depth_target + 1) / 2

        assert depth_output.shape == depth_target.shape

        batch_size = depth_target.shape[0]

        depth_output = torch.clamp(depth_output, min=0.00001)

        depth_output = torch.log10(depth_output)
        depth_target = torch.log10(depth_target)


        all_result = torch.pow(depth_output - depth_target, 2)


        self.meter.update((all_result.mean() ** .5).item(), batch_size)


    def report(self):
        return ' DepthLogMSE {:.5f}'.format(self.meter.avg)

    def average(self):
        average_report = {
            'scene_classification': self.meter.avg
        }
        return average_report
class DepthLog(BaseMetric):

    def __init__(self, args):
        self.args = args
        self.meter = AverageMeter()

    def record_output(self, output, target, batch_size=1):

        depth_output = output['depth']
        depth_target = target['depth']
        depth_output = (depth_output + 1) / 2
        depth_target = (depth_target + 1) / 2

        assert depth_output.shape == depth_target.shape

        batch_size = depth_target.shape[0]

        depth_output = torch.clamp(depth_output, min=0.00001)

        depth_output = torch.log10(depth_output)
        depth_target = torch.log10(depth_target)


        all_result = torch.abs(depth_output - depth_target)


        self.meter.update((all_result.mean()).item(), batch_size)


    def report(self):
        return ' DepthLog {:.5f}'.format(self.meter.avg)

    def average(self):
        average_report = {
            'scene_classification': self.meter.avg
        }
        return average_report



class ClassificationMetric(BaseMetric):

    def __init__(self, args):
        self.args = args
        self.meter = AverageMeter()

    def record_output(self, output, target, batch_size=1):

        class_label_output = output['actual_class_label']
        class_label_target = target['class_label']

        assert class_label_output.shape == class_label_target.shape

        corrects = class_label_output == class_label_target #batchsize
        batch_size = len(corrects)
        self.meter.update((corrects.sum().float() / batch_size).item(), batch_size)


    def report(self):
        return ' ClassificationMetric {:.5f}'.format(self.meter.avg)

    def average(self):
        average_report = {
            'class_label': self.meter.avg
        }
        return average_report

class MoveDetectorMetric(BaseMetric):

    def __init__(self, args):
        self.args = args
        self.imu_names = args.imu_names
        self.meter = {imu_name: AverageMeter() for imu_name in self.imu_names}

    def record_output(self, output, target, batch_size=1):

        direction_output = output['move_label']
        direction_target = target['move_label']

        assert direction_output.shape == direction_target.shape

        direction_output_labels = output['cleaned_move_label']
        corrects = direction_output_labels == direction_target #batchsize x seqlen x imunum

        not_masked = direction_target != -1

        for imu_ind, imu_name in enumerate(self.imu_names):
            this_imu_acc = corrects[:,:,imu_ind][not_masked[:,:,imu_ind]]
            if len(this_imu_acc) == 0:
                continue
            self.meter[imu_name].update(this_imu_acc.sum().float().item() / len(this_imu_acc))



    def report(self):
        all_averages = [v for (k,v) in self.average().items()]
        return ' MoveDetect {:.5f}'.format(sum(all_averages) / len(all_averages))

    def final_report(self):
        return self.report()

    def average(self):
        average_report = {imu_name: averagemet.avg for (imu_name, averagemet) in self.meter.items()}
        all_averages = [val for val  in average_report.values()]
        average_report['all'] = sum(all_averages) / len(all_averages)
        return average_report
