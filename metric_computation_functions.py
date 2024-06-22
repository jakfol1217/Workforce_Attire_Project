from tqdm import tqdm
from munkres import Munkres
import numpy as np
import torch
import json

def calculate_iou(bb1, bb2):
    # bb1, bb2: list of lists, bounding boxes should be of the form [x1, y1, x2, y2] (absolute, not relative)
    assert bb1[0] < bb1[2] and bb1[1] < bb1[3] and bb2[0] < bb2[2] and bb2[1] < bb2[3]
    o_x1 = max(bb1[0], bb2[0]) 
    o_y1 = max(bb1[1], bb2[1])
    o_x2 = min(bb1[2], bb2[2])
    o_y2 = min(bb1[3], bb2[3])
    overlap_area = rectangle_area(o_x1, o_y1, o_x2, o_y2)
    union_area = rectangle_area(*bb1) + rectangle_area(*bb2)
    iou = overlap_area/(union_area - overlap_area)
    return iou
    

def create_label_dict(labs):
    # create dictionary that assigns label number to name (and in reverse)
    label_dict = dict()
    label_dict.update({key:val for key,val in enumerate(labs)})

    label_dict[-1] = "none"

    label_dict_reverse = {
        val:key for key,val in label_dict.items()
    }
    return label_dict, label_dict_reverse


    
def rectangle_area(x1, y1, x2, y2):
    # compute area of a rectangle
    return max((x2 - x1), 0) * max((y2 - y1), 0)

def compare_labels(ground_truth, prediction):
    # function for matching labels between bounding boxes of given ground truths and predictions
    bb_gt = ground_truth['objects']['bbox']
    if torch.is_tensor(prediction['boxes']):
        bb_p = prediction['boxes'].detach().tolist()
    else:
        bb_p = prediction['boxes']
    labels_gt = ground_truth['objects']['category']
    if torch.is_tensor(prediction['labels']):
        labels_p = prediction['labels'].detach().tolist()
    else:
        labels_p = prediction['labels']
    if torch.is_tensor(prediction['scores']):
        scores_p = prediction['scores'].detach().numpy()
    else:
        scores_p = np.array(prediction['scores'])
    
    labels_matched_p = []
    labels_matched_gt = []
    ious = []
    
    # Situation in which we have no ground truths
    if len(labels_gt) == 0:
        for label in labels_p:
            labels_matched_p.append(label)
            labels_matched_gt.append(-1)
            ious.append(-1)
        return labels_matched_gt, labels_matched_p, scores_p, ious
    
    # Situation in which we have no predictions
    if len(labels_p) == 0:
        scores_p = scores_p.tolist()
        for label in labels_gt:
            labels_matched_gt.append(label)
            labels_matched_p.append(-1)
            scores_p.append(0)
            ious.append(-1)
        return labels_matched_gt, labels_matched_p, scores_p, ious
    
    iou_matrix = np.zeros((len(bb_p), len(bb_gt)))
    for i in range(len(bb_p)):
        for j in range(len(bb_gt)):
            iou_matrix[i, j] = calculate_iou(bb_p[i], bb_gt[j])
    
    max_cost = iou_matrix[np.unravel_index(iou_matrix.argmax(), iou_matrix.shape)]
    
    len_diff = len(bb_p) - len(bb_gt)
    if len_diff > 0: # adding padding if there are more predictions than ground truths
        iou_matrix = np.c_[iou_matrix, np.zeros((len(bb_p), len_diff)) + max_cost]
        for i in range(len_diff):
            labels_gt.append(-1)
    
    cost_matrix = max_cost - iou_matrix
    m = Munkres()
    optimal_idxes = m.compute(cost_matrix)
    used_idxes_p = []
    used_idxes_gt = []
    
    for (i_p, i_gt) in optimal_idxes:
        # adding matching labels
        labels_matched_p.append(labels_p[i_p])
        labels_matched_gt.append(labels_gt[i_gt])
        # removing used indexes
        used_idxes_gt.append(i_gt)
        used_idxes_p.append(i_p)
        # adding appropriate iou scores
        ious.append(iou_matrix[(i_p, i_gt)])
        
    # appending leftover predictions
    for i, p in enumerate(labels_p):
        if i in used_idxes_p:
            continue
        labels_matched_p.append(p)
        labels_matched_gt.append(-1)
        used_idxes_p.append(i)
        ious.append(-1)
        
    # rearranging scores
    scores_p = scores_p[used_idxes_p].tolist()

    # appending leftover ground truths
    for i, gt in enumerate(labels_gt):
        if i in used_idxes_gt:
            continue
        labels_matched_p.append(-1)
        labels_matched_gt.append(gt)
        ious.append(-1)
        scores_p.append(0)
    
    return labels_matched_gt, labels_matched_p, scores_p, ious

def assess_model(dataset, label_dict, detection_method, k=None, **kwargs):
    # compute matching labels, IOUs and bounding box scores for all elements of the dataset
    """
    Params:
    dataset -- dataset on which we are testing
    label_dict -- dictionary assigning indexes to classes
    detection_method -- bounding box detection method to use
    k -- k for top k assessment
    Returns (dict elements):
    labels_gt -- ground truth labels, as cloth item names. "None" means that there was a prediction, but no ground truth
    labels_p -- prediction labels, as cloth item names. "None" means that there was ground truth, but no prediction (false negative)
    scores -- confidence scores of bounding boxes. 0 if box is missing
    iou -- IOU scores of bounding boxes. -1 if box is missing
    """
    assessment_dict = {
        "labels_gt":[],
        "labels_p":[],
        "scores":[],
        "iou":[]
    }
    for data in tqdm(dataset):
        if k is None:
            labels_gt, labels_p, scores, iou = compare_labels(data, detection_method(data['image'], **kwargs))
        else:
            labels_gt, labels_p, scores, iou = compare_labels_topk(data, detection_method(data['image'], **kwargs), k)
        assessment_dict['labels_gt'].extend(labels_gt)
        assessment_dict['labels_p'].extend(labels_p)
        assessment_dict['scores'].extend(scores)
        assessment_dict['iou'].extend(iou)
    
    labels_p, labels_gt = translate_labels(assessment_dict['labels_p'], assessment_dict['labels_gt'], label_dict)
    assessment_dict['labels_gt'] =  np.array(labels_gt)
    assessment_dict['labels_p'] = np.array(labels_p)
    assessment_dict['scores'] = np.array(assessment_dict['scores'])
    assessment_dict['iou'] = np.array(assessment_dict['iou'])
    return assessment_dict



def translate_labels(labels_p, labels_gt, label_dict):
    # translating labels into clothing names
    
    new_labels_p = []
    new_labels_gt = []
    for p, gt in zip(labels_p, labels_gt):
        print(p)
        print(type(p))
        if isinstance(p, np.ndarray):
            trans_p = []
            for p_ele in p:
                trans_p.append(label_dict[p_ele])
            new_labels_p.append(trans_p)
        else:
            new_labels_p.append(label_dict[p])
        new_labels_gt.append(label_dict[gt])
    return new_labels_p, new_labels_gt

def calculate_positives_negatives(metrics, classes, threshold=0.5):
    # calculates the number of true/false positives and false negatives for each class
    fnp_dict = {
        key:{
            "false positives":0,
            "true positives":0,
            "false negatives":0

        } for key in classes
    }

    for p, gt, iou in zip(metrics['labels_p'], metrics['labels_gt'], metrics['iou']):
        if p == "none" and gt != "none":
            fnp_dict[gt]['false negatives'] += 1
            continue
        if p != "none" and gt == "none":
            fnp_dict[p]['false positives'] += 1
            continue
        if p == gt and iou > threshold:
            fnp_dict[p]['true positives'] += 1
            continue
        if p != gt or iou <= threshold:
            fnp_dict[p]['false positives'] += 1
            fnp_dict[gt]['false negatives'] += 1
            continue
    return fnp_dict

def write_fnp_dict(fnp_dict, filename):
    if not filename.endswith(".json"):
        filename += ".json"
    with open(filename, "w") as f:
        f.write(json.dumps(fnp_dict))

def calculate_precision(cls):
    if cls['true positives']== 0:
        return 0.0
    return cls['true positives']/(cls['true positives'] + cls['false positives'])

def calculate_recall(cls):
    if cls['true positives'] == 0:
        return 0.0
    return cls['true positives']/(cls['true positives'] + cls['false negatives'])

def average_scores(metrics, fnps):
    uniqs, counts = np.unique(metrics['labels_gt'], return_counts=True)
    count_dict = {
        u: c for (u, c) in zip(uniqs, counts)
    }
    av_prec = 0
    av_rec = 0
    for c in count_dict.keys():
        av_prec += (calculate_precision(fnps[c]) * count_dict[c])
        av_rec += (calculate_recall(fnps[c]) * count_dict[c])
    
    all_elems = len(metrics['labels_gt']) - count_dict["none"]

    return av_prec/all_elems, av_rec/all_elems

def compute_confusion_matrix(metrics, label_dict_reverse, threshold=0.4):
    # compute confusion matrix for a given metrics dict
    # threshold -- IOU threshold
    confusion_matrix = np.zeros((len(label_dict_reverse), len(label_dict_reverse)))
    for p, gt, iou in zip(metrics['labels_p'], metrics['labels_gt'], metrics['iou']):
        row = label_dict_reverse[gt]
        col = label_dict_reverse[p]
        if iou > threshold or iou == -1:
            confusion_matrix[row, col] += 1
        else:
            confusion_matrix[row, -1] += 1
            confusion_matrix[-1, col] += 1
    return confusion_matrix