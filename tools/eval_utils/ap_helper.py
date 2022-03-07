import os
import sys
import numpy as np
import torch

from tools.eval_utils.eval_det import eval_det
from tools.eval_utils.eval_det import get_iou_obb
from tools.eval_utils.box_util import get_3d_box

def in_hull(p, hull):
    from scipy.spatial import Delaunay
    if not isinstance(hull,Delaunay):
        hull = Delaunay(hull)
    return hull.find_simplex(p)>=0

def extract_pc_in_box3d(pc, box3d):
    ''' pc: (N,3), box3d: (8,3) '''
    box3d_roi_inds = in_hull(pc[:,0:3], box3d)
    return pc[box3d_roi_inds,:], box3d_roi_inds

def flip_axis_to_camera(pc):
    ''' Flip X-right,Y-forward,Z-up to X-right,Y-down,Z-forward
    Input and output are both (N,3) array
    '''
    pc2 = np.copy(pc)
    pc2[...,[0,1,2]] = pc2[...,[0,2,1]] # cam X,Y,Z = depth X,-Z,Y
    pc2[...,1] *= -1
    return pc2

def flip_axis_to_depth(pc):
    pc2 = np.copy(pc)
    pc2[...,[0,1,2]] = pc2[...,[0,2,1]] # depth X,Y,Z = cam X,Z,-Y
    pc2[...,2] *= -1
    return pc2

def softmax(x):
    ''' Numpy function for softmax'''
    shape = x.shape
    probs = np.exp(x - np.max(x, axis=len(shape)-1, keepdims=True))
    probs /= np.sum(probs, axis=len(shape)-1, keepdims=True)
    return probs

def parse_predictions(pred_dict):
    max_obj = 147
    bsize = len(pred_dict)
    pred_center, pred_angle, pred_size, pred_sem_cls, obj_prob = np.zeros([bsize,max_obj,3]),np.zeros([bsize,max_obj])\
        ,np.zeros([bsize,max_obj,3]),np.zeros([bsize,max_obj]),np.zeros([bsize,max_obj])
    nonempty_box_mask = np.zeros((bsize, max_obj))
    for i in range(bsize): # batch size
        for j in range(max_obj): # max obj
            if j < pred_dict[i]['pred_boxes'].shape[0]: # actual num_objs after nms
                nonempty_box_mask[i,j] = 1
                pred_center[i,j] = pred_dict[i]['pred_boxes'][j,0:3].detach().cpu().numpy()
                pred_angle[i,j] = pred_dict[i]['pred_boxes'][j,6].detach().cpu().numpy()
                pred_size[i,j] = pred_dict[i]['pred_boxes'][j,3:6].detach().cpu().numpy()
                pred_sem_cls[i,j] = pred_dict[i]['pred_labels'][j].detach().cpu().numpy()
                obj_prob[i,j] = pred_dict[i]['pred_scores'][j].detach().cpu().numpy()

    # Since we operate in upright_depth coord for points, while util functions
    # assume upright_camera coord.

    pred_corners_3d_upright_camera = np.zeros((bsize, max_obj, 8, 3))
    pred_center_upright_camera = flip_axis_to_camera(pred_center)
    for i in range(bsize):
        for j in range(max_obj):
            heading_angle = pred_angle[i,j]
            box_size = pred_size[i,j]
            corners_3d_upright_camera = get_3d_box(box_size, heading_angle, pred_center_upright_camera[i,j,:]) # 3rscan is l h w
            pred_corners_3d_upright_camera[i,j] = corners_3d_upright_camera

    # -------------------------------------
    # Remove predicted boxes without any point within them..
    """
    batch_pc = pcd.cpu().numpy()[:,:,0:3] # B,N,3
    for i in range(bsize):
        pc = batch_pc[i,:,:] # (N,3)
        for j in range(K):
            box3d = pred_corners_3d_upright_camera[i,j,:,:] # (8,3)
            box3d = flip_axis_to_depth(box3d)
            pc_in_box,inds = extract_pc_in_box3d(pc, box3d)
            if len(pc_in_box) < 5:
                nonempty_box_mask[i,j] = 0
    # -------------------------------------
    """

    pred_mask = np.zeros((bsize, max_obj))
    for i in range(bsize):
        nonempty_box_inds = np.where(nonempty_box_mask[i,:]==1)[0]
        pred_mask[i, nonempty_box_inds] = 1

    batch_pred_map_cls = [] # a list (len: batch_size) of list (len: num of predictions per sample) of tuples of pred_cls, pred_box and conf (0-1)
    for i in range(bsize):
        batch_pred_map_cls.append([(pred_sem_cls[i,j].item(), pred_corners_3d_upright_camera[i,j], obj_prob[i,j]) \
            for j in range(pred_center.shape[1]) if pred_mask[i,j]==1 and obj_prob[i,j]>0.05])

    return batch_pred_map_cls

def parse_groundtruths(data_dict):
    center_label = data_dict['gt_boxes'][:,:,0:3]
    print(center_label.shape)
    heading_angle = data_dict['gt_boxes'][:,:,6]
    box_size = data_dict['gt_boxes'][:,:,3:6]
    sem_cls_label = data_dict['gt_boxes'][:,:,7]
    bsize = center_label.shape[0]

    K2 = center_label.shape[1] # K2==NUM_OBJ
    gt_corners_3d_upright_camera = np.zeros((bsize, K2, 8, 3))
    gt_center_upright_camera = flip_axis_to_camera(center_label[:,:,0:3].detach().cpu().numpy())
    for i in range(bsize):
        for j in range(K2):
            corners_3d_upright_camera = get_3d_box(box_size[i,j,:].detach().cpu().numpy(), heading_angle[i,j].detach().cpu().numpy(), gt_center_upright_camera[i,j,:]) # 3rscan is l h w
            gt_corners_3d_upright_camera[i,j] = corners_3d_upright_camera

    batch_gt_map_cls = []
    for i in range(bsize):
        batch_gt_map_cls.append([(sem_cls_label[i,j].item(), gt_corners_3d_upright_camera[i,j]) for j in range(gt_corners_3d_upright_camera.shape[1])])

    return batch_gt_map_cls

class APCalculator(object):
    ''' Calculating Average Precision '''
    def __init__(self, ap_iou_thresh=0.25, class2type_map=None):
        """
        Args:
            ap_iou_thresh: float between 0 and 1.0
                IoU threshold to judge whether a prediction is positive.
            class2type_map: [optional] dict {class_int:class_name}
        """
        self.ap_iou_thresh = ap_iou_thresh
        self.class2type_map = class2type_map
        self.reset()
        
    def step(self, batch_pred_map_cls, batch_gt_map_cls):
        """ Accumulate one batch of prediction and groundtruth.
        
        Args:
            batch_pred_map_cls: a list of lists [[(pred_cls, pred_box_params, score),...],...]
            batch_gt_map_cls: a list of lists [[(gt_cls, gt_box_params),...],...]
                should have the same length with batch_pred_map_cls (batch_size)
        """
        
        bsize = len(batch_pred_map_cls)
        assert(bsize == len(batch_gt_map_cls))
        for i in range(bsize):
            self.gt_map_cls[self.scan_cnt] = batch_gt_map_cls[i] 
            self.pred_map_cls[self.scan_cnt] = batch_pred_map_cls[i] 
            self.scan_cnt += 1
    
    def compute_metrics(self):
        """ Use accumulated predictions and groundtruths to compute Average Precision.
        """
        rec, prec, ap = eval_det(self.pred_map_cls, self.gt_map_cls, ovthresh=self.ap_iou_thresh, get_iou_func=get_iou_obb)
        ret_dict = {} 
        for key in sorted(ap.keys()):
            clsname = self.class2type_map[key] if self.class2type_map else str(key)
            #ret_dict['%s Average Precision'%(clsname)] = ap[key]
            ret_dict[clsname] = ap[key]
        ret_dict['mAP'] = np.mean(list(ap.values()))
        #rec_list = []
        #for key in sorted(ap.keys()):
            #clsname = self.class2type_map[key] if self.class2type_map else str(key)
            #try:
                #ret_dict['%s Recall'%(clsname)] = rec[key][-1]
                #rec_list.append(rec[key][-1])
            #except:
                #ret_dict['%s Recall'%(clsname)] = 0
                #rec_list.append(0)
        #ret_dict['AR'] = np.mean(rec_list)
        return ret_dict

    def reset(self):
        self.gt_map_cls = {} # {scan_id: [(classname, bbox)]}
        self.pred_map_cls = {} # {scan_id: [(classname, bbox, score)]}
        self.scan_cnt = 0