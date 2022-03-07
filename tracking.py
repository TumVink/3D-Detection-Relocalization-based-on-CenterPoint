#import mayavi.mlab as mlab
import argparse
import glob
from pathlib import Path
import open3d as o3d


import numpy as np
import torch
import os
import json

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import RIO
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils
#from tools.visual_utils import visualize_utils as V
#from tools.visual_utils import visualize_o3d as V_o3d

import pandas as pd

EXCEL_PATH = "~/Downloads/VoteNet_1/dataset/config/Classes.xlsx"
df = pd.read_excel(EXCEL_PATH, sheet_name='Mapping')
def label2idx(label):
    # input: global label : "sofa"
    # output: index in rio7: 1
    return df[df['Label'] == label]['RIO7 Index'].iloc[0]




class DemoDataset(RIO):
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None):
        """
        Args:
            root_path:
            dataset_cfg:
            class_names:
            training:
            logger:
        """
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names, training=training, root_path=root_path, logger=logger
        )
        self.root_path = root_path
        #self.ext = ext
        data_file_list = glob.glob(str(root_path / f'*{self.ext}')) if self.root_path.is_dir() else [self.root_path]

        data_file_list.sort()
        self.sample_file_list = data_file_list

    def __len__(self):
        return len(self.sample_file_list)

    def __getitem__(self, index):
        if self.ext == '.bin':
            points = np.fromfile(self.sample_file_list[index], dtype=np.float32).reshape(-1, 4)
        elif self.ext == '.npy':
            points = np.load(self.sample_file_list[index])
        else:
            raise NotImplementedError

        input_dict = {
            'points': points,
            'frame_id': index,
        }

        data_dict = self.prepare_data(data_dict=input_dict)
        return data_dict


def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    #parser.add_argument('--cfg_file', type=str, default='cfgs/rio_configs/rio_model.yaml',
    #                    help='specify the config for demo')
    parser.add_argument('--data_path', type=str, default='demo_data',
                        help='specify the point cloud data file or directory')
    parser.add_argument('--ckpt', type=str, default='output/rio_model/default/ckpt/checkpoint_epoch_179.pth', help='specify the pretrained model')
    parser.add_argument('--ext', type=str, default='.bin', help='specify the extension of your point cloud data file')

    args = parser.parse_args()
    cfg_file = os.getcwd() + "/tools/cfgs/rio_configs/rio_model.yaml"
    cfg_from_yaml_file(cfg_file, cfg)

    return args, cfg


def main():
    args, cfg = parse_config()
    aliagn = False
    logger = common_utils.create_logger()
    logger.info('--------------------- Viz -------------------------')
    demo_dataset = RIO(
        dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES, training=False,
        root_path=Path(args.data_path), logger=logger
    )
    logger.info(f'Total number of samples: \t{len(demo_dataset)}')

    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=demo_dataset)
    model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=True)
    model.cuda()
    model.eval()
    TP_sum = 0
    FN_sum = 0
    FP_sum = 0
    t_sum = 0.0
    a_sum = 0.0
    recall = 0.0
    precision = 0.0
    t_median = []
    a_median = []
    with torch.no_grad():
        for idx, data_dict in enumerate(demo_dataset):
            logger.info(f'Visualized sample index: \t{idx + 1}')
            data_dict = demo_dataset.collate_batch([data_dict])
            load_data_to_gpu(data_dict)
            pred_dicts, _ = model.forward(data_dict)

            # aliagn boxes in rescan to ref
            cam_matrix, inquire_id_ls,trans_ls,ref_id = extract_info(data_dict)

            if aliagn:
                pred_boxes = aliagn_box(pred_dicts[0]['pred_boxes'], cam_matrix)
            else:
                if pred_dicts[0]['pred_boxes'] is not None and not isinstance(pred_dicts[0]['pred_boxes'], np.ndarray):
                    pred_boxes = pred_dicts[0]['pred_boxes'].cpu().numpy()  # (N,7)

            if pred_dicts[0]['pred_labels'] is not None and not isinstance(pred_dicts[0]['pred_labels'], np.ndarray):
                pred_labels = pred_dicts[0]['pred_labels'].cpu().numpy()  #(N,7)

            # extract center and size of inquire id in ref
            gt_boxes_inquire = extract_gt_boxes(ref_id, inquire_id_ls)

            # assign target_boxes to each inquire
            TP, FN, FP, t,angle,t_ls,a_ls = assign_target_eval(gt_boxes_inquire,pred_boxes,pred_labels,cam_matrix,trans_ls)

            TP_sum += TP
            FN_sum += FN
            FP_sum += FP
            t_sum += t
            a_sum += angle
            t_median.extend(t_ls)
            a_median.extend(a_ls)


            # # VIZ
            # geometry_list = []
            # geometry_list = V_o3d.get_boxes(data_dict['gt_boxes'][0,:,:7],False,geometry_list)
            # geometry_list = V_o3d.get_boxes(pred_dicts[0]['pred_boxes'],True,geometry_list)
            # geometry_list = V_o3d.get_pcd_from_np(data_dict['points'][:,1:],geometry_list)
            #
            #
            #
            # o3d.visualization.draw_geometries(geometry_list)#,point_size=1.0,line_width=2.0)
    recall = TP_sum/(FN_sum+TP_sum)
    precision = TP_sum/(TP_sum+FP_sum)
    t_error = t_sum/TP_sum
    a_sum = a_sum/TP_sum
    t_median = np.array(t_median)
    a_median = np.array(a_median)
    t_median = np.median(t_median)
    a_median = np.median(a_median)
    print("@20:")
    print(recall)
    print(precision)
    print(t_error)
    print(a_sum)
    print(t_median)
    print(a_median)


    logger.info('Demo done.')

def aliagn_box(boxes,tf_matrix):
    #aliagn box in scan with ref
    #output: np.array [N,7]
    if boxes is not None and not isinstance(boxes, np.ndarray):
        boxes = boxes.cpu().numpy()  #(N,7)
    rotation = np.array([tf_matrix[0],tf_matrix[1],tf_matrix[2],
                 tf_matrix[4],tf_matrix[5],tf_matrix[6],
                 tf_matrix[8],tf_matrix[9],tf_matrix[10]]).reshape(3,3)

    trans = np.array([tf_matrix[12],tf_matrix[13],tf_matrix[14]]).reshape(1,3)

    boxes[:,3:6] = boxes[:,3:6] @ rotation# + trans

    return boxes


# def extract_matrix(data_dict):
#
#     return data_dict['scan_dict'][0]['cam_trans']

def assign_target_eval(gt_boxes_inquire,pred_boxes,pred_labels,cam_matrix,trans_ls):
    # A match system for tracking

    # output: target boxes in inquire order
    #         np.array [N,] xyz+size+angle
    #gt_boxes_inquire: np.array(N,3+3+1+1)
    #pred_boxes: np.array(M,3+3+1)
    #pred_labels: np.array(M,)

    N = gt_boxes_inquire.shape[0]

    FN = 0
    TP = 0
    FP = 0
    t_sum = 0
    angle_sum=0
    t_ls = []
    a_ls = []

    for i in range(N):
        sizes = gt_boxes_inquire[i,3:6]

        #Filter One: The same label
        lb = gt_boxes_inquire[i,-1]
        mask = np.where(pred_labels == lb)
        if mask[0].size ==0:
            FN +=1
        else:
            pred_boxes_masked = pred_boxes[mask,:].reshape(-1,7)

            #Filter Two: The nearst distance for size
            dist2 = np.linalg.norm(pred_boxes_masked[:,3:6]-sizes,axis=1)
            target_box = pred_boxes[dist2.argmin(),:]

            pred_trans = compute_trans(target_box,gt_boxes_inquire[i,:],cam_matrix)

            pass_or_not, t, angle_error = satisfy_req(pred_trans,trans_ls[i])

            if pass_or_not:
                TP +=1
                t_sum +=t
                t_ls.append(t)
                a_ls.append(angle_error)
                angle_sum += angle_error
                print("yes!")
            else:
                FP +=1

    return TP,FN,FP,t_sum,angle_sum,t_ls,a_ls


def satisfy_req(pred_trans,trans_ls):
    from scipy.spatial.transform import Rotation as R

    #True: satifies the relocalization requirement
    R_inv = np.linalg.inv(pred_trans)
    #r_pred = R.from_matrix(pred_trans[:3,:3])
    #xyz_pred = r_pred.as_euler('zxy',degrees=True)

    cam_matrix = trans_ls
    R_GT = np.array([cam_matrix[0],cam_matrix[4],cam_matrix[8],cam_matrix[12],
                 cam_matrix[1],cam_matrix[5],cam_matrix[9],cam_matrix[13],
                 cam_matrix[2],cam_matrix[6],cam_matrix[10],cam_matrix[14],
                 0,0,0,1]).reshape(4,4)
    R_delta = R_inv@R_GT

    r = R.from_matrix(R_delta[:3,:3])
    xyz_angle = r.as_euler('zxy',degrees=True) #np.array[1,3]



    if abs(pred_trans[0,3] - trans_ls[12])<0.2 and abs(pred_trans[1,3] - trans_ls[13])<0.2 and abs(pred_trans[2,3] - trans_ls[14])<0.2 :
        if (np.abs(xyz_angle[0])%90) <= 20:
            t_error = np.mean([abs(pred_trans[0,3] - trans_ls[12]),abs(pred_trans[1,3] - trans_ls[13]),abs(pred_trans[2,3] - trans_ls[14])])
            angle_error = np.abs(xyz_angle[0])%90
            return True, t_error, angle_error
        else:
            return False, 1, 1
    else:
        return False,1,1

def satisfy_req2(pred_trans,trans_ls):
    #True: satifies the relocalization requirement
    if (abs(pred_trans[0,3] - trans_ls[12]) + abs(pred_trans[1,3] - trans_ls[13]) + abs(pred_trans[2,3] - trans_ls[14]))<=0.6:
        return True
    else:
        return False

def compute_trans(target_box,gt_boxes_inquire,cam_matrix):
    #target_box:[,7]
    #gt_boxes_inquire:[,8]
    angle = target_box[6] - gt_boxes_inquire[6]
    dx= target_box[3] - gt_boxes_inquire[3]
    dy = target_box[4] - gt_boxes_inquire[4]
    dz = target_box[5] - gt_boxes_inquire[5]
    local_Trans = np.array([np.cos(angle),-np.sin(angle),0,dx,
         np.sin(angle),np.cos(angle),0,dy,
         0,0,1,dz,
         0,0,0,1]).reshape(4,4)

    cam_matrix = np.array([cam_matrix[0],cam_matrix[4],cam_matrix[8],cam_matrix[12],
                 cam_matrix[1],cam_matrix[5],cam_matrix[9],cam_matrix[13],
                 cam_matrix[2],cam_matrix[6],cam_matrix[10],cam_matrix[14],
                 0,0,0,1]).reshape(4,4)

    Trans = local_Trans @ cam_matrix
    return Trans


def extract_info(data_dict):
    #extract inquire ids and cam_trans and ref_if
    #output: ls(len=16), list(len=N), ls, string
    id_list = []
    trans_list = []
    obj_list = data_dict['scan_dict'][0]['obj_movement']
    for i in range(len(obj_list)):
        id = obj_list[i]['instance_reference']
        trans = obj_list[i]['transform']
        id_list.append(id)
        trans_list.append(trans)

    ref_id = data_dict['scan_dict'][0]['ref_id']
    tf_matrix = data_dict['scan_dict'][0]['cam_trans']
    return tf_matrix, id_list, trans_list, ref_id

def extract_gt_boxes(ref_id, inquire_id_ls):
    #output: gt_boxes of inquire id
    #          np.array [N,3+3+1+1]  xyz+size+angle+cls
    data_dir = cfg.DATA_CONFIG.DATA_PATH + ref_id
    NUM_OBJ = len(inquire_id_ls)
    gt_boxes = []

    with open(data_dir + "/semseg.v2.json", 'r') as load_f:
        load_dict = json.load(load_f)
        seg_groups = load_dict['segGroups']

        idxes_rio7, centroids, sizes, orientation, heading_angles, box_label_mask, objs_id = [(0)] * NUM_OBJ,\
                                                                                             [(-1000, -1000, -1000)] * NUM_OBJ, [(0, 0, 0)] * NUM_OBJ, \
                                                                                             [(0, 0, 0, 0, 0, 0, 0, 0,0)] * NUM_OBJ, [( 0)] * NUM_OBJ, np.zeros(NUM_OBJ), \
                                                                                             [(0)] * NUM_OBJ  # idxes_rio7 are idx in rio7

        #         vote_label_mask = np.zeros((choices.shape[0],2)) #shape:[sample_size,2]
        #         point_votes = np.zeros((choices.shape[0],3)) # shape: sample_size,3. corresponding to "vote-label" in VoteNet

        #count = 0
        for i in range(len(seg_groups)):

            obj_id = load_dict['segGroups'][i]['objectId']
            label = load_dict['segGroups'][i]['label']
            idx_rio7 = label2idx(label)

            if obj_id in inquire_id_ls and idx_rio7!=0:

                obb = load_dict['segGroups'][i]['obb']
                orientation_matrix = obb['normalizedAxes']
                heading_angle = np.arccos(orientation_matrix[0])
                if orientation_matrix[1] < 0:
                    heading_angle = -heading_angle
                if heading_angle >= np.pi:
                    heading_angle = 0

                heading_angle = heading_angle.item() # conver nparray back to scalar



                gt_box = [obb['centroid'][0], obb['centroid'][1], obb['centroid'][2],
                          obb['axesLengths'][0], obb['axesLengths'][2], obb['axesLengths'][1],
                          heading_angle,idx_rio7]

                gt_boxes.append(gt_box)


        return np.array(gt_boxes)

if __name__ == '__main__':
    main()
