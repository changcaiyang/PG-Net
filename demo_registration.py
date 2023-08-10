import json
import torch
import copy
import argparse
from easydict import EasyDict as edict
from models.PGNet import PGNet
from utils.pointcloud import estimate_normal, make_point_cloud
# from models import PointDSC
import torch
import numpy as np
import open3d as o3d
from utils.SE3 import *
from sklearn.metrics import recall_score, precision_score, f1_score
from libs.loss import TransformationLoss, ClassificationLoss

'''class open3d.geometry.LineSet


  LineSet = Type.LineSet
  PointCloud = Type.PointCloud
 

  create_from_point_cloud_correspondences(cloud0, cloud1, correspondences)

            cloud0 (open3d.geometry.PointCloud)

            cloud1 (open3d.geometry.PointCloud)

            correspondences (List[Tuple[int, int]])

    return open3d.geometry.LineSet'''


def extract_fcgf_features(pcd_path, downsample, device, weight_path='misc/ResUNetBN2C-feat32-3dmatch-v0.05.pth'):
    raw_src_pcd = o3d.io.read_point_cloud(pcd_path)
    pts = np.array(raw_src_pcd.points)
    from misc.fcgf import ResUNetBN2C as FCGF
    from misc.cal_fcgf import extract_features
    fcgf_model = FCGF(
        1,
        32,
        bn_momentum=0.05,
        conv1_kernel_size=7,
        normalize_feature=True
    ).to(device)
    checkpoint = torch.load(weight_path)
    fcgf_model.load_state_dict(checkpoint['state_dict'])
    fcgf_model.eval()

    xyz_down, features = extract_features(
        fcgf_model,
        xyz=pts,
        rgb=None,
        normal=None,
        voxel_size=downsample,
        skip_check=True,
    )
    return raw_src_pcd, xyz_down.astype(np.float32), features.detach().cpu().numpy()


def extract_fpfh_features(pcd_path, downsample, device):
    raw_src_pcd = o3d.io.read_point_cloud(pcd_path)
    estimate_normal(raw_src_pcd, radius=downsample * 2)
    src_pcd = raw_src_pcd.voxel_down_sample(downsample)
    src_features = o3d.registration.compute_fpfh_feature(src_pcd,
                                                         o3d.geometry.KDTreeSearchParamHybrid(radius=downsample * 5,
                                                                                              max_nn=100))
    src_features = np.array(src_features.data).T
    src_features = src_features / (np.linalg.norm(src_features, axis=1, keepdims=True) + 1e-6)
    return raw_src_pcd, np.array(src_pcd.points), src_features
    #return raw_src_pcd.points, np.array(src_pcd.points).astype(np.float32), src_features.astype(np.float32)



def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    if not source_temp.has_normals():
        estimate_normal(source_temp)
        estimate_normal(target_temp)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp])


# def vis_pair(src_pcd, tgt_pcd, src_keypts, tgt_keypts, corr, gt_labels, pred_labels, ransac_labels, max_corr=None):


def vis_pair(src_pcd, tgt_pcd, src_keypts, tgt_keypts, corr, gt_labels, pred_labels, max_corr=None):
    keypts0 = make_point_cloud(src_keypts)
    keypts1 = make_point_cloud(tgt_keypts)
    ## add offset to target pcd for visualzation
    tgt_pcd.translate([0, 2.5, 0])
    keypts1.translate([0, 2.5, 0])

    i = 0
    if max_corr is not None:
        sel_ind = np.random.choice(len(gt_labels), max_corr)
        corr = corr[sel_ind]
        gt_labels = gt_labels[sel_ind]
        pred_labels = pred_labels[sel_ind]
        # ransac_labels = ransac_labels[sel_ind]


    # visualize the filtered correspondence set of PointSM
    right_idx = np.where((gt_labels > 0) & (pred_labels > 0))[0]
    wrong_idx = np.where((gt_labels == 0) & (pred_labels > 0))[0]
    result_geometries = draw_matching_result(src_pcd, tgt_pcd, keypts0, keypts1, corr, right_idx, wrong_idx)

    o3d.visualization.draw_geometries(result_geometries)


def draw_matching_result(source, target, source_keypts, target_keypts, corr, right_idx, wrong_idx):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    estimate_normal(source_temp, radius=0.10, max_nn=30)
    estimate_normal(target_temp, radius=0.10, max_nn=30)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    right_line = o3d.geometry.LineSet.create_from_point_cloud_correspondences(source_keypts, target_keypts,
                                                                              corr[right_idx])
    right_line.paint_uniform_color([0, 1, 0])
    wrong_line = o3d.geometry.LineSet.create_from_point_cloud_correspondences(source_keypts, target_keypts,
                                                                              corr[wrong_idx])
    wrong_line.paint_uniform_color([1, 0, 0])
    # o3d.visualization.draw_geometries([source_temp, target_temp, right_line, wrong_line])
    geometries_list = [source_temp, target_temp, right_line, wrong_line]
    return geometries_list


'''def create_from_point_cloud_correspondences():

    cloud0 = open3d.geometry.PointCloud()
    cloud1 = o3d.geometry.PointCloud()
    correspondences = o3d.List[Tuple[int, int]]
    return o3d.geometry.LineSet'''


'''def __init__(self,
             inlier_threshold=0.10,
             downsample=0.03,
             augment_axis=0,
             augment_rotation=1.0,
             augment_translation=0.01,
             ):
    self.inlier_threshold = inlier_threshold
    self.downsample = downsample
    self.augment_axis = augment_axis
    self.augment_rotation = augment_rotation
    self.augment_translation = augment_translation'''


'''def __init__(
        self,
        root,
        split,
        descriptor='fcgf',
        in_dim=6,
        inlier_threshold=0.10,
        num_node=5000,
        use_mutual=True,
        downsample=0.03,
        augment_axis=1,
        augment_rotation=1.0,
        augment_translation=0.01,
        ):
    self.root = root
    self.split = split
    self.descriptor = descriptor
    assert descriptor in ['fpfh', 'fcgf']
    self.in_dim = in_dim
    self.inlier_threshold = inlier_threshold
    self.num_node = num_node
    self.use_mutual = use_mutual
    self.downsample = downsample
    self.augment_axis = augment_axis
    self.augment_rotation = augment_rotation
    self.augment_translation = augment_translation'''


if __name__ == '__main__':
    from config import str2bool

    parser = argparse.ArgumentParser()
    parser.add_argument('--chosen_snapshot', default='PointDSC_3DMatch_yw1', type=str, help='snapshot dir')
    parser.add_argument('--pcd1', default='data/3DMatch/fragments/sun3d-hotel_umd-maryland_hotel1/cloud_bin_10.ply', type=str)
    parser.add_argument('--pcd2', default='data/3DMatch/fragments/sun3d-hotel_umd-maryland_hotel1/cloud_bin_9.ply', type=str)
    parser.add_argument('--descriptor', default='fpfh', type=str, choices=['fcgf', 'fpfh'])
    parser.add_argument('--use_gpu', default=True, type=str2bool)
    args = parser.parse_args()

    config_path = f'/media/lixiaohan/下载/huohu/PointDSC-master/snapshot/{args.chosen_snapshot}/config.json'
    config = json.load(open(config_path, 'r'))
    config = edict(config)


    if args.use_gpu:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    model = PGNet(
        in_dim=config.in_dim,
        num_layers=config.num_layers,
        num_channels=config.num_channels,
        num_iterations=config.num_iterations,
        ratio=config.ratio,
        sigma_d=config.sigma_d,
        k=config.k,
        nms_radius=config.inlier_threshold,
    ).to(device)
    miss = model.load_state_dict(
        torch.load(f'/media/lixiaohan/下载/huohu/PointDSC-master/snapshot/{args.chosen_snapshot}/models/model_best.pkl', map_location=device), strict=False)
    print(miss)
    model.eval()

    # extract features
    if args.descriptor == 'fpfh':
        raw_src_pcd, src_pts, src_features = extract_fpfh_features(args.pcd1, config.downsample, device)
        raw_tgt_pcd, tgt_pts, tgt_features = extract_fpfh_features(args.pcd2, config.downsample, device)
    else:
        raw_src_pcd, src_pts, src_features = extract_fcgf_features(args.pcd1, config.downsample, device)
        raw_tgt_pcd, tgt_pts, tgt_features = extract_fcgf_features(args.pcd2, config.downsample, device)

    augment_axis = 0
    augment_rotation = 1.0
    augment_translation = 0.01
    inlier_threshold = 0.10
    '''# matching
    distance = np.sqrt(2 - 2 * (src_features @ tgt_features.T) + 1e-6)
    source_idx = np.argmin(distance, axis=1)
    source_dis = np.min(distance, axis=1)
    corr = np.concatenate([np.arange(source_idx.shape[0])[:, None], source_idx[:, None]], axis=-1)
    #target_idx = np.argmin(distance, axis=0)
    #mutual_nearest = (target_idx[source_idx] == np.arange(source_idx.shape[0]))
    #corr = np.concatenate([np.where(mutual_nearest == 1)[0][:, None], source_idx[mutual_nearest][:, None]], axis=-1)
    src_keypts = src_pts[corr[:, 0]]
    tgt_keypts = tgt_pts[corr[:, 1]]
    corr_pos = np.concatenate([src_keypts, tgt_keypts], axis=-1)
    corr_pos = corr_pos - corr_pos.mean(0)
    #labels = (distance < 0.10).astype(np.int)

    # compute ground truth transformation
    orig_trans = np.eye(4).astype(np.float32)
    # data augmentation (add data augmentation to original transformation)
    #src_keypts += np.random.rand(src_keypts.shape[0], 3) * 0.005
    #tgt_keypts += np.random.rand(tgt_keypts.shape[0], 3) * 0.005
    #aug_R = rotation_matrix(0, 1.0)
    #aug_T = translation_matrix(0.01)
    aug_R = rotation_matrix(augment_axis, augment_rotation)
    aug_T = translation_matrix(augment_translation)
    aug_trans = integrate_trans(aug_R, aug_T)
    tgt_keypts = transform(tgt_keypts, aug_trans)
    gt_trans = concatenate(aug_trans, orig_trans)

    frag1 = src_keypts[corr[:, 0]]
    frag2 = tgt_keypts[corr[:, 1]]
    #frag1 = src_keypts
    #frag2 = tgt_keypts
    frag1_warp = transform(frag1, gt_trans)
    distance1 = np.sqrt(np.sum(np.power(frag1_warp - frag2, 2), axis=1))
    labels = (distance1 < inlier_threshold).astype(np.int)
    
    # outlier rejection
    data = {
        'corr_pos': torch.from_numpy(corr_pos)[None].to(device).float(),
        'src_keypts': torch.from_numpy(src_keypts)[None].to(device).float(),
        'tgt_keypts': torch.from_numpy(tgt_keypts)[None].to(device).float(),
        'testing': True,
    }
    res = model(data)
    vis_pair(raw_src_pcd, raw_tgt_pcd, src_keypts, tgt_keypts, corr, labels, res['final_labels'][0].detach().cpu().numpy(), max_corr=None)
'''


    # compute ground truth transformation
    orig_trans = np.eye(4).astype(np.float32)
    # data augmentation (add data augmentation to original transformation)
    #src_pts += np.random.rand(src_pts.shape[0], 3) * 0.005
    #tgt_pts += np.random.rand(tgt_pts.shape[0], 3) * 0.005
    #aug_R = rotation_matrix(0, 1.0)
    #aug_T = translation_matrix(0.01)
    aug_R = rotation_matrix(augment_axis, augment_rotation)
    aug_T = translation_matrix(augment_translation)
    #aug_R = rotation_matrix(1, 1.0)
    #aug_T = translation_matrix(0.01)
    aug_trans = integrate_trans(aug_R, aug_T)
    tgt_pts1 = transform(tgt_pts, aug_trans)
    #gt_trans = concatenate(aug_trans, orig_trans)
    '''gt_trans = np.array([0.7588320340,	-0.2199405670,	0.6130253600,	-0.3326754260,
0.2430566790,	0.9688849490,	0.0467483293,	0.1518452290,
-0.6042328990,	0.1135257780,	0.7886789600,	-0.1404618600,
0.0000000000,	0.0000000000,	0.0000000000,	1.0000000000]).reshape(4, 4)'''
    gt_trans = np.array([0.8459279730,	0.1179025020,	-0.5201008210,	0.4389697990,
-0.1049496770,	0.9929882640,	0.0544047076,	-0.1232777150,
0.5228684630,	0.0085619490,	0.8523703790,	0.3425145740,
0.0000000000,	0.0000000000,	0.0000000000,	1.0000000000]).reshape(4, 4)
    #gt_trans = concatenate(aug_trans, orig_trans)
    num_node = 2000
    # select {self.num_node} numbers of keypoints
    N_src = src_features.shape[0]
    N_tgt = tgt_features.shape[0]
    if num_node == 'all':
        src_sel_ind = np.arange(N_src)
        tgt_sel_ind = np.arange(N_tgt)
    else:
        src_sel_ind = np.random.choice(N_src, num_node)
        tgt_sel_ind = np.random.choice(N_tgt, num_node)
    src_desc = src_features[src_sel_ind, :]
    tgt_desc = tgt_features[tgt_sel_ind, :]
    src_keypts = src_pts[src_sel_ind, :]
    tgt_keypts = tgt_pts1[tgt_sel_ind, :]

    # construct the correspondence set by nearest neighbor searching in feature space.
    distance = np.sqrt(2 - 2 * (src_desc @ tgt_desc.T) + 1e-6)
    source_idx = np.argmin(distance, axis=1)
    source_dis = np.min(distance, axis=1)
    corr = np.concatenate([np.arange(source_idx.shape[0])[:, None], source_idx[:, None]], axis=-1)
    src_keypts2 = src_keypts[corr[:, 0]]
    tgt_keypts2 = tgt_keypts[corr[:, 1]]
    corr_pos = np.concatenate([src_keypts2, tgt_keypts2], axis=-1)
    corr_pos = corr_pos - corr_pos.mean(0)

    # compute the ground truth label
    frag1 = src_keypts[corr[:, 0]]
    frag2 = tgt_keypts[corr[:, 1]]
    frag1_warp = transform(frag1, gt_trans)
    distance = np.sqrt(np.sum(np.power(frag1_warp - frag2, 2), axis=1))
    labels = (distance < inlier_threshold).astype(np.int)

    # outlier rejection
    data = {
        'corr_pos': torch.from_numpy(corr_pos)[None].to(device).float(),
        'src_keypts': torch.from_numpy(src_keypts2)[None].to(device).float(),
        'tgt_keypts': torch.from_numpy(tgt_keypts2)[None].to(device).float(),
        'testing': True,
    }
    res = model(data)
    gt, pred_labels = labels, res['final_labels']
    precision = precision_score(gt, pred_labels[0].detach().cpu().numpy())
    recall = recall_score(gt, pred_labels[0].detach().cpu().numpy())
    f1 = f1_score(gt, pred_labels[0].detach().cpu().numpy())
    evaluate_metric = TransformationLoss(re_thre=15, te_thre=30)
    gt_trans1 = np.reshape(gt_trans, (1, 4, 4))
    #gt_trans1 = gt_trans.unsqueeze(0)
    src_keypts11 = np.reshape(src_keypts, (1, num_node, 3))

    loss, RR, Re, Te, rmse = evaluate_metric(res['final_trans'], gt_trans1, src_keypts11, tgt_keypts, pred_labels)
    '''R, t = decompose_trans(res['final_trans'])
    gt_trans1 = np.reshape(gt_trans, (1, 4, 4))
    gt_R, gt_t = decompose_trans(gt_trans1)
    recall1 = 0
    RE = torch.tensor(0.0)
    TE = torch.tensor(0.0)
    RMSE = torch.tensor(0.0)
    loss = torch.tensor(0.0)

    re = torch.acos(torch.clamp((torch.trace(R.T[0] @ gt_R[0]) - 1) / 2.0, min=-1, max=1))
    te = torch.sqrt(torch.sum((t - gt_t) ** 2))
    warp_src_keypts = transform(src_keypts, res['final_trans'])
    rmse = torch.norm(warp_src_keypts - tgt_keypts, dim=-1).mean()
    re = re * 180 / np.pi
    te = te * 10
    if te < 30 and re < 15:
        recall1 += 1
    RE += re
    TE += te
    RMSE += rmse

    pred_inliers = torch.where(pred_labels > 0)[0]
    if len(pred_inliers) < 1:
        loss += torch.tensor(0.0).to(res['final_trans'].device)
    else:
        warp_src_keypts = transform(src_keypts, res['final_trans'])
        loss += ((warp_src_keypts - tgt_keypts) ** 2).sum(-1).mean()'''

    print("IP:", precision)
    print("IR:", recall)
    print("f1:", f1)

    print("RR:", RR)
    print("rmse:", rmse)
    print("Re:", Re)
    print("Te:", Te)

    draw_registration_result(raw_src_pcd, raw_tgt_pcd, res['final_trans'][0].detach().cpu().numpy())

    #vis_pair(raw_src_pcd, raw_tgt_pcd, src_keypts, tgt_keypts, corr, labels, res['final_labels'][0].detach().cpu().numpy(), max_corr=None)

    # First plot the original state of the point clouds
    #draw_registration_result(raw_src_pcd, raw_tgt_pcd, np.identity(4))

    # Plot point clouds after registration
    #draw_registration_result(raw_src_pcd, raw_tgt_pcd, res['final_trans'][0].detach().cpu().numpy())


    '''# matching
    distance = np.sqrt(2 - 2 * (src_features @ tgt_features.T) + 1e-6)
    source_idx = np.argmin(distance, axis=1)
    source_dis = np.min(distance, axis=1)
    corr = np.concatenate([np.arange(source_idx.shape[0])[:, None], source_idx[:, None]], axis=-1)
    src_keypts = src_pts[corr[:,0]]
    tgt_keypts = tgt_pts[corr[:,1]]
    corr_pos = np.concatenate([src_keypts, tgt_keypts], axis=-1)
    corr_pos = corr_pos - corr_pos.mean(0)

    gt_trans = np.array([0.9733984910,	-0.0568002321,	0.2219664630,	0.2317160950,
                        0.0501955408,	0.9981157770,	-0.0352888630,	0.1851320300,
                         -0.2235526440,	0.0232083994,	0.9744155100,	-0.2480002010,
                         0.0000000000,	0.0000000000,	0.0000000000,	1.0000000000]).reshape(4, 4)
    # compute the ground truth label
    frag1 = src_keypts[corr[:, 0]]
    frag2 = tgt_keypts[corr[:, 1]]
    frag1_warp = transform(frag1, gt_trans)
    distance = np.sqrt(np.sum(np.power(frag1_warp - frag2, 2), axis=1))
    labels = (distance < 0.1).astype(np.int)

    # outlier rejection
    data = {
            'corr_pos': torch.from_numpy(corr_pos)[None].to(device).float(),
            'src_keypts': torch.from_numpy(src_keypts)[None].to(device).float(),
            'tgt_keypts': torch.from_numpy(tgt_keypts)[None].to(device).float(),
            'testing': True,
            }
    res = model(data)
    vis_pair(raw_src_pcd, raw_tgt_pcd, src_keypts, tgt_keypts, corr, labels, res['final_labels'][0].detach().cpu().numpy(), max_corr=None)

    #First plot the original state of the point clouds
    #draw_registration_result(raw_src_pcd, raw_tgt_pcd, np.identity(4))

    #Plot point clouds after registration
    #draw_registration_result(raw_src_pcd, raw_tgt_pcd, res['final_trans'][0].detach().cpu().numpy())'''
