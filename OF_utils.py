import numpy as np
from skimage.registration import optical_flow_tvl1
import matplotlib.pyplot as plt
from skimage.transform import warp
import torch
from PIL import Image

from routes import CLASS_ID_TYPE, FILE_TYPE, FRAME_KEYWORD, LAB_PATH, VIDEO_FRAMES_PATH
from utils import get_score

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def optical_flow_sklearn(image1, image2):
    # use grascale version of rgb images
    image1 = image1[:, :, 0]
    image2 = image2[:, :, 0]

    # set image value between 0 and 1
    image1 = image1 / np.max(image1)
    image2 = image2 / np.max(image2)

    return optical_flow_tvl1(image1, image2)


def warp_mask_sklearn(frame1, frame2, mask1):
    "frame1, frame2 and mask1 are numpy array"
    # get optical flow from frame2 to frame1
    v, u = optical_flow_sklearn(frame2, frame1)

    nr, nc = mask1.shape
    row_coords, col_coords = np.meshgrid(np.arange(nr),
                                         np.arange(nc),
                                         indexing='ij')

    # calculate mask2 by warping mask1 with the frame optical field
    mask2 = warp(mask1,
                 np.array([row_coords + v, col_coords + u]),
                 mode='constant')

    return mask2


def warp_mask_sklearn_RAFT(mask, flow):
    "frame1, frame2 and mask1 are numpy array"
    # get optical flow from frame2 to frame1
    u, v = flow

    nr, nc = mask.shape
    row_coords, col_coords = np.meshgrid(np.arange(nr),
                                         np.arange(nc),
                                         indexing='ij')

    # calculate mask2 by warping mask1 with the frame optical field
    warped_mask = warp(mask,
                       np.array([row_coords + v, col_coords + u]),
                       mode='constant')

    return warped_mask


def warp_mask_custom(mask, flow):
    flow1_hor, flow1_ver = np.round(flow)
    new_mask = np.zeros_like(mask)

    remap_inds_x = np.zeros_like(mask)
    remap_inds_x = np.arange(mask.shape[0])[:, None]
    remap_inds_x = np.repeat(remap_inds_x, mask.shape[1], axis=1)

    remap_inds_y = np.zeros_like(mask)
    remap_inds_y = np.arange(mask.shape[1])[None]
    remap_inds_y = np.repeat(remap_inds_y, mask.shape[0], axis=0)

    remap_inds_y = remap_inds_y + flow1_hor
    remap_inds_x = remap_inds_x + flow1_ver
    remap_inds_x = remap_inds_x.astype(np.int32)
    remap_inds_y = remap_inds_y.astype(np.int32)

    inds_to_zeros = np.where(remap_inds_y > mask.shape[1] - 1, True, False)
    inds_to_zeros = np.logical_or(inds_to_zeros,
                                  np.where(remap_inds_y < 0, True, False))
    inds_to_zeros = np.logical_or(
        inds_to_zeros, np.where(remap_inds_x > mask.shape[0] - 1, True, False))
    inds_to_zeros = np.logical_or(inds_to_zeros,
                                  np.where(remap_inds_x < 0, True, False))

    remap_inds_x = np.clip(remap_inds_x, 0, mask.shape[0] - 1)
    remap_inds_y = np.clip(remap_inds_y, 0, mask.shape[1] - 1)

    new_mask[remap_inds_x, remap_inds_y] = mask
    new_mask[inds_to_zeros] = 0

    return new_mask


def calc_OF(img1, img2, model):
    img1 = (img1 - np.min(img1)) / (np.max(img1) - np.min(img1))
    img2 = (img2 - np.min(img2)) / (np.max(img2) - np.min(img2))
    img1 = img1 * 2 - 1
    img2 = img2 * 2 - 1
    img1 = torch.Tensor(img1).permute(0, 3, 1, 2)
    img2 = torch.Tensor(img2).permute(0, 3, 1, 2)

    with torch.no_grad():
        list_of_flows = model(img1.to(DEVICE), img2.to(DEVICE))
    predicted_flows = list_of_flows[-1]
    return predicted_flows.cpu().numpy()


def get_all_flows_pos(all_frames, n_final, class_ID, step, raft_model):
    batch_size = 128
    left_imgs = []
    right_imgs = []
    prev_n = []
    all_flows = {}
    n = all_frames[0]
    while n < n_final:
        left_path = VIDEO_FRAMES_PATH + class_ID + "/frame" + str(n) + '.png'
        right_path = VIDEO_FRAMES_PATH + class_ID + "/frame" + str(
            n + step) + '.png'
        img_left = np.array(Image.open(left_path).resize((216, 216)))
        img_right = np.array(Image.open(right_path).resize((216, 216)))

        left_imgs.append(img_left)
        right_imgs.append(img_right)
        prev_n.append(str(n))

        if len(left_imgs) == batch_size:
            left_imgs = np.stack(left_imgs)
            right_imgs = np.stack(right_imgs)
            predicted_flows = calc_OF(left_imgs, right_imgs, raft_model)
            for i, number in enumerate(prev_n):
                all_flows[number] = predicted_flows[i]
            left_imgs = []
            right_imgs = []
            prev_n = []
        n += step

    if len(left_imgs) > 0:
        left_imgs = np.stack(left_imgs)
        right_imgs = np.stack(right_imgs)
        predicted_flows = calc_OF(left_imgs, right_imgs, raft_model)
        for i, n in enumerate(prev_n):
            all_flows[n] = predicted_flows[i]

    return all_flows


def get_all_flows_neg(all_frames, n_final, class_ID, step, raft_model):
    batch_size = 128
    left_imgs = []
    right_imgs = []
    prev_n = []
    all_flows = {}
    n = all_frames[-1]
    while n > n_final:
        right_path = VIDEO_FRAMES_PATH + class_ID + "/frame" + str(n) + '.png'
        left_path = VIDEO_FRAMES_PATH + class_ID + "/frame" + str(
            n - step) + '.png'
        img_left = np.array(Image.open(left_path).resize((216, 216)))
        img_right = np.array(Image.open(right_path).resize((216, 216)))

        left_imgs.append(img_left)
        right_imgs.append(img_right)
        prev_n.append(str(n))

        if len(left_imgs) == batch_size:
            left_imgs = np.stack(left_imgs)
            right_imgs = np.stack(right_imgs)
            predicted_flows = calc_OF(right_imgs, left_imgs, raft_model)
            for i, number in enumerate(prev_n):
                all_flows[number] = predicted_flows[i]
            left_imgs = []
            right_imgs = []
            prev_n = []
        n -= step

    if len(left_imgs) > 0:
        left_imgs = np.stack(left_imgs)
        right_imgs = np.stack(right_imgs)
        predicted_flows = calc_OF(right_imgs, left_imgs, raft_model)
        for i, n in enumerate(prev_n):
            all_flows[n] = predicted_flows[i]

    return all_flows


def compute_flow_mask_pos_raft(class_ID,
                               OF_preds,
                               unlabeled_frames,
                               labeled_frames,
                               raft_model,
                               step=1):

    n_init = int(labeled_frames[0])
    n = n_init
    n_final = unlabeled_frames[-1]

    all_frames = sorted(labeled_frames + unlabeled_frames)
    all_flows = get_all_flows_pos(all_frames, n_final, class_ID, step,
                                  raft_model)

    while n < n_final:
        if n in labeled_frames:
            n_init = n
            mask_path = LAB_PATH + class_ID + '/frame' + str(n) + '.png'
            flow_mask_pos = np.array(
                Image.open(mask_path).convert('L').resize(
                    (216, 216), resample=Image.Resampling.NEAREST))
            flow_mask_pos = flow_mask_pos != 0
            flow_mask_pos = flow_mask_pos.astype('float')

        flow = all_flows[str(n)]
        # flow_mask_pos = warp_mask_sklearn_RAFT(flow_mask_pos, -flow)
        flow_mask_pos = warp_mask_custom(flow_mask_pos, flow)
        if np.max(flow_mask_pos) != 0:
            flow_mask_pos = (flow_mask_pos - np.min(flow_mask_pos)) / (
                np.max(flow_mask_pos) - np.min(flow_mask_pos))

        n += step
        if n in unlabeled_frames:
            key = class_ID + "/frame" + str(n)
            if key not in OF_preds:
                OF_preds[key] = dict()
            flow_mask_pos_save = np.array(
                Image.fromarray(flow_mask_pos).resize(
                    (220, 220), resample=Image.Resampling.NEAREST))
            OF_preds[key]['flow_mask_+'] = flow_mask_pos_save
            OF_preds[key]['+_distance'] = abs(n - n_init)

    return OF_preds


def compute_flow_mask_neg_raft(class_ID,
                               OF_preds,
                               unlabeled_frames,
                               labeled_frames,
                               raft_model,
                               step=1):

    n_init = int(labeled_frames[-1])
    n = n_init
    n_final = unlabeled_frames[0]

    all_frames = sorted(labeled_frames + unlabeled_frames)
    all_flows = get_all_flows_neg(all_frames, n_final, class_ID, step,
                                  raft_model)

    while n > n_final:
        if n in labeled_frames:
            n_init = n
            mask_path = LAB_PATH + class_ID + '/frame' + str(n) + '.png'
            flow_mask_neg = np.array(
                Image.open(mask_path).convert('L').resize(
                    (216, 216), resample=Image.Resampling.NEAREST))
            flow_mask_neg = flow_mask_neg != 0
            flow_mask_neg = flow_mask_neg.astype('float')

        flow = all_flows[str(n)]
        # flow_mask_neg = warp_mask_sklearn_RAFT(flow_mask_neg, -flow)
        flow_mask_neg = warp_mask_custom(flow_mask_neg, flow)
        if np.max(flow_mask_neg) != 0:
            flow_mask_neg = (flow_mask_neg - np.min(flow_mask_neg)) / (
                np.max(flow_mask_neg) - np.min(flow_mask_neg))

        n -= step
        if n in unlabeled_frames:
            key = class_ID + "/frame" + str(n)
            if key not in OF_preds:
                OF_preds[key] = dict()
            flow_mask_neg_save = np.array(
                Image.fromarray(flow_mask_neg).resize(
                    (220, 220), resample=Image.Resampling.NEAREST))
            OF_preds[key]['flow_mask_-'] = flow_mask_neg_save
            OF_preds[key]['-_distance'] = abs(n - n_init)

    return OF_preds


def compute_flow_mask_pos(class_ID,
                          objectives,
                          unlabeled_dataset,
                          unlabeled_frames,
                          labeled_frames,
                          step=1,
                          plot=False):
    n_init = int(labeled_frames[0])
    n = n_init
    n_final = unlabeled_frames[-1]
    while n < n_final:
        if n in labeled_frames:
            # path to all class masks + class ID + .png
            n_init = n
            mask_path = LAB_PATH + class_ID + '/frame' + str(n) + '.png'
            _, flow_mask_pos = unlabeled_dataset.open_path(mask_path,
                                                           mask_path,
                                                           toTensor=False)

        left_path = VIDEO_FRAMES_PATH + class_ID + "/frame" + str(n) + '.png'
        img_left, _ = unlabeled_dataset.open_path(left_path,
                                                  left_path,
                                                  toTensor=False)

        right_path = VIDEO_FRAMES_PATH + class_ID + "/frame" + str(
            n + step) + '.png'
        img_right, _ = unlabeled_dataset.open_path(right_path,
                                                   right_path,
                                                   toTensor=False)
        if plot:
            if n in unlabeled_frames:
                fig, axs = plt.subplots(1, 2, figsize=(10, 5))
                axs[0].imshow(img_left)
                axs[0].imshow(flow_mask_pos, alpha=0.5)
                axs[0].set_title(f'flow mask pos {n}', color='white')
                axs[0].set_xticks([])
                axs[0].set_yticks([])

                mask_path = left[:ind2 + 5] + str(n) + '.png'
                _, true_mask = unlabeled_dataset.open_path(mask_path,
                                                           mask_path,
                                                           toTensor=False)
                axs[1].imshow(img_left)
                axs[1].imshow(true_mask, alpha=0.5)
                axs[1].set_title(f'flow mask pos {n}', color='white')
                axs[1].set_xticks([])
                axs[1].set_yticks([])
                plt.show()
            else:
                plt.imshow(img_left)
                plt.imshow(flow_mask_pos, alpha=0.5)
                plt.title(f'flow mask pos {n}', color='white')
                plt.xticks([])
                plt.yticks([])
                plt.show()

        flow_mask_pos = warp_mask_sklearn(img_left, img_right, flow_mask_pos)
        if np.max(flow_mask_pos) != 0:
            flow_mask_pos = flow_mask_pos / np.max(flow_mask_pos)

        n += step
        if n in unlabeled_frames:
            key = class_ID + "/frame" + str(n)
            if key not in objectives:
                objectives[key] = dict()
            objectives[key]['flow_mask_+'] = flow_mask_pos
            objectives[key]['+_distance'] = abs(n - n_init)

    return objectives


def compute_flow_mask_neg(class_ID,
                          objectives,
                          unlabeled_dataset,
                          unlabeled_frames,
                          labeled_frames,
                          step=1,
                          plot=False):
    n_init = int(labeled_frames[-1])
    n = n_init
    n_final = unlabeled_frames[0]
    while n > n_final:
        if n in labeled_frames:
            # path to all class masks + class ID + .png
            n_init = n
            mask_path = LAB_PATH + class_ID + '/frame' + str(n) + '.png'
            _, flow_mask_neg = unlabeled_dataset.open_path(mask_path,
                                                           mask_path,
                                                           toTensor=False)

        right_path = VIDEO_FRAMES_PATH + class_ID + "/frame" + str(n) + '.png'
        img_right, _ = unlabeled_dataset.open_path(right_path,
                                                   right_path,
                                                   toTensor=False)

        left_path = VIDEO_FRAMES_PATH + class_ID + "/frame" + str(
            n - step) + '.png'
        img_left, _ = unlabeled_dataset.open_path(left_path,
                                                  left_path,
                                                  toTensor=False)
        if plot:
            if n in unlabeled_frames:
                fig, axs = plt.subplots(1, 2, figsize=(10, 5))
                axs[0].imshow(img_right)
                axs[0].imshow(flow_mask_neg, alpha=0.5)
                axs[0].set_title(f'flow mask pos {n}', color='white')
                axs[0].set_xticks([])
                axs[0].set_yticks([])

                mask_path = right[:ind2 + 5] + str(n) + '.png'
                _, true_mask = unlabeled_dataset.open_path(mask_path,
                                                           mask_path,
                                                           toTensor=False)
                axs[1].imshow(img_right)
                axs[1].imshow(true_mask, alpha=0.5)
                axs[1].set_title(f'flow mask pos {n}', color='white')
                axs[1].set_xticks([])
                axs[1].set_yticks([])
                plt.show()
            else:
                plt.imshow(img_right)
                plt.imshow(flow_mask_neg, alpha=0.5)
                plt.title(f'flow mask pos {n}', color='white')
                plt.xticks([])
                plt.yticks([])
                plt.show()

        flow_mask_neg = warp_mask_sklearn(img_right, img_left, flow_mask_neg)
        if np.max(flow_mask_neg) != 0:
            flow_mask_neg = flow_mask_neg / np.max(flow_mask_neg)

        n -= step
        if n in unlabeled_frames:
            key = class_ID + "/frame" + str(n)
            if key not in objectives:
                objectives[key] = dict()
            objectives[key]['flow_mask_-'] = flow_mask_neg
            objectives[key]['-_distance'] = abs(n - n_init)

    return objectives


def get_OF_RAFT(train_dataset,
                unlabeled_dataset,
                ML_preds,
                raft_model,
                step=1,
                metric='DICE'):

    OFvsGT_scores = {}
    MLvsOF_scores = {}
    ind_keyword = len(LAB_PATH + CLASS_ID_TYPE)
    # for each video sequence
    for class_ID, data_paths in train_dataset.data_path.items():
        OF_preds = {}

        ## unlabeled frames as a list of integer
        unlabeled_frames = []
        for _, mask in unlabeled_dataset.data_path[class_ID]:
            unlabeled_frames.append(
                int(mask[ind_keyword + len(FRAME_KEYWORD):-len(FILE_TYPE)]))

        if len(unlabeled_frames) > 0:
            ## labeled frames as a list of integer
            labeled_frames = []
            for _, mask in data_paths:
                labeled_frames.append(
                    int(mask[ind_keyword +
                             len(FRAME_KEYWORD):-len(FILE_TYPE)]))

            ## begin left to right propagation ##
            OF_preds = compute_flow_mask_pos_raft(class_ID,
                                                  OF_preds,
                                                  unlabeled_frames,
                                                  labeled_frames,
                                                  raft_model=raft_model,
                                                  step=step)

            ## begin right to left propagation ##
            OF_preds = compute_flow_mask_neg_raft(class_ID,
                                                  OF_preds,
                                                  unlabeled_frames,
                                                  labeled_frames,
                                                  raft_model=raft_model,
                                                  step=step)

            ## weight final flow with distance weight ##
            for k2, v2 in OF_preds.items():
                if 'flow_mask_+' in v2 and 'flow_mask_-' in v2:
                    dist = v2['+_distance'] + v2['-_distance']
                    pos_weight = 1 - v2['+_distance'] / dist
                    neg_weight = 1 - v2['-_distance'] / dist
                    OF_preds[k2] = pos_weight * v2[
                        'flow_mask_+'] + neg_weight * v2['flow_mask_-']
                elif 'flow_mask_+' not in v2:
                    OF_preds[k2] = v2['flow_mask_-']
                elif 'flow_mask_-' not in v2:
                    OF_preds[k2] = v2['flow_mask_+']

            ## get quality of ML preds against OF mask and quality of OF mask against GT ##
            for k2, v2 in OF_preds.items():
                ML_mask = np.load(ML_preds[k2])
                MLvsOF_score = get_score(v2, ML_mask, metric=metric)
                _, GT = unlabeled_dataset.open_path(LAB_PATH + k2 + '.png',
                                                    LAB_PATH + k2 + '.png',
                                                    toTensor=False)
                OFvsGT_score = get_score(GT, v2, metric=metric)

                MLvsOF_scores[k2] = MLvsOF_score
                OFvsGT_scores[k2] = OFvsGT_score

    return MLvsOF_scores, OFvsGT_scores


def get_OF(train_dataset, unlabeled_dataset, ML_preds, step=1, metric='DICE'):

    OFvsGT_scores = {}
    MLvsOF_scores = {}
    ind_keyword = len(LAB_PATH + CLASS_ID_TYPE)
    # for each video sequence
    for class_ID, data_paths in train_dataset.data_path.items():
        OF_preds = {}

        ## unlabeled frames as a list of integer
        unlabeled_frames = []
        for _, mask in unlabeled_dataset.data_path[class_ID]:
            unlabeled_frames.append(
                int(mask[ind_keyword + len(FRAME_KEYWORD):-len(FILE_TYPE)]))

        if len(unlabeled_frames) > 0:
            ## labeled frames as a list of integer
            labeled_frames = []
            for _, mask in data_paths:
                labeled_frames.append(
                    int(mask[ind_keyword +
                             len(FRAME_KEYWORD):-len(FILE_TYPE)]))

            ## define all mask path, class Id and frame number
            OF_preds = compute_flow_mask_pos(class_ID,
                                             OF_preds,
                                             unlabeled_dataset,
                                             unlabeled_frames,
                                             labeled_frames,
                                             step=step)

            ## define all mask path, class Id and frame number
            OF_preds = compute_flow_mask_neg(class_ID,
                                             OF_preds,
                                             unlabeled_dataset,
                                             unlabeled_frames,
                                             labeled_frames,
                                             step=step)

            for k2, v2 in OF_preds.items():
                if 'flow_mask_+' in v2 and 'flow_mask_-' in v2:
                    dist = v2['+_distance'] + v2['-_distance']
                    pos_weight = 1 - v2['+_distance'] / dist
                    neg_weight = 1 - v2['-_distance'] / dist
                    OF_preds[k2] = pos_weight * v2[
                        'flow_mask_+'] + neg_weight * v2['flow_mask_-']
                elif 'flow_mask_+' not in v2:
                    OF_preds[k2] = v2['flow_mask_-']
                elif 'flow_mask_-' not in v2:
                    OF_preds[k2] = v2['flow_mask_+']

            for k2, v2 in OF_preds.items():
                ML_mask = np.load(ML_preds[k2])
                MLvsOF_score = get_score(v2, ML_mask, metric=metric)
                _, GT = unlabeled_dataset.open_path(LAB_PATH + k2 + '.png',
                                                    LAB_PATH + k2 + '.png',
                                                    toTensor=False)
                OFvsGT_score = get_score(GT, v2, metric=metric)

                MLvsOF_scores[k2] = MLvsOF_score
                OFvsGT_scores[k2] = OFvsGT_score

    return MLvsOF_scores, OFvsGT_scores


def update_OF_score_RAFT(MLvsOF_scores,
                         class_ID,
                         all_frames,
                         ML_preds,
                         new_labeled,
                         curr_labeled,
                         raft_model,
                         step=1,
                         metric='DICE'):

    # for each video sequence
    OF_preds = {}
    labeled_frames = new_labeled[class_ID] + curr_labeled[class_ID]
    labeled_frames = sorted([int(nb) for nb in labeled_frames])
    unlabeled_frames = [
        int(nb) for nb in all_frames[class_ID] if int(nb) not in labeled_frames
    ]

    if len(unlabeled_frames) > 0:
        OF_preds = compute_flow_mask_pos_raft(class_ID,
                                              OF_preds,
                                              unlabeled_frames,
                                              labeled_frames,
                                              raft_model=raft_model,
                                              step=step)

        OF_preds = compute_flow_mask_neg_raft(class_ID,
                                              OF_preds,
                                              unlabeled_frames,
                                              labeled_frames,
                                              raft_model=raft_model,
                                              step=step)

        for k2, v2 in OF_preds.items():
            if 'flow_mask_+' in v2 and 'flow_mask_-' in v2:
                dist = v2['+_distance'] + v2['-_distance']
                pos_weight = 1 - v2['+_distance'] / dist
                neg_weight = 1 - v2['-_distance'] / dist
                OF_preds[k2] = pos_weight * v2[
                    'flow_mask_+'] + neg_weight * v2['flow_mask_-']
            elif 'flow_mask_+' not in v2:
                OF_preds[k2] = v2['flow_mask_-']
            elif 'flow_mask_-' not in v2:
                OF_preds[k2] = v2['flow_mask_+']

        for k2, v2 in OF_preds.items():
            ML_mask = np.load(ML_preds[k2])
            MLvsOF_score = get_score(v2, ML_mask, metric=metric)
            MLvsOF_scores[k2] = MLvsOF_score

    return MLvsOF_scores


def update_OF_score(MLvsOF_scores,
                    class_ID,
                    all_frames,
                    unlabeled_dataset,
                    ML_preds,
                    new_labeled,
                    curr_labeled,
                    step=1,
                    metric='DICE'):
    # for each video sequence
    OF_preds = {}
    labeled_frames = new_labeled[class_ID] + curr_labeled[class_ID]
    labeled_frames = sorted([int(nb) for nb in labeled_frames])
    unlabeled_frames = [
        int(nb) for nb in all_frames[class_ID] if int(nb) not in labeled_frames
    ]

    if len(unlabeled_frames) > 0:
        OF_preds = compute_flow_mask_pos(class_ID,
                                         OF_preds,
                                         unlabeled_dataset,
                                         unlabeled_frames,
                                         labeled_frames,
                                         step=step)

        OF_preds = compute_flow_mask_neg(class_ID,
                                         OF_preds,
                                         unlabeled_dataset,
                                         unlabeled_frames,
                                         labeled_frames,
                                         step=step)

        for k2, v2 in OF_preds.items():
            if 'flow_mask_+' in v2 and 'flow_mask_-' in v2:
                dist = v2['+_distance'] + v2['-_distance']
                pos_weight = 1 - v2['+_distance'] / dist
                neg_weight = 1 - v2['-_distance'] / dist
                OF_preds[k2] = pos_weight * v2[
                    'flow_mask_+'] + neg_weight * v2['flow_mask_-']
            elif 'flow_mask_+' not in v2:
                OF_preds[k2] = v2['flow_mask_-']
            elif 'flow_mask_-' not in v2:
                OF_preds[k2] = v2['flow_mask_+']

        for k2, v2 in OF_preds.items():
            ML_mask = np.load(ML_preds[k2])
            MLvsOF_score = get_score(v2, ML_mask, metric=metric)
            MLvsOF_scores[k2] = MLvsOF_score

    return MLvsOF_scores
