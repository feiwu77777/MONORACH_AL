# from torchvision.models.optical_flow import raft_large
import torch
import collections
import json
import numpy as np

from routes import CLASS_ID_TYPE, FILE_TYPE, FRAME_KEYWORD, IMG_PATH, PRINT_PATH
from utils import embedding_similarity
from OF_utils import get_OF, get_OF_RAFT, update_OF_score, update_OF_score_RAFT

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
THRESH = 0.5


def density_OF_query(ML_preds,
                     curr_labeled,
                     train_dataset,
                     unlabeled_dataset,
                     num_query,
                     n_round,
                     SEED,
                     step=1,
                     metric='DICE'):

    MLvsOF_scores, OFvsGT_scores = get_OF(train_dataset,
                                          unlabeled_dataset,
                                          ML_preds,
                                          step=step,
                                          metric=metric)
    MLvsOF_scores_per_video = collections.defaultdict(list)
    for k, v in MLvsOF_scores.items():
        classID = k[:len(CLASS_ID_TYPE) - 1]
        number = k[len(CLASS_ID_TYPE) + len(FRAME_KEYWORD):]

        MLvsOF_scores_per_video[classID].append((k, v))

    for k, v in MLvsOF_scores_per_video.items():
        MLvsOF_scores_per_video[k] = sorted(v, key=lambda x: x[1])

    with open(f'results/MLvsOF_scores_SEED={SEED}_round={n_round}.json',
              'w') as f:
        json.dump(MLvsOF_scores, f)
    with open(f'results/OFvsGT_scores_SEED={SEED}_round={n_round}.json',
              'w') as f:
        json.dump(OFvsGT_scores, f)

    count = 0
    selected = {}
    ind_per_class = {}
    for k, v in MLvsOF_scores_per_video.items():
        selected[k] = np.zeros(len(v), dtype=bool)
        ind_per_class[k] = 0
    new_labeled = collections.defaultdict(list)

    # prioritize videos with lowest OF score and videos that has less selected frames
    lengths = np.unique([len(v) for v in curr_labeled.values()])
    assert len(lengths) <= 2
    l1 = min(lengths)
    l2 = max(lengths)
    viv1 = []
    viv2 = []
    for k in MLvsOF_scores_per_video.keys():
        if len(curr_labeled[k]) == l1:
            viv1.append(MLvsOF_scores_per_video[k][0])
        elif l2 != l1 and len(curr_labeled[k]) == l2:
            viv2.append(MLvsOF_scores_per_video[k][0])
    viv = sorted(viv1, key=lambda x: x[1]) + sorted(viv2, key=lambda x: x[1])
    viv = [path for path, score in viv]

    while True:
        for img_name in viv:
            class_name = img_name[:len(CLASS_ID_TYPE) - 1]
            frames = MLvsOF_scores_per_video[class_name]
            stay = True
            thresh = THRESH
            while stay:
                ind = ind_per_class[class_name]
                img_name, score = frames[ind]
                number = img_name[len(CLASS_ID_TYPE) + len(FRAME_KEYWORD):]
                # new_path = IMG_PATH + img_name + ".png"
                # if addSample(new_labeled, curr_labeled, unlabeled_dataset,
                #              new_path, class_name, IMG_PATH, thresh):
                new_labeled[class_name].append(number)
                count += 1
                selected[class_name][ind] = True
                stay = False

                ind_per_class[class_name] = ind + 1
                if ind_per_class[class_name] == len(frames):
                    ind_per_class[class_name] = 0
                    thresh += 0.05
                    with open(PRINT_PATH, "a") as f:
                        f.write(
                            f"=== video {class_name} thresh is now: {thresh}\n"
                        )
                if count == num_query:
                    np.save(
                        f'results/selected_SEED={SEED}_round={n_round}.npy',
                        selected)
                    return new_labeled


def RAFTxSim_query(curr_labeled,
                   train_dataset,
                   unlabeled_dataset,
                   ML_preds,
                   num_query,
                   n_round,
                   SEED,
                   step=1,
                   metric='DICE'):

    count = 0
    new_labeled = collections.defaultdict(list)
    all_frames = collections.defaultdict(list)
    for k, v in ML_preds.items():
        class_name = k[:len(CLASS_ID_TYPE) - 1]
        number = k[len(CLASS_ID_TYPE) + len(FRAME_KEYWORD):]
        all_frames[class_name].append(number)

    ################# Sample with MLvsOF and similarity RAFT ####################
    raft_model = raft_large(pretrained=True, progress=False).to(DEVICE)
    raft_model = raft_model.eval()
    MLvsOF_scores, OFvsGT_scores = get_OF_RAFT(train_dataset,
                                               unlabeled_dataset,
                                               ML_preds,
                                               raft_model=raft_model,
                                               step=step,
                                               metric=metric)
    with open(f'results/MLvsOF_scores_SEED={SEED}_round={n_round}.json',
              'w') as f:
        json.dump(MLvsOF_scores, f)
    with open(f'results/OFvsGT_scores_SEED={SEED}_round={n_round}.json',
              'w') as f:
        json.dump(OFvsGT_scores, f)

    labeled_frames = list(train_dataset.data_pool[:, 0])
    unlabeled_frames = list(unlabeled_dataset.data_pool[:, 0])

    while count < num_query:
        distances, names = embedding_similarity(labeled_frames,
                                                unlabeled_frames)
        distances = (distances - torch.min(distances)) / (
            torch.max(distances) - torch.min(distances))

        dist_dict = {}
        for i, n in enumerate(names):
            dist_dict[n] = distances[i].item()

        with open(f'results/embedding_dist_SEED={SEED}_round={n_round}.json',
                  'w') as f:
            json.dump(dist_dict, f)

        selected = None
        max_score = 0
        for k, v in dist_dict.items():
            score = 2 * (v * (1 - MLvsOF_scores[k])) / (v +
                                                        (1 - MLvsOF_scores[k]))
            if score > max_score:
                max_score = score
                selected = k

        class_name = selected[:len(CLASS_ID_TYPE) - 1]
        number = selected[len(CLASS_ID_TYPE) + len(FRAME_KEYWORD):]

        new_path = IMG_PATH + selected + FILE_TYPE
        labeled_frames.append(new_path)
        prev_length = len(unlabeled_frames)
        unlabeled_frames.remove(new_path)
        assert len(unlabeled_frames) == prev_length - 1

        new_labeled[class_name].append(number)

        MLvsOF_scores = update_OF_score_RAFT(MLvsOF_scores,
                                             class_name,
                                             all_frames,
                                             ML_preds,
                                             new_labeled,
                                             curr_labeled,
                                             raft_model=raft_model,
                                             step=step,
                                             metric=metric)
        count += 1

    return new_labeled


def RAFT_query(curr_labeled,
               train_dataset,
               unlabeled_dataset,
               ML_preds,
               num_query,
               n_round,
               SEED,
               step=1,
               metric='DICE'):

    count = 0
    new_labeled = collections.defaultdict(list)
    all_frames = collections.defaultdict(list)
    for k, v in ML_preds.items():
        class_name = k[:len(CLASS_ID_TYPE) - 1]
        number = k[len(CLASS_ID_TYPE) + len(FRAME_KEYWORD):]
        all_frames[class_name].append(number)

    ############### Sample with only MLvsOF without similarity RAFT ################
    raft_model = raft_large(pretrained=True, progress=False).to(DEVICE)
    raft_model = raft_model.eval()
    MLvsOF_scores, OFvsGT_scores = get_OF_RAFT(train_dataset,
                                               unlabeled_dataset,
                                               ML_preds,
                                               raft_model=raft_model,
                                               step=step,
                                               metric=metric)
    with open(f'results/MLvsOF_scores_SEED={SEED}_round={n_round}.json',
              'w') as f:
        json.dump(MLvsOF_scores, f)
    with open(f'results/OFvsGT_scores_SEED={SEED}_round={n_round}.json',
              'w') as f:
        json.dump(OFvsGT_scores, f)

    while count < num_query:
        candidates = sorted([(k, v) for k, v in MLvsOF_scores.items()],
                            key=lambda x: x[1])
        for selected, score in candidates:
            class_name = selected[:len(CLASS_ID_TYPE) - 1]
            number = selected[len(CLASS_ID_TYPE) + len(FRAME_KEYWORD):]
            if number not in new_labeled[class_name]:
                new_labeled[class_name].append(number)
                break

        MLvsOF_scores = update_OF_score_RAFT(MLvsOF_scores,
                                             class_name,
                                             all_frames,
                                             ML_preds,
                                             new_labeled,
                                             curr_labeled,
                                             raft_model=raft_model,
                                             step=step,
                                             metric=metric)
        count += 1

    return new_labeled


def OF_query(curr_labeled,
             train_dataset,
             unlabeled_dataset,
             ML_preds,
             num_query,
             n_round,
             SEED,
             step=1,
             metric='DICE'):

    count = 0
    new_labeled = collections.defaultdict(list)
    all_frames = collections.defaultdict(list)
    for k, v in ML_preds.items():
        class_name = k[:len(CLASS_ID_TYPE) - 1]
        number = k[len(CLASS_ID_TYPE) + len(FRAME_KEYWORD):]
        all_frames[class_name].append(number)

    ############### Sample with only MLvsOF without similarity ################
    MLvsOF_scores, OFvsGT_scores = get_OF(train_dataset,
                                          unlabeled_dataset,
                                          ML_preds,
                                          step=step,
                                          metric=metric)
    with open(f'results/MLvsOF_scores_SEED={SEED}_round={n_round}.json',
              'w') as f:
        json.dump(MLvsOF_scores, f)
    with open(f'results/OFvsGT_scores_SEED={SEED}_round={n_round}.json',
              'w') as f:
        json.dump(OFvsGT_scores, f)

    while count < num_query:
        candidates = sorted([(k, v) for k, v in MLvsOF_scores.items()],
                            key=lambda x: x[1])
        for selected, score in candidates:
            class_name = selected[:len(CLASS_ID_TYPE) - 1]
            number = selected[len(CLASS_ID_TYPE) + len(FRAME_KEYWORD):]
            if number not in new_labeled[class_name]:
                new_labeled[class_name].append(number)
                break

        MLvsOF_scores = update_OF_score(MLvsOF_scores,
                                        class_name,
                                        all_frames,
                                        unlabeled_dataset,
                                        ML_preds,
                                        new_labeled,
                                        curr_labeled,
                                        step=step,
                                        metric=metric)
        count += 1

    return new_labeled
