from torch.utils.data import DataLoader
import torch
import collections
import numpy as np
import json
import os
from sklearn.cluster import KMeans
import torch.nn.functional as F
from custom_Kmeans import CustomKMeans

from routes import CLASS_ID_CUT, CLASS_ID_TYPE, FILE_TYPE, FRAME_KEYWORD, IMG_PATH, LAB_PATH, PRINT_PATH
from utils import embedding_similarity, euc_distance, resnet_embedding, simCLR_embedding, center_diff, average_center
from kmeans_utils import get_fixed_clusters, CustomKMeans_simpler, CustomKMedian

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
THRESH = 0.5


def random_query(unlabeled_dataset, num_query):
    all_names = list(unlabeled_dataset.data_pool[:, 1])
    np.random.shuffle(all_names)

    new_labeled = collections.defaultdict(list)
    ind_keyword = len(LAB_PATH + CLASS_ID_TYPE)
    for n in all_names[:num_query]:
        class_name = n[ind_keyword - len(CLASS_ID_TYPE):ind_keyword -
                       len(CLASS_ID_CUT)]
        number = n[ind_keyword + len(FRAME_KEYWORD):-len(FILE_TYPE)]
        new_labeled[class_name].append(number)

    return new_labeled


def density_query(train_dataset, unlabeled_dataset, num_query, n_round, SEED):
    unlabeled_frames = list(unlabeled_dataset.data_pool[:, 1])
    labeled_frames = list(train_dataset.data_pool[:, 1])

    ind_keyword = len(LAB_PATH + CLASS_ID_TYPE)
    labeled_distances = collections.defaultdict(list)
    for frame in labeled_frames:
        classId = frame[ind_keyword - len(CLASS_ID_TYPE):ind_keyword - 1]
        frameNb = int(frame[ind_keyword + len(FRAME_KEYWORD):-len(FILE_TYPE)])
        labeled_distances[classId].append(frameNb)

    unlabeled_distances = collections.defaultdict(list)
    for frame in unlabeled_frames:
        classId = frame[ind_keyword - len(CLASS_ID_TYPE):ind_keyword - 1]
        frameNb = int(frame[ind_keyword + len(FRAME_KEYWORD):-len(FILE_TYPE)])
        unlabeled_distances[classId].append(frameNb)
        if classId not in labeled_distances:
            labeled_distances[classId] = []

    new_labeled = collections.defaultdict(list)
    n_count = 0
    keys = sorted([(k, v) for k, v in labeled_distances.items()],
                  key=lambda x: len(x[1]))
    keys = [k for k, v in keys]
    while n_count < num_query:
        for k in keys:
            v = labeled_distances[k]
            D = collections.defaultdict(list)
            unlab_video = unlabeled_distances[k]

            if len(v) == 0:
                mid = int(len(unlab_video) / 2)
                new_labeled[k].append(str(unlab_video[mid]).zfill(5))
                labeled_distances[k].append(unlab_video[mid])
                n_count += 1
            
            else:
                for dist in unlab_video:
                    for labeled_dist in v:
                        D[dist].append(abs(dist - labeled_dist))

                max_k = None
                max_v = 0
                for k2, v2 in D.items():
                    D[k2] = min(v2)
                    if D[k2] > max_v:
                        max_v = D[k2]
                        max_k = k2
                if max_k is not None:
                    new_labeled[k].append(str(max_k).zfill(5))
                    labeled_distances[k].append(max_k)
                    n_count += 1

            if n_count == num_query:
                return new_labeled

    return new_labeled


def entropy_query(copy_model,
                  unlabeled_dataset,
                  num_query,
                  n_round,
                  SEED,
                  smooth=1e-7):
    ML_entropy = {}
    copy_model.eval()
    unlabeled_dataloader = DataLoader(unlabeled_dataset,
                                      batch_size=128,
                                      shuffle=False)
    with torch.no_grad():
        for img, label, names in unlabeled_dataloader:
            img, label = img.to(DEVICE), label.to(DEVICE)
            pred = copy_model(img)
            proba_pos = torch.sigmoid(pred).cpu().squeeze(1)
            proba_neg = 1 - proba_pos
            all_proba = torch.stack((proba_pos, proba_neg), axis=-1)

            log_proba = torch.log(all_proba + smooth)
            entropy = (all_proba * log_proba).sum(-1)

            for i, name in enumerate(names):
                ML_entropy[name] = torch.mean(entropy[i]).item()

    # didnt put the "-" sign in entropy so we just sort from smaller
    # to bigger instead of bigger to smaller with the "-" sign
    ML_entropy = sorted([(k, v) for k, v in ML_entropy.items()],
                        key=lambda x: x[1])
    with open(f'results/ML_entropy_SEED={SEED}_round={n_round}.json',
              'w') as f:
        json.dump(ML_entropy, f)
    count = 0
    new_labeled = collections.defaultdict(list)
    for selected, score in ML_entropy:
        class_name = selected[:len(CLASS_ID_TYPE) - 1]
        number = selected[len(CLASS_ID_TYPE) + len(FRAME_KEYWORD):]
        new_labeled[class_name].append(number)
        count += 1
        if count == num_query:
            break
    return new_labeled


def density_entropy_query(copy_model,
                          curr_labeled,
                          unlabeled_dataset,
                          num_query,
                          n_round,
                          SEED,
                          smooth=1e-7):
    ML_entropy = {}
    copy_model.eval()
    unlabeled_dataloader = DataLoader(unlabeled_dataset,
                                      batch_size=128,
                                      shuffle=False)
    with torch.no_grad():
        for img, label, names in unlabeled_dataloader:
            img, label = img.to(DEVICE), label.to(DEVICE)
            pred = copy_model(img)
            proba_pos = torch.sigmoid(pred).cpu().squeeze(1)
            proba_neg = 1 - proba_pos
            all_proba = torch.stack((proba_pos, proba_neg), axis=-1)

            log_proba = torch.log(all_proba + smooth)
            entropy = (all_proba * log_proba).sum(-1)

            for i, name in enumerate(names):
                ML_entropy[name] = torch.mean(entropy[i]).item()

    ML_entropy_per_video = collections.defaultdict(list)
    for k, v in ML_entropy.items():
        classID = k[:len(CLASS_ID_TYPE) - 1]
        number = k[len(CLASS_ID_TYPE) + len(FRAME_KEYWORD):]
        ML_entropy_per_video[classID].append((k, v))

    for k, v in ML_entropy_per_video.items():
        ML_entropy_per_video[k] = sorted(v, key=lambda x: x[1])

    with open(f'results/ML_entropy_SEED={SEED}_round={n_round}.json',
              'w') as f:
        json.dump(ML_entropy, f)

    count = 0
    selected = {}
    ind_per_class = {}
    for k, v in ML_entropy_per_video.items():
        selected[k] = np.zeros(len(v), dtype=bool)
        ind_per_class[k] = 0
    new_labeled = collections.defaultdict(list)

    # prioritize videos with lowest entropy score and videos that has less selected frames
    lengths = np.unique([len(v) for v in curr_labeled.values()])
    lengths = sorted(lengths)
    if len(lengths) > 2:
        l1 = lengths[-2]
        l2 = lengths[-1]
    else:
        l1 = min(lengths)
        l2 = max(lengths)
    viv1 = []
    viv2 = []
    for k in ML_entropy_per_video.keys():
        if len(curr_labeled[k]) == l1:
            viv1.append(ML_entropy_per_video[k][0])
        elif l2 != l1 and len(curr_labeled[k]) == l2:
            viv2.append(ML_entropy_per_video[k][0])
    viv = sorted(viv1, key=lambda x: x[1]) + sorted(viv2, key=lambda x: x[1])
    viv = [path for path, score in viv]

    while True:
        for img_name in viv:
            class_name = img_name[:len(CLASS_ID_TYPE) - 1]
            frames = ML_entropy_per_video[class_name]
            stay = True
            thresh = THRESH
            while stay:
                ind = ind_per_class[class_name]
                img_name, score = frames[ind]
                number = img_name[len(CLASS_ID_TYPE) + len(FRAME_KEYWORD):]
                # new_path = IMG_PATH + img_name + ".jpg"
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


def similarity_query(train_dataset, unlabeled_dataset, num_query):
    count = 0
    new_labeled = collections.defaultdict(list)

    labeled_frames = list(train_dataset.data_pool[:, 0])
    unlabeled_frames = list(unlabeled_dataset.data_pool[:, 0])

    while count < num_query:
        distances, names = embedding_similarity(labeled_frames,
                                                unlabeled_frames)
        ind = torch.argmax(distances).item()
        selected = names[ind]

        class_ID = selected[:len(CLASS_ID_TYPE) - 1]
        number = selected[len(CLASS_ID_TYPE) + len(FRAME_KEYWORD):]

        new_path = IMG_PATH + selected + FILE_TYPE
        labeled_frames.append(new_path)
        prev_length = len(unlabeled_frames)
        unlabeled_frames.remove(new_path)
        assert len(unlabeled_frames) == prev_length - 1

        new_labeled[class_ID].append(number)
        count += 1

    return new_labeled


def GT_query(num_query, n_round, SEED):
    ##### sample with MLvsGT and no similarity #########
    count = 0
    new_labeled = collections.defaultdict(list)
    with open(f'results/MLvsGT_scores_SEED={SEED}_round={n_round}.json',
              'r') as f:
        MLvsGT_scores = json.load(f)
    candidates = sorted([(k, v) for k, v in MLvsGT_scores.items()],
                        key=lambda x: x[1])
    for selected, score in candidates:
        class_name = selected[:len(CLASS_ID_TYPE) - 1]
        number = selected[len(CLASS_ID_TYPE) + len(FRAME_KEYWORD):]
        new_labeled[class_name].append(number)
        count += 1
        if count == num_query:
            break
    return new_labeled


def GTxSim_query(train_dataset, unlabeled_dataset, num_query, n_round, SEED):
    count = 0
    new_labeled = collections.defaultdict(list)
    ##### sample with MLvsGT and similarity #########
    with open(f'results/MLvsGT_scores_SEED={SEED}_round={n_round}.json',
              'r') as f:
        MLvsGT_scores = json.load(f)
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
            score = 2 * (v * (1 - MLvsGT_scores[k])) / (v +
                                                        (1 - MLvsGT_scores[k]))
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
        count += 1

    return new_labeled

def k_means_entropy_query(copy_model,
                          unlabeled_dataset,
                          num_query,
                          n_cluster,
                          n_round,
                          SEED,
                          smooth=1e-7, 
                          embedding_method='resnet',
                          weight_path=None):
    ML_entropy = {}
    copy_model.eval()
    unlabeled_dataloader = DataLoader(unlabeled_dataset,
                                      batch_size=64,
                                      shuffle=False)

    if not os.path.isdir('results/embeddings/'):
        os.mkdir('results/embeddings/')

    with torch.no_grad():
        for img, label, names in unlabeled_dataloader:
            img, label = img.to(DEVICE), label.to(DEVICE)
            pred = copy_model(img)
            proba_pos = torch.sigmoid(pred).cpu().squeeze(1)
            proba_neg = 1 - proba_pos
            all_proba = torch.stack((proba_pos, proba_neg), axis=-1)

            log_proba = torch.log(all_proba + smooth)
            entropy = (all_proba * log_proba).sum(-1)

            for i, name in enumerate(names):
                entropy_value = torch.mean(entropy[i]).item()
                ML_entropy[name] = entropy_value
    with open(f'results/ML_entropy_SEED={SEED}_round={n_round}.json',
              'w') as f:
        json.dump(ML_entropy, f)

    # {classID/frame000: array([]), ...}
    if n_round == 0 and embedding_method == 'resnet':
        embeddings = resnet_embedding(unlabeled_dataloader)
        torch.save(embeddings,
                   f'results/embeddings/embeddings_SEED={SEED}.pth')
        selected = {str(i): [] for i in range(n_cluster)}
    elif n_round == 0 and embedding_method == 'simCLR':
        embeddings = simCLR_embedding(unlabeled_dataloader, weight_path=weight_path)
        torch.save(embeddings,
                   f'results/embeddings/embeddings_SEED={SEED}.pth')
        selected = {str(i): [] for i in range(n_cluster)}
    else:
        embeddings = torch.load(
            f'results/embeddings/embeddings_SEED={SEED}.pth')
        with open(
                f'results/embeddings/selected_SEED={SEED}_round={n_round - 1}.json',
                'r') as f:
            selected = json.load(f)

    kmeans = KMeans(n_clusters=n_cluster, random_state=0)
    kmeans.fit(np.stack(list(embeddings.values())))
    labels = kmeans.labels_

    cluster_scores = collections.defaultdict(list)
    for i, k in enumerate(embeddings.keys()):
        if k in ML_entropy:
            label = labels[i]
            cluster_scores[label].append((k, ML_entropy[k]))
    for k, v in cluster_scores.items():
        cluster_scores[k] = sorted(v, key=lambda x: x[1])

    lengths = np.unique([len(v) for v in selected.values()])
    if len(lengths) > 2:
        l1 = sorted(lengths)[-2]
    else:
        l1 = min(lengths)
    l2 = max(lengths)
    viv1 = []
    viv2 = []
    for k, v in cluster_scores.items():
        selected_frames = selected[str(k)]
        if len(selected_frames) == l1:
            for name, score in v:
                if name not in selected_frames:
                    viv1.append((name, score, k))
                    break
        elif l2 != l1 and len(selected_frames) == l2:
            for name, score in v:
                if name not in selected_frames:
                    viv2.append((name, score, k))
                    break
    viv = sorted(viv1, key=lambda x: x[1]) + sorted(viv2, key=lambda x: x[1])

    # {classID: ['000', ...], ...}
    new_labeled = collections.defaultdict(list)
    for i in range(num_query):
        name, score, cluster = viv[i]
        class_name = name[:len(CLASS_ID_TYPE) - 1]
        number = name[len(CLASS_ID_TYPE) + len(FRAME_KEYWORD):]
        new_labeled[class_name].append(number)

        selected[str(cluster)].append(name)

    with open(f'results/embeddings/selected_SEED={SEED}_round={n_round}.json',
              'w') as f:
        json.dump(selected, f)
    return new_labeled

def k_means_fulldataset_center_query(copy_model,
                                     train_dataset,
                                     unlabeled_dataset,
                                     all_train_dataset,
                                     num_query,
                                     n_round,
                                     SEED,
                                     smooth=1e-7,
                                     embedding_method='resnet', 
                                     weight_path='../pretrained_models/auris_seg_simCLR/checkpoint.pth',
                                     sphere=False,
                                     use_kmedian=False):
    copy_model.eval()
    all_train_dataloader = DataLoader(all_train_dataset,
                                      batch_size=64,
                                      shuffle=False)

    if not os.path.isdir('results/embeddings/'):
        os.mkdir('results/embeddings/')

    if n_round == 0 and embedding_method == 'resnet':
        embeddings = resnet_embedding(all_train_dataloader)
        torch.save(embeddings,
                   f'results/embeddings/embeddings_SEED={SEED}.pth')
    elif n_round == 0 and embedding_method == 'CARL':
        embeddings = CARL_embedding(all_train_dataset, weight_path=weight_path)
        torch.save(embeddings,
                   f'results/embeddings/embeddings_SEED={SEED}.pth')
    elif n_round == 0 and embedding_method == 'simCLR':
        embeddings = simCLR_embedding(all_train_dataloader, weight_path=weight_path)
        torch.save(embeddings,
                   f'results/embeddings/embeddings_SEED={SEED}.pth')
    else:
        embeddings = torch.load(
            f'results/embeddings/embeddings_SEED={SEED}.pth')

    if sphere:
        for k, v in embeddings.items():
            embeddings[k] = F.normalize(v, dim=0)

    if use_kmedian:
        centers, _ = CustomKMedian(embeddings, n_clusters=len(train_dataset) + num_query, random_state=0)
    else:
        kmeans = CustomKMeans(n_clusters=len(train_dataset) + num_query, random_state=0, sphere=sphere)
        kmeans.fit(np.stack(list(embeddings.values())))
        centers = kmeans.cluster_centers_
        # centers, _ = CustomKMeans_simpler(embeddings, n_clusters=len(train_dataset) + num_query, random_state=0, sphere=sphere)


    labeled_frames = [
        path[len(IMG_PATH):-4] for path, label_path in train_dataset.data_pool
    ]
    labeled_embedding = {}
    for k in labeled_frames:
        labeled_embedding[k] = embeddings[k]

    fixed_cluster, centers = get_fixed_clusters(centers, labeled_embedding)

    if use_kmedian:
        new_centers, k_means_centers_name = CustomKMedian(embeddings=embeddings, centers=centers, fixed_cluster=fixed_cluster)
    else:
        new_centers, k_means_centers_name = CustomKMeans_simpler(embeddings=embeddings, centers=centers, fixed_cluster=fixed_cluster, sphere=sphere)

    new_labeled = collections.defaultdict(list)
    for i, center in enumerate(new_centers):
        if i in fixed_cluster:
            continue
        min_ = 100000
        name = None
        for k, v in embeddings.items():
            if k not in k_means_centers_name[i]:
                continue
            dist = euc_distance(torch.tensor(center), v).item()
            if dist < min_:
                min_ = dist
                name = k
        class_name = name[:len(CLASS_ID_TYPE) - 1]
        number = name[len(CLASS_ID_TYPE) + len(FRAME_KEYWORD):]
        new_labeled[class_name].append(number)

    k_means_name_centers = {}
    for k, v in k_means_centers_name.items():
        for name in v:
            k_means_name_centers[name] = k
    with open(f'results/k_means_name_centers_SEED={SEED}_round={n_round}.json', 'w') as f:
        json.dump(k_means_name_centers, f)
    with open(f'results/fixed_cluster_SEED={SEED}_round={n_round}.json', 'w') as f:
        json.dump(fixed_cluster, f)
        
    return new_labeled

def k_means_fulldataset_entropy_query(copy_model,
                                      train_dataset,
                                      unlabeled_dataset,
                                      all_train_dataset,
                                      num_query,
                                      n_round,
                                      SEED,
                                      smooth=1e-7,
                                      embedding_method='resnet',
                                      weight_path='../pretrained_models/skateboard_carl/checkpoint.pth'):
    ML_entropy = {}
    copy_model.eval()
    unlabeled_dataloader = DataLoader(unlabeled_dataset,
                                      batch_size=64,
                                      shuffle=False)
    all_train_dataloader = DataLoader(all_train_dataset,
                                      batch_size=64,
                                      shuffle=False)

    with torch.no_grad():
        for img, label, names in unlabeled_dataloader:
            img, label = img.to(DEVICE), label.to(DEVICE)
            pred = copy_model(img)
            proba_pos = torch.sigmoid(pred).cpu().squeeze(1)
            proba_neg = 1 - proba_pos
            all_proba = torch.stack((proba_pos, proba_neg), axis=-1)

            log_proba = torch.log(all_proba + smooth)
            entropy = (all_proba * log_proba).sum(-1)

            for i, name in enumerate(names):
                entropy_value = torch.mean(entropy[i]).item()
                ML_entropy[name] = -entropy_value
    with open(f'results/ML_entropy_SEED={SEED}_round={n_round}.json',
              'w') as f:
        json.dump(ML_entropy, f)

    if not os.path.isdir(f'results/embeddings'):
        os.mkdir(f'results/embeddings')

    if n_round == 0 and embedding_method == 'resnet':
        embeddings = resnet_embedding(all_train_dataloader)
        torch.save(embeddings,
                   f'results/embeddings/embeddings_SEED={SEED}.pth')
    elif n_round == 0 and embedding_method == 'simCLR':
        embeddings = simCLR_embedding(all_train_dataloader, weight_path=weight_path)
        torch.save(embeddings,
                   f'results/embeddings/embeddings_SEED={SEED}.pth')
    else:
        embeddings = torch.load(
            f'results/embeddings/embeddings_SEED={SEED}.pth')

    kmeans = KMeans(n_clusters=len(train_dataset) + num_query, random_state=0)
    kmeans.fit(np.stack(list(embeddings.values())))
    centers = kmeans.cluster_centers_

    labeled_frames = [
        path[len(IMG_PATH):-4] for path, label_path in train_dataset.data_pool
    ]
    labeled_embedding = {}
    for k in labeled_frames:
        labeled_embedding[k] = embeddings[k]
    labeled_embedding_values = []
    labeled_embedding_keys = []
    for k, v in labeled_embedding.items():
        labeled_embedding_keys.append(k)
        labeled_embedding_values.append(v)
    # list of list
    # one list per image: [dist to cluster x, ...]
    center_labeled_dist = torch.cdist(torch.stack(labeled_embedding_values),
                                      torch.tensor(centers))

    L1 = []
    for labeled_frame in center_labeled_dist:
        L2 = []
        for center_n, dist_to_center_n in enumerate(labeled_frame):
            L2.append((center_n, dist_to_center_n.item()))
        L1.append(L2)
    for i, L2 in enumerate(L1):
        L2 = sorted(L2, key=lambda x: x[1])
        L1[i] = (labeled_embedding_keys[i], L2)
    L1 = sorted(L1, key=lambda x: x[1][0][1])
    center_labeled_dist = L1

    fixed_cluster = {}
    for labeled_img, cluster_distances in center_labeled_dist:
        ind = 0
        while True:
            cluster, distance = cluster_distances[ind]
            if cluster not in fixed_cluster:
                fixed_cluster[cluster] = labeled_img
                centers[cluster] = labeled_embedding[labeled_img]
                break
            else:
                ind += 1

    embeddings_keys = []
    embeddings_values = []
    for k, v in embeddings.items():
        embeddings_keys.append(k)
        embeddings_values.append(v)

    nb_iter = 0
    while nb_iter < 300:
        prev_centers = np.array(centers)
        k_means_centers_name = collections.defaultdict(list)
        k_means_centers = collections.defaultdict(list)
        center_fulldataset_dist = torch.cdist(
            torch.tensor(centers),
            torch.stack(embeddings_values))  # (1, number of samples)
        center_fulldataset_dist = torch.argmin(center_fulldataset_dist, axis=0)
        for i, n in enumerate(center_fulldataset_dist):
            k_means_centers[n.item()].append(embeddings_values[i])
            k_means_centers_name[n.item()].append(embeddings_keys[i])

        for k, v in k_means_centers.items():
            if k not in fixed_cluster:
                new_center = average_center(v)
                centers[k] = new_center

        diff = center_diff(centers, prev_centers)
        if diff.item() < 1e-4:
            with open(PRINT_PATH, "a") as f:
                f.write(f"-- kmeans converged in {nb_iter} iter\n")
            break

        nb_iter += 1

    new_labeled = collections.defaultdict(list)
    for i, center in enumerate(centers):
        if i in fixed_cluster:
            continue
        ent = -float('inf')
        name = None
        for k, v in embeddings.items():
            if k not in k_means_centers_name[i]:
                continue
            if ML_entropy[k] > ent:
                ent = ML_entropy[k]
                name = k
        class_name = name[:len(CLASS_ID_TYPE) - 1]
        number = name[len(CLASS_ID_TYPE) + len(FRAME_KEYWORD):]
        new_labeled[class_name].append(number)

    k_means_name_centers = {}
    for k, v in k_means_centers_name.items():
        for name in v:
            k_means_name_centers[name] = k
    with open(f'results/k_means_name_centers_SEED={SEED}_round={n_round}.json', 'w') as f:
        json.dump(k_means_name_centers, f)
    with open(f'results/fixed_cluster_SEED={SEED}_round={n_round}.json', 'w') as f:
        json.dump(fixed_cluster, f)

    return new_labeled