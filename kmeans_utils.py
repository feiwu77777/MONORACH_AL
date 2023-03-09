import numpy as np
import collections
import torch
from utils import average_center, center_diff, median_center
from routes import PRINT_PATH
from sklearn.preprocessing import normalize
from sklearn.cluster import kmeans_plusplus

def get_fixed_clusters(centers, labeled_embedding):
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
    return fixed_cluster, centers


def CustomKMeans_simpler(embeddings, centers=None, fixed_cluster=None,  n_clusters=None, random_state=0, max_iter=300, tolerance=1e-4, notebook=False, sphere=False):
    
    embeddings_keys = []
    embeddings_values = []
    for k, v in embeddings.items():
        embeddings_keys.append(k)
        embeddings_values.append(v)
    
    if centers is None:
        centers, indices = kmeans_plusplus(np.stack(embeddings_values), n_clusters=n_clusters, random_state=random_state)

    nb_iter = 0
    while nb_iter < max_iter:
        prev_centers = np.array(centers)
        k_means_centers_name = collections.defaultdict(list)
        k_means_centers = collections.defaultdict(list)

        ## compute euclidian distance between each sample and each center
        center_fulldataset_dist = torch.cdist(
            torch.tensor(centers),
            torch.stack(embeddings_values))
        center_fulldataset_dist = torch.argmin(center_fulldataset_dist, axis=0)  # (1, number of samples)

        ## assign each sample to the closest center
        for i, n in enumerate(center_fulldataset_dist):
            k_means_centers[n.item()].append(embeddings_values[i])
            k_means_centers_name[n.item()].append(embeddings_keys[i])

        ## compute new centers
        for k, v in k_means_centers.items():
            if fixed_cluster is None:
                new_center = average_center(v)
                centers[k] = new_center
            elif fixed_cluster is not None and k not in fixed_cluster:
                new_center = average_center(v)
                centers[k] = new_center

        if sphere:
            centers = normalize(centers)

        # norm_center = 0
        # for i, center in enumerate(centers):
        #     norm_center += np.linalg.norm(center)
        # print('centers', norm_center / len(centers))

        ## check convergence
        diff = center_diff(centers, prev_centers)
        if diff.item() < tolerance:
            if notebook:
                print(f"-- kmeans converged in {nb_iter} iter {diff.item()}")
            else:
                with open(PRINT_PATH, "a") as f:
                    f.write(f"-- kmeans converged in {nb_iter} iter {diff.item()}\n")
            break

        nb_iter += 1
    
    return centers, k_means_centers_name

def CustomKMedian(embeddings, centers=None, fixed_cluster=None,  n_clusters=None, random_state=0, max_iter=300, tolerance=1e-4, notebook=False):
    
    embeddings_keys = []
    embeddings_values = []
    for k, v in embeddings.items():
        embeddings_keys.append(k)
        embeddings_values.append(v)
    
    if centers is None:
        centers, indices = kmeans_plusplus(np.stack(embeddings_values), n_clusters=n_clusters, random_state=random_state)

    nb_iter = 0
    while nb_iter < max_iter:
        prev_centers = np.array(centers)
        k_means_centers_name = collections.defaultdict(list)
        k_means_centers = collections.defaultdict(list)

        ## compute euclidian distance between each sample and each center
        center_fulldataset_dist = torch.cdist(
            torch.tensor(centers),
            torch.stack(embeddings_values))
        center_fulldataset_dist = torch.argmin(center_fulldataset_dist, axis=0)  # (1, number of samples)

        ## assign each sample to the closest center
        for i, n in enumerate(center_fulldataset_dist):
            k_means_centers[n.item()].append(embeddings_values[i])
            k_means_centers_name[n.item()].append(embeddings_keys[i])

        ## compute new centers
        for k, v in k_means_centers.items():
            if fixed_cluster is None:
                new_center = median_center(v)
                centers[k] = new_center
            elif fixed_cluster is not None and k not in fixed_cluster:
                new_center = median_center(v)
                centers[k] = new_center

        # norm_center = 0
        # for i, center in enumerate(centers):
        #     norm_center += np.linalg.norm(center)
        # print('centers', norm_center / len(centers))

        ## check convergence
        diff = center_diff(centers, prev_centers)
        if diff.item() < tolerance:
            if notebook:
                print(f"-- kmeans converged in {nb_iter} iter {diff.item()}")
            else:
                with open(PRINT_PATH, "a") as f:
                    f.write(f"-- kmeans converged in {nb_iter} iter {diff.item()}\n")
            break

        nb_iter += 1
    
    return centers, k_means_centers_name