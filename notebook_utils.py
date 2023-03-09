import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.pyplot import figure
import json
from PIL import Image
import torch
import collections
from utils import average_center
from sklearn.preprocessing import normalize
import torch.nn.functional as F

from routes import IMG_PATH, LAB_PATH, FILE_TYPE, FRAME_KEYWORD, CLASS_ID_TYPE

def draw_sampled_frames(curr_labels, new_labels, exp_folder, name_to_nb, name='', img_path=IMG_PATH, color='green', title=None, save_name=None):
    void = " "
    figure(figsize=(15, 8), dpi=80)
    for i, video_id in enumerate(name_to_nb.keys()):
        frames = os.listdir(f'{img_path}{video_id}/')
        frames = [f for f in frames if f != '.DS_Store']
        frames = sorted(frames, key=lambda x: int(x[len(FRAME_KEYWORD):-len(FILE_TYPE)]))
        frames = [f[len(FRAME_KEYWORD):-len(FILE_TYPE)] for f in frames]
        # min_ind = int(frames[0][len(FRAME_KEYWORD):-len(FILE_TYPE)])
        # max_ind = int(frames[-1][len(FRAME_KEYWORD):-len(FILE_TYPE)])
        min_ind = 0
        max_ind = len(frames)
        plt.vlines([void], 0, max_ind - min_ind)
        void += " "
        if video_id in new_labels:
            for frame in new_labels[video_id]:
                ind = frames.index(frame)
                # plt.plot([video_id], [int(frame) - min_ind], marker='o', markersize=6, color=color)
                plt.plot([video_id], [ind], marker='o', markersize=6, color=color)
        if video_id in curr_labels:
            for frame in curr_labels[video_id]:
                ind = frames.index(frame)
                # plt.plot([video_id], [int(frame) - min_ind], marker='o', markersize=6, color="black")
                plt.plot([video_id], [ind], marker='o', markersize=6, color="black")
        elif video_id not in new_labels and video_id not in curr_labels:
            plt.plot([video_id], [0], marker='o', markersize=3, color="white");

    if title is None:
        title = name[len(exp_folder + 'AL_'):]
    # make legend for color
    plt.plot([video_id], [-500], marker='o', markersize=9, color=color, label='newly sampled frames')
    plt.plot([video_id], [-500], marker='o', markersize=9, color="black", label='already sampled frames')
    plt.legend()
    plt.ylim(-10, 170)
    plt.xticks(rotation=70)
    plt.title(title, fontsize=20)
    plt.ylabel('frame index', fontsize=20)

    if save_name is not None:
        plt.savefig(f'../../../MICCAI/{save_name}.pdf', bbox_inches='tight')
    plt.show();

def draw_sampled_frames_compare(name1, name2, round_, SEED, exp_folder, name_to_nb, img_path=IMG_PATH, color1='orange', color2='magenta', label1=None, label2=None, save_name=None, title=None):
    new_labels1 = get_sampled_labels(name1 + '/', path2='new', round_=round_, seed=SEED)
    curr_labels1 = get_sampled_labels(name1 + '/', path2='curr', round_=round_, seed=SEED)
    new_labels2 = get_sampled_labels(name2 + '/', path2='new', round_=round_, seed=SEED)
    curr_labels2 = get_sampled_labels(name2 + '/', path2='curr', round_=round_, seed=SEED)
    
    void = " "
    for k, v in new_labels1.items():
        curr_labels1[k] = curr_labels1.get(k, []) + v
    
    for k, v in new_labels2.items():
        curr_labels2[k] = curr_labels2.get(k, []) + v

    figure(figsize=(15, 8), dpi=80)
    for video_id in name_to_nb.keys():
        frames = os.listdir(f'{img_path}{video_id}/')
        frames = [f for f in frames if f != '.DS_Store']
        frames = sorted(frames, key=lambda x: int(x[len(FRAME_KEYWORD):-len(FILE_TYPE)]))
        frames = [f[len(FRAME_KEYWORD):-len(FILE_TYPE)] for f in frames]
        min_ind = 0
        max_ind = len(frames)
        plt.vlines([void], 0, max_ind - min_ind)
        void += " "
        if video_id in curr_labels1:
            for frame in curr_labels1[video_id]:
                ind = frames.index(frame)
                plt.plot([video_id], [ind], marker='o', markersize=8, color=color1)
        if video_id in curr_labels2:
            for frame in curr_labels2[video_id]:
                ind = frames.index(frame)
                plt.plot([video_id], [ind], marker='X', markersize=6, color=color2)
        elif video_id not in curr_labels1 and video_id not in curr_labels2:
            plt.plot([video_id], [0], marker='o', markersize=3, color="white");
    
    if label1 is None:
        label1 = name1[len(exp_folder + 'AL_'):]
    if label2 is None:
        label2 = name2[len(exp_folder + 'AL_'):]
    if title is None:
        f'Training frames at round {round_ + 1}'

    # make legend for color
    plt.plot([video_id], [-15], marker='o', markersize=9, color=color1, label=label1)
    plt.plot([video_id], [-15], marker='X', markersize=6, color=color2, label=label2)
    plt.legend()

    plt.ylim(-10)
    plt.xticks(rotation=30, fontsize=15)
    plt.yticks(fontsize=15)
    plt.title(title, fontsize=25)
    plt.ylabel('frame index', fontsize=25)
    if save_name is not None:
        plt.savefig(f'../../../MICCAI/{save_name}.pdf', bbox_inches='tight')
    plt.show();
    
def draw_sampled_frames_cluster(all_labels, new_labels, name_to_nb, fixed_cluster, title='', k_means_name_centers=None, cluster=0, img_path=IMG_PATH):        
    void = " "

    if str(cluster) in fixed_cluster:
        title = title + ' which is fixed'
    figure(figsize=(15, 8), dpi=80)
    for video_id in name_to_nb.keys():
        frames = sorted(os.listdir(f'{img_path}{video_id}/'))
        min_ind = int(frames[0][:-4])
        max_ind = int(frames[-1][:-4])
        plt.vlines([void], 0, max_ind - min_ind)
        void += " "
        if video_id in all_labels:
            for frame in all_labels[video_id]:
                frame_id = video_id + '/' + frame
                frame_cluster = k_means_name_centers[frame_id]
                color = 'green'
                if video_id in new_labels and frame in new_labels[video_id]:
                    color = 'red'
                if frame_cluster == cluster:
                    if str(cluster) in fixed_cluster and frame_id  == fixed_cluster[str(cluster)]:
                        color = 'black'
                    plt.plot([video_id], [int(frame) - min_ind], marker='o', markersize=3, color=color)
                else:
                    plt.plot([video_id], [-5], marker='o', markersize=3, color="white")
    plt.xticks(rotation=70)
    plt.title(title, fontsize=20)
    plt.ylabel('frame number', fontsize=20)
    plt.show();

# generate matplotlib marker and color pair
def get_marker_color_pair(color_seed=0):
    markers = ['o', 'v', '^', '<', '>', 's', 'p', '*', 'h', 'H', 'D', 'd', 'P', 'X', '8', '4', '3', '2', '1', 'p', 'h', '+', 'x', 'X', '|', '_']
    colors = ['r', 'b', 'g', 'c', 'm', 'y', 'k']
    marker_color_pair = []
    for marker in markers:
        for color in colors:
            marker_color_pair.append((marker, color))
    
    np.random.seed(color_seed)
    np.random.shuffle(marker_color_pair)
    return marker_color_pair

def draw_all_sampled_frames_cluster(all_labeled, new_labels, fixed_cluster, img_path, k_means_name_centers=None, color_seed=0, title=None, save_name=None, marker_color_pair=None, figsize=(15, 20)):        
    if marker_color_pair is None:
        marker_color_pair = get_marker_color_pair(color_seed=color_seed)
    void = " "
    # figure(figsize=figsize)
    figure(figsize=(15, 8), dpi=80)
    for video_id, frames in all_labeled.items():
        # frames_folder = sorted(os.listdir(f'{img_path}{video_id}/'), key=lambda x: int(x[len(FRAME_KEYWORD):-len(FILE_TYPE)]))
        # min_ind = int(frames[0])
        # max_ind = int(frames[-1])
        # plt.vlines([void], 0, max_ind - min_ind)
        plt.vlines([void], 0, len(frames))
        void += " "
        for i, frame in enumerate(frames):
            frame_id = video_id + f'/{FRAME_KEYWORD}' + frame
            frame_cluster = k_means_name_centers[frame_id]
            marker, color = marker_color_pair[frame_cluster]
            markersize = 6
            if video_id in new_labels and frame in new_labels[video_id]:
                # color = 'red'
                markersize = 12
            elif frame_id in fixed_cluster.values():
                markersize = 3
            # plt.plot([video_id], [int(frame) - min_ind], marker=marker, markersize=markersize, color=color)
            plt.plot([video_id], [i], marker=marker, markersize=markersize, color=color)

    # make legend for color
    # move the legend to the right
    # remove the line in the legend figure
    for cluster in np.unique(list(k_means_name_centers.values())):
        marker, color = marker_color_pair[cluster]
        plt.plot([video_id], [-15], marker=marker, markersize=6, color=color, label=f'cluster {cluster}')
    
    # put legend on the bottom right
    plt.legend(bbox_to_anchor=(1.1, 0), loc='lower right', borderaxespad=0., handletextpad=1.5, handlelength=0)

    if title is None:
        title = 'All clusters'
    plt.xticks(rotation=70)
    plt.title(title, fontsize=20)
    plt.ylabel('frame index', fontsize=20)
    plt.ylim(-1)

    if save_name:
        plt.savefig(f'../../../{save_name}.pdf', bbox_inches='tight')
    plt.show();

    return marker_color_pair[:len(np.unique(list(k_means_name_centers.values())))]

def get_sampled_labels(name, round_=0, seed=0, path2="new"):
    for folder in sorted(os.listdir(name)):
        if folder.isnumeric() and path2 + f'_labeled_SEED={seed}_round={round_}.json' in os.listdir(name + folder):
            with open(name + f'{folder}/{path2}_labeled_SEED={seed}_round={round_}.json', 'r') as f:
                label = json.load(f)
            return label

def get_embeddings(name, seed=0):
    for folder in sorted(os.listdir(name)):
        if folder.isnumeric():
            embedding_dir = 'embeddings'
            if embedding_dir not in os.listdir(name + folder):
                embedding_dir = 'checkpoints'
            if embedding_dir not in os.listdir(name + folder):
                continue
            saved_embeddings = os.listdir(name + folder + '/' + embedding_dir)
            for file in saved_embeddings:
                if f'embeddings_SEED={seed}.pth' == file:
                    embeddings = torch.load(name + f'{folder}/{embedding_dir}/embeddings_SEED={seed}.pth')
                    return embeddings

def plot_curr_training_images(name_to_nb, curr_labels, new_labels, nb_col=6, figsize=(15, 8), img_path=IMG_PATH, color='green'):
    all_frame_paths = []
    for video_id in name_to_nb.keys():
        curr_frames = curr_labels.get(video_id, [])
        new_frames = new_labels.get(video_id, [])
        curr_frames = [(frame, 'black') for frame in curr_frames]
        new_frames = [(frame, color) for frame in new_frames]
        frames = curr_frames + new_frames
        frames = sorted(frames, key=lambda x: int(x[0]))
        frame_paths = [(f'{img_path}{video_id}/{FRAME_KEYWORD}{frame}{FILE_TYPE}', color) for frame, color in frames]
        all_frame_paths += frame_paths
    curr_col = 0
    curr_video = None
    header = ''
    for frame_path, color in all_frame_paths:
        if curr_col == 0:
            fig, axs = plt.subplots(1, nb_col, figsize=figsize)
        img = Image.open(frame_path)
        img = np.array(img)

        frame_id = frame_path[len(img_path): -len(FILE_TYPE)]
        video_id = frame_id.split('/')[0]
        if curr_video != video_id:
            curr_video = video_id
            header = '*** '
        else:
            header = ''
        axs[curr_col].imshow(img)
        # remove white space around image
        axs[curr_col].margins(x=0)
        axs[curr_col].axis('off')
        axs[curr_col].set_title(header + frame_id, fontsize=8, color=color)
        curr_col += 1

        if curr_col == nb_col:
            plt.show()
            curr_col = 0

    if curr_col > 0 and curr_col < nb_col:
        for i in range(curr_col, nb_col):
            axs[i].imshow(np.zeros_like(img))
            axs[i].axis('off')

def plot_curr_training_images_savefig(name_to_nb, curr_labels, new_labels, nb_col=3, figsize=(15, 8), img_path=IMG_PATH, color='green', save_name=None):
    all_frame_paths = []
    for video_id in name_to_nb.keys():
        new_frames = new_labels.get(video_id, [])
        new_frames = curr_labels.get(video_id, []) + new_frames
        frame_paths = [f'{img_path}{video_id}/{FRAME_KEYWORD}{frame}{FILE_TYPE}' for frame in new_frames]
        frame_paths = sorted(frame_paths, key=lambda x: int(x[len(img_path + CLASS_ID_TYPE + FRAME_KEYWORD): -len(FILE_TYPE)]))
        all_frame_paths += frame_paths
    curr_col = 0
    header = ''
    
    frames_id = ['43', '62', '80', '142', '158']
    for frame_path in all_frame_paths:
        if frame_path[len(img_path):len(img_path + CLASS_ID_TYPE) - 1] != '15-00':
            curr_col = 0
            continue
        if curr_col == 0:
            fig, axs = plt.subplots(1, nb_col, figsize=figsize)
            plt.subplots_adjust(wspace=0.0, hspace=0.05)
        img = Image.open(frame_path)
        img = np.array(img)

        frame_id = frame_path[len(img_path): -len(FILE_TYPE)]
        video_id = frame_id.split('/')[0]
        axs[curr_col].imshow(img)
        # remove white space around image
        axs[curr_col].margins(x=0)
        axs[curr_col].axis('off')
        use_color = color
        if curr_col == 2:
            use_color = 'black'
        axs[curr_col].set_title(header + f'frame {frames_id[curr_col]}', fontsize=10, color=use_color)
        curr_col += 1
        if curr_col == nb_col:
            if save_name is not None:
                plt.savefig(f'../../../MICCAI/{save_name}.pdf', bbox_inches='tight')
            plt.show()
            return

def vec_sim(u, v):
    return torch.dot(u, v) / (torch.norm(u) * torch.norm(v))


def plot_most_similar_images(new_labels, curr_labels, embeddings, img_path=IMG_PATH, nb_col=6, nb_of_similar_img=None, figsize=(15, 6), use_dist=True, plot=True):
    avg_score = []
    for new_video_id, new_frames in new_labels.items():
        for new_frame in new_frames:
            new_frame_id = new_video_id + '/' + new_frame
            new_frame_embedding = embeddings[new_frame_id]

            all_sim_scores = []
            all_dist_scores = []
            for curr_video_id, curr_frames in curr_labels.items():
                for curr_frame in curr_frames:
                    curr_frame_id = curr_video_id + '/' + curr_frame
                    curr_frame_embedding = embeddings[curr_frame_id]
                    all_sim_scores.append([curr_frame_id, vec_sim(new_frame_embedding, curr_frame_embedding).item()])
                    all_dist_scores.append([curr_frame_id, torch.dist(new_frame_embedding, curr_frame_embedding).item()])
            all_dist_scores = sorted(all_dist_scores, key=lambda x: x[1])
            all_sim_scores = sorted(all_sim_scores, key=lambda x: x[1], reverse=True)

            if use_dist:
                avg_score.append(all_dist_scores[0][1])
            else:
                avg_score.append(all_sim_scores[0][1])

            if plot:
                running_nb_of_similar_img = nb_of_similar_img
                fig_col = min(nb_col, nb_of_similar_img) if nb_of_similar_img else nb_col    
                fig, axs = plt.subplots(1, fig_col, figsize=figsize)
                new_frame = np.array(Image.open(img_path + new_frame_id + '.jpg'))
                axs[0].imshow(new_frame)
                axs[0].set_title(f'{new_frame_id}', fontsize=8, color='green', weight='bold')
                axs[0].axis('off')
                curr_col = 1

                if use_dist:
                    all_scores = all_dist_scores
                else:
                    all_scores = all_sim_scores
                    
                for curr_frame_id, sim_score in all_scores:
                    if curr_col == 0:
                        fig, axs = plt.subplots(1, fig_col, figsize=figsize)
                    frame = np.array(Image.open(img_path + curr_frame_id + '.jpg'))
                    axs[curr_col].imshow(frame)
                    axs[curr_col].set_title(f'{curr_frame_id} {round(sim_score, 3)}', fontsize=8)
                    axs[curr_col].axis('off')
                    curr_col += 1
                    if nb_of_similar_img is not None and curr_col == running_nb_of_similar_img:
                        break
                    if curr_col == nb_col:
                        plt.show()
                        curr_col = 0
                        running_nb_of_similar_img -= nb_col
                
                if curr_col != 0 and curr_col < nb_col and nb_of_similar_img > nb_col:
                    for i in range(curr_col, nb_col):
                        axs[i].imshow(np.zeros_like(frame))
                        axs[i].axis('off')
                plt.show()
            
    return avg_score


def display_cluster_img(cluster,
                        embeddings,
                        all_labeled,
                        k_means_name_centers,
                        dataset,
                        fixed_cluster,
                        new_labels,
                        nb_col=10,
                        figsize = (15, 8),
                        img_path=IMG_PATH,
                        lab_path=LAB_PATH,
                        KMEDIAN=False,
                        sphere=False):

    k_means_centers_frames = collections.defaultdict(list)
    for k, v in k_means_name_centers.items():
        k_means_centers_frames[v].append(k)

    if KMEDIAN:
        if str(cluster) in fixed_cluster:
            cluster_center = fixed_cluster[str(cluster)]
        elif cluster in fixed_cluster:
            cluster_center = fixed_cluster[cluster]
        else:
            for frame_id in k_means_centers_frames[cluster]:
                video_id, frame_nb = frame_id.split('/')
                if video_id in new_labels and frame_nb in new_labels[video_id]:
                    cluster_center = frame_id
                    break
        
        cluster_center_embeddings = embeddings[cluster_center]

    else:
        if str(cluster) in fixed_cluster:
            cluster_center = fixed_cluster[str(cluster)]
            cluster_center_embeddings = embeddings[cluster_center]

        elif cluster in fixed_cluster:
            cluster_center = fixed_cluster[cluster]
            cluster_center_embeddings = embeddings[cluster_center]

        else:
            clusters_embeddings = []
            for frame_id in k_means_centers_frames[cluster]:
                clusters_embeddings.append(embeddings[frame_id])
            cluster_center_embeddings = torch.tensor(average_center(torch.stack(clusters_embeddings)))
            if sphere:
                cluster_center_embeddings = F.normalize(cluster_center_embeddings, dim=0)
    
    all_images_ids = []

    col = 0
    for k, frames in all_labeled.items():
        for frame in frames:
            frame_id = k + f'/{FRAME_KEYWORD}' + frame
            frame_cluster = k_means_name_centers[frame_id]
            frame_embedding = embeddings[frame_id]
            dist_to_center = torch.dist(frame_embedding, cluster_center_embeddings).item()

            color = 'black'
            if str(cluster) in fixed_cluster and fixed_cluster[str(cluster)] == frame_id:
                color = 'blue'
            if k in new_labels and frame in new_labels[k]:
                color = 'red'
            if frame_cluster == cluster:
                if col == 0:
                    fig, axs = plt.subplots(1, nb_col, figsize=figsize)
                
                all_images_ids.append(frame_id)
                frame_path = img_path + frame_id + '.png'
                mask_path = lab_path + frame_id + '.png'
                img, lab = dataset.open_path(frame_path, mask_path)
                axs[col].imshow(img.permute((1,2,0)))
                axs[col].set_xticks([])
                axs[col].set_yticks([])
                axs[col].set_title(f'{frame_id} - {round(dist_to_center, 4)}', color=color, fontsize=8)
                col += 1
            if col == nb_col:
                plt.show()
                col = 0
    if col < nb_col:
        for i in range(col, nb_col):
            axs[i].imshow(np.zeros_like(lab))
            axs[i].set_xticks([])
            axs[i].set_yticks([])

    return all_images_ids

def display_cluster_img_savefig(cluster,
                        embeddings,
                        all_labeled,
                        k_means_name_centers,
                        dataset,
                        fixed_cluster,
                        new_labels,
                        center_color='red',
                        target_frame=None,
                        frames_to_plot=None,
                        nb_col=10,
                        figsize = (15, 8),
                        img_path=IMG_PATH,
                        lab_path=LAB_PATH,
                        KMEDIAN=False,
                        sphere=False,
                        save_name=None):

    k_means_centers_frames = collections.defaultdict(list)
    for k, v in k_means_name_centers.items():
        k_means_centers_frames[v].append(k)

    if KMEDIAN:
        if str(cluster) in fixed_cluster:
            cluster_center = fixed_cluster[str(cluster)]
        elif cluster in fixed_cluster:
            cluster_center = fixed_cluster[cluster]
        else:
            for frame_id in k_means_centers_frames[cluster]:
                video_id, frame_nb = frame_id.split('/')
                if video_id in new_labels and frame_nb in new_labels[video_id]:
                    cluster_center = frame_id
                    break
        
        cluster_center_embeddings = embeddings[cluster_center]

    else:
        if str(cluster) in fixed_cluster:
            cluster_center = fixed_cluster[str(cluster)]
            cluster_center_embeddings = embeddings[cluster_center]

        elif cluster in fixed_cluster:
            cluster_center = fixed_cluster[cluster]
            cluster_center_embeddings = embeddings[cluster_center]

        else:
            clusters_embeddings = []
            for frame_id in k_means_centers_frames[cluster]:
                clusters_embeddings.append(embeddings[frame_id])
            cluster_center_embeddings = torch.tensor(average_center(torch.stack(clusters_embeddings)))
            if sphere:
                cluster_center_embeddings = F.normalize(cluster_center_embeddings, dim=0)
    
    all_images_ids = []

    col = 0
    for k, frames in all_labeled.items():
        for frame in frames:
            frame_id = k + '/' + frame
            # if int(frame) < 125:
            #     continue
            if frames_to_plot is not None and int(frame) not in frames_to_plot:
                    continue
            frame_cluster = k_means_name_centers[frame_id]
            frame_embedding = embeddings[frame_id]
            dist_to_center = torch.dist(frame_embedding, cluster_center_embeddings).item()

            color = 'black'
            if str(cluster) in fixed_cluster and fixed_cluster[str(cluster)] == frame_id:
                color = 'blue'
            if k in new_labels and frame in new_labels[k]:
                color = center_color
            
            if target_frame is not None and target_frame == int(frame):
                color = 'blue'
            if frame_cluster == cluster:
                if col == 0:
                    fig, axs = plt.subplots(1, nb_col, figsize=figsize)
                    # reduce blank space between subplots
                    fig.subplots_adjust(wspace=0, hspace=0)
                
                all_images_ids.append(frame_id)
                frame_path = img_path + frame_id + '.jpg'
                mask_path = lab_path + frame_id + '.png'
                img, lab = dataset.open_path(frame_path, mask_path)
                axs[col].imshow(img.permute((1,2,0)))
                axs[col].set_xticks([])
                axs[col].set_yticks([])
                axs[col].set_title(f'{frame_id[-5:]} - {round(dist_to_center, 4)}', color=color, fontsize=8)
                col += 1
            if col == nb_col:
                if save_name is not None:
                    plt.savefig(f'../../../{save_name}.pdf', bbox_inches='tight')
                return all_images_ids

def image_to_cluster_distance(image_id, cluster, embeddings, k_means_name_centers, fixed_cluster, new_labels, KMEDIAN=False, sphere=False):
    k_means_centers_frames = collections.defaultdict(list)
    for k, v in k_means_name_centers.items():
        k_means_centers_frames[v].append(k)

    if KMEDIAN:
        if str(cluster) in fixed_cluster:
            cluster_center = fixed_cluster[str(cluster)]
        elif cluster in fixed_cluster:
            cluster_center = fixed_cluster[cluster]
        else:
            for frame_id in k_means_centers_frames[cluster]:
                video_id, frame_nb = frame_id.split('/')
                if video_id in new_labels and frame_nb in new_labels[video_id]:
                    cluster_center = frame_id
                    break

        cluster_center_embeddings = embeddings[cluster_center]
    
    else:
        if str(cluster) in fixed_cluster:
            cluster_center = fixed_cluster[str(cluster)]
            cluster_center_embeddings = embeddings[cluster_center]

        elif cluster in fixed_cluster:
            cluster_center = fixed_cluster[cluster]
            cluster_center_embeddings = embeddings[cluster_center]
        else:
            clusters_embeddings = []
            for frame_id in k_means_centers_frames[cluster]:
                clusters_embeddings.append(embeddings[frame_id])
            cluster_center_embeddings = torch.tensor(average_center(torch.stack(clusters_embeddings)))
            if sphere:
                cluster_center_embeddings = F.normalize(cluster_center_embeddings, dim=0)
    
    frame_embedding = embeddings[image_id]
    dist_to_center = torch.dist(frame_embedding, cluster_center_embeddings).item()
    return dist_to_center

def image_to_image_distance(image_id1, image_id2, embeddings):    
    frame_embedding = embeddings[image_id1]
    frame_embedding2 = embeddings[image_id2]
    dist_to_center = torch.dist(frame_embedding, frame_embedding2).item()
    return dist_to_center

def most_similar_images_metrics(name, total_seed=10, total_round=20, img_path=IMG_PATH, use_dist=False):
    all_seed_avg = []
    for SEED in range(1, total_seed+1):
        all_rounds_avg = []
        for round_ in range(total_round):
            new_labels = get_sampled_labels(name + '/', path2='new', round_=round_, seed=SEED)
            curr_labels = get_sampled_labels(name + '/', path2='curr', round_=round_, seed=SEED)
            embeddings = get_embeddings(name + '/', seed=SEED)
            avg_score = plot_most_similar_images(new_labels, curr_labels, embeddings, img_path=img_path, use_dist=use_dist, plot=False)
            all_rounds_avg.append(np.mean(avg_score))
        all_seed_avg.append(all_rounds_avg)
    all_seed_avg = np.array(all_seed_avg)
    plt.plot(np.mean(all_seed_avg, axis=0))
    plt.title('new samples to current dataset proximity')
    if use_dist:
        plt.ylabel('image embedding space euclidian distance');
    else:
        plt.ylabel('image embedding space cosine similarity');
    plt.xlabel('round');
    