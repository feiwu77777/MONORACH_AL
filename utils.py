import numpy as np
from VAE_model import VAE
from dataset import VAE_DataHandler
from sklearn.metrics import roc_curve
from sklearn.metrics import auc as auc_score
import collections
import torch
import random
import os
from torchvision.models import resnet34
from torch.utils.data import DataLoader

from routes import PRINT_PATH

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def set_random(SEED):
    random.seed(SEED)
    os.environ['PYTHONHASHSEED'] = str(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def resume_dataloader(model, loader, epochs):
    model.train()
    with torch.no_grad():
        for epoch in range(epochs):
            for i, (x, _, _) in enumerate(loader):
                if i == 0:
                    with open(PRINT_PATH, "a") as f:
                        f.write(
                            f"=== epoch {epoch + 1}, first sample: {x[0, 0, 120, 120]}\n"
                        )
                x = x.to(DEVICE)
                outputs = model(x)


def get_score(y_true, y_pred, metric='DICE'):
    if metric == 'DICE':
        score = dice_coef(y_true, y_pred)
    elif metric == 'AUROC':
        y_true = y_true > 0.5
        if np.sum(y_true) == 0:
            return 1
        y_true = y_true.flatten()
        y_pred = y_pred.flatten()
        fpr, tpr, _ = roc_curve(y_true, y_pred)
        score = auc_score(fpr, tpr)
    elif metric == 'BCE':
        smooth = 1e-7
        y_true = y_true.flatten()
        y_pred = y_pred.flatten()
        score = -np.sum(y_true * np.log(y_pred + smooth) +
                        (1 - y_true) * np.log(1 - y_pred + smooth))
    return score


def dice_coef(y_true, y_pred, smooth=1):
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()

    intersection = np.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (np.sum(y_true_f) +
                                           np.sum(y_pred_f) + smooth)


def pixelChange(x1, x2):
    x1 = x1[:, :, 0] / np.max(x1[:, :, 0])
    x2 = x2[:, :, 0] / np.max(x2[:, :, 0])

    res = np.sum(np.around(x1, 1) != np.around(x2, 1)) / (x1.shape[0] *
                                                          x1.shape[1])
    return res


def mutual_information_2d(x, y, sigma=1, normalized=False):
    bins = (256, 256)
    EPS = np.finfo(float).eps
    jh = np.histogram2d(x, y, bins=bins)[0]

    # smooth the jh with a gaussian filter of given sigma
    ndimage.gaussian_filter(jh, sigma=sigma, mode='constant', output=jh)

    # compute marginal histograms
    jh = jh + EPS
    sh = np.sum(jh)
    jh = jh / sh
    s1 = np.sum(jh, axis=0).reshape((-1, jh.shape[0]))
    s2 = np.sum(jh, axis=1).reshape((jh.shape[1], -1))

    # Normalised Mutual Information of:
    # Studholme,  jhill & jhawkes (1998).
    # "A normalized entropy measure of 3-D medical image alignment".
    # in Proc. Medical Imaging 1998, vol. 3338, San Diego, CA, pp. 132-143.
    if normalized:
        mi = ((np.sum(s1 * np.log(s1)) + np.sum(s2 * np.log(s2))) /
              np.sum(jh * np.log(jh))) - 1
    else:
        mi = (np.sum(jh * np.log(jh)) - np.sum(s1 * np.log(s1)) -
              np.sum(s2 * np.log(s2)))

    return mi


def addSample(new_labeled, curr_labeled, unlabeled_dataset, new_path, class_ID,
              IMG_PATH, thresh):
    x_new, _ = unlabeled_dataset.open_path(new_path, new_path, toTensor=False)
    if class_ID not in new_labeled and class_ID not in curr_labeled:
        return True
    v1 = new_labeled[class_ID]
    v2 = curr_labeled[class_ID]
    for nb in v1:
        img_path = IMG_PATH + class_ID + nb + ".jpg"
        x, _ = unlabeled_dataset.open_path(img_path, img_path, toTensor=False)

        diff = mutual_information_2d(x.ravel(), x_new.ravel())
        if diff > thresh:
            return False
    for nb in v2:
        img_path = IMG_PATH + class_ID + nb + ".jpg"
        x, _ = unlabeled_dataset.open_path(img_path, img_path, toTensor=False)

        diff = mutual_information_2d(x.ravel(), x_new.ravel())
        if diff > thresh:
            return False
    return True


def pixelChangeV2(x1, x2):
    assert (np.min(x1) >= 0)
    assert (np.max(x1) <= 1)
    assert (np.min(x2) >= 0)
    assert (np.max(x2) <= 1)

    res = np.sum(np.around(x1, 1) != np.around(x2, 1)) / (x1.shape[0] *
                                                          x1.shape[1])
    return res


def get_similarity_score(new_labeled, curr_labeled, ML_preds, all_frames):

    all_sims = collections.defaultdict(int)
    for k, v in all_frames.items():
        labeled_frames = new_labeled[k] + curr_labeled[k]
        unlabeled_frames = [nb for nb in v if nb not in labeled_frames]

        for nb in unlabeled_frames:
            mask1 = np.load(ML_preds[k + nb])
            for nb2 in labeled_frames:
                mask2 = np.load(ML_preds[k + nb2])
                sim = 1 - pixelChangeV2(mask1, mask2)
                if sim > all_sims[k + nb]:
                    all_sims[k + nb] = sim

    return all_sims


def resnet_embedding(dataloader, reduce_FM=True):
    activation = {}

    def get_activation(name):
        def hook(model, input, output):
            activation[name] = output.detach()

        return hook

    model = resnet34(pretrained=True)
    model.eval()
    # chose the layer for the embedding
    model.layer2[3].register_forward_hook(get_activation('layer2.3'))
    embedding_resnets = {}
    with torch.no_grad():
        for images, y, names in dataloader:
            output = model(images)
            for i, n in enumerate(names):
                if reduce_FM:
                    embedding_resnets[n] = torch.mean(
                        activation['layer2.3'][i], axis=(1, 2))
                else:
                    embedding_resnets[n] = activation['layer2.3'][i]

    return embedding_resnets


def AE_embeddings(data_path, vae_net):

    dataset = VAE_DataHandler(data_path)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=False)
    embeddings = []
    names = []
    with torch.no_grad():
        for i, (image, name) in enumerate(dataloader):
            recon_data, embedding, log_var = vae_net(image.to(DEVICE))
            # embedding = torch.zeros((8, 512, 3, 3))
            embedding = torch.mean(embedding, axis=(2, 3))
            embeddings.append(embedding)
            names.extend(name)
    embeddings = torch.cat(embeddings, axis=0)

    return embeddings, names


def embedding_similarity(labeled_frames,
                         unlabeled_frames,
                         weight_path='../pretrained_models/vae/all_video.pt'):

    vae_net = VAE(channel_in=3, ch=64).to(DEVICE)
    save_file = torch.load(weight_path)
    vae_net.load_state_dict(save_file['model_state_dict'])
    vae_net.eval()

    labeled_embeddings, _ = AE_embeddings(labeled_frames, vae_net)
    unlabeled_embeddings, names = AE_embeddings(unlabeled_frames, vae_net)

    distances = torch.cdist(unlabeled_embeddings, labeled_embeddings)
    distances = torch.mean(distances, axis=1)

    return distances, names


def simCLR_embedding(dataloader, 
                    arg=None, 
                    weight_path='../pretrained_models/skateboard_simCLR/pretrained_epoch=10.pth.tar', 
                    cpu=False):
    activation = {}

    def get_activation(name):
        def hook(model, input, output):
            activation[name] = output.detach()

        return hook
    
    model = resnet34(pretrained=False)
    model.avgpool.register_forward_hook(get_activation('avgpool'))

    if cpu:
        weight = torch.load(weight_path, map_location='cpu')
    else:
        weight = torch.load(weight_path)

    epoch = weight['epoch']
    with open(PRINT_PATH, "a") as f:
        f.write(f"-- loaded embedding model weight from epoch {epoch}\n")

    state_dict = weight['state_dict']
    for k in list(state_dict.keys()):
        if k.startswith('backbone.'):
            if k.startswith('backbone') and not k.startswith('backbone.fc'):
                # remove prefix
                state_dict[k[len("backbone."):]] = state_dict[k]
        del state_dict[k]
    log = model.load_state_dict(state_dict, strict=False)
    assert log.missing_keys == ['fc.weight', 'fc.bias']

    model = model.to(DEVICE)
    model.eval()

    embeddings = {}
    with torch.no_grad():
        for images, y, names in dataloader:
            images = images.to(DEVICE)
            
            output = model(images)

            for i, name in enumerate(names):
                embeddings[name] = activation['avgpool'][i].squeeze().detach().cpu()

    return embeddings

def euc_distance(x1, x2):
    return torch.sqrt(torch.sum((x1 - x2)**2))


def average_center(embeddings):
    embeddings = np.stack(embeddings)
    return np.mean(embeddings, axis=0)

def median_center(embeddings):
    embeddings = np.stack(embeddings)
    return np.median(embeddings, axis=0)

def center_diff(centers, prev_centers):
    diff = 0
    for i, center in enumerate(centers):
        diff_ = euc_distance(torch.tensor(center),
                             torch.tensor(prev_centers[i]))
        # print(f'cluster {i}', diff_)
        diff += diff_
    return diff