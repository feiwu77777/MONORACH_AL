import os
import collections
import torch
from utils import get_score
from routes import CLASS_ID_TYPE, PRINT_PATH, SAVE_MASK_PATH
import numpy as np
import json
from PIL import Image

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def validate(model, dataloader, criterion, metric='DICE'):
    #print("\n--------Validating---------\n")
    model.eval()
    valid_loss = 0.0
    score = 0
    counter = 0
    score_counter = 0

    with torch.no_grad():
        for i, (image, mask, names) in enumerate(dataloader):
            counter += 1
            image, mask = image.to(DEVICE), mask.to(DEVICE)

            outputs = model(image)
            loss = criterion(outputs.squeeze(1), mask)
            pred = torch.sigmoid(outputs).squeeze(1)
            valid_loss += loss.item()

            for j in range(len(mask)):
                score_counter += 1
                score += get_score(mask[j].detach().cpu().numpy(),
                                   pred[j].detach().cpu().numpy(),
                                   metric=metric)
    image = Image.fromarray((image[0].permute((1, 2, 0)).detach().cpu().numpy() * 255).astype(np.uint8))
    label = Image.fromarray((mask[0].detach().cpu().numpy() * 255).astype(np.uint8))
    pred = Image.fromarray((pred[0].detach().cpu().numpy() * 255).astype(np.uint8))
    image.save(f"results/preview_test_image.png")
    label.save(f"results/preview_test_label.png")
    pred.save(f"results/preview_test_pred.png")

    valid_loss = valid_loss / counter
    score = score / score_counter
    return valid_loss, score


def validate_copy(model, copy_model, dataloader, metric='DICE'):
    #print("\n--------Validating---------\n")
    model.eval()
    copy_model.eval()
    score = 0
    copy_score = 0
    score_counter = 0

    with torch.no_grad():
        for i, (image, mask, names) in enumerate(dataloader):
            image, mask = image.to(DEVICE), mask.to(DEVICE)

            outputs = model(image)
            copy_outputs = copy_model(image)
            pred = torch.sigmoid(outputs).squeeze(1)
            copy_pred = torch.sigmoid(copy_outputs).squeeze(1)

            for j in range(len(mask)):
                score_counter += 1
                score += get_score(mask[j].detach().cpu().numpy(),
                                   pred[j].detach().cpu().numpy(),
                                   metric=metric)
                copy_score += get_score(mask[j].detach().cpu().numpy(),
                                        copy_pred[j].detach().cpu().numpy(),
                                        metric=metric)

    score = score / score_counter
    copy_score = copy_score / score_counter
    return score, copy_score


def train_validate(model, dataloader, metric='DICE'):
    #print("\n--------Validating---------\n")
    model.eval()
    score = 0
    counter = 0
    score_counter = 0

    score_per_class = collections.defaultdict(float)
    counter_per_class = collections.defaultdict(int)
    with torch.no_grad():
        for i, (image, mask, names) in enumerate(dataloader):
            counter += 1
            image, mask = image.to(DEVICE), mask.to(DEVICE)

            outputs = model(image)
            pred = torch.sigmoid(outputs).squeeze(1)

            for j in range(len(mask)):
                score_counter += 1
                res = get_score(mask[j].detach().cpu().numpy(),
                                pred[j].detach().cpu().numpy(),
                                metric=metric)
                score += res
                score_per_class[names[j][:-5]] += res
                counter_per_class[names[j][:-5]] += 1

    for k, v in score_per_class.items():
        score_per_class[k] = v / counter_per_class[k]
    score = score / score_counter

    score_per_class['all'] = score
    return score_per_class


def train_validate_v2(model,
                      dataloader,
                      SEED,
                      n_round,
                      metric1='DICE',
                      metric2='AUROC',
                      save=True):
    #print("\n--------Validating---------\n")
    model.eval()
    score = 0
    counter = 0
    score_counter = 0

    score_per_class = collections.defaultdict(float)
    counter_per_class = collections.defaultdict(int)

    ML_preds = {}
    ML_scores = {}

    with torch.no_grad():
        for i, (image, mask, names) in enumerate(dataloader):
            counter += 1
            image, mask = image.to(DEVICE), mask.to(DEVICE)

            outputs = model(image)
            pred = torch.sigmoid(outputs).squeeze(1)

            for j in range(len(mask)):
                score_counter += 1
                res = get_score(mask[j].detach().cpu().numpy(),
                                pred[j].detach().cpu().numpy(),
                                metric=metric1)
                score += res
                class_ID = names[j][:len(CLASS_ID_TYPE)]
                score_per_class[class_ID] += res
                counter_per_class[class_ID] += 1

                if save:
                    save_path = SAVE_MASK_PATH + names[j] + '.npy'
                    if not os.path.exists(SAVE_MASK_PATH):
                        os.mkdir(SAVE_MASK_PATH)
                    if not os.path.exists(SAVE_MASK_PATH + class_ID):
                        os.mkdir(SAVE_MASK_PATH + class_ID)
                    np.save(save_path, pred[j].detach().cpu().numpy())
                    ML_preds[names[j]] = save_path

                    res = get_score(mask[j].detach().cpu().numpy(),
                                    pred[j].detach().cpu().numpy(),
                                    metric=metric2)
                    ML_scores[names[j]] = res
    
    image = Image.fromarray((image[0].permute((1, 2, 0)).detach().cpu().numpy() * 255).astype(np.uint8))
    label = Image.fromarray((mask[0].detach().cpu().numpy() * 255).astype(np.uint8))
    pred = Image.fromarray((pred[0].detach().cpu().numpy() * 255).astype(np.uint8))
    image.save(f"results/preview_train_image.png")
    label.save(f"results/preview_train_label.png")
    pred.save(f"results/preview_train_pred.png")

    if save:
        with open(f'results/MLvsGT_scores_SEED={SEED}_round={n_round}.json',
                  'w') as f:
            json.dump(ML_scores, f)
        with open(f'results/ML_preds_SEED={SEED}_round={n_round}.json',
                  'w') as f:
            json.dump(ML_preds, f)

    for k, v in score_per_class.items():
        score_per_class[k] = v / counter_per_class[k]
    score = score / score_counter

    score_per_class['all'] = score
    return score_per_class, ML_preds


def fit(model,
        copy_model,
        running_coef,
        dataloader,
        optimizer,
        criterion,
        print_first=False,
        metric='DICE'):
    #print('-------------Training---------------')
    model.train()
    train_running_loss = 0.0
    score = 0
    counter = 0
    score_counter = 0

    # num of batches
    for i, (image, mask, names) in enumerate(dataloader):
        if print_first and i == 0:
            with open(PRINT_PATH, "a") as f:
                f.write(f"=== first sample: {image[0, 0, 120, 120]}\n")
        counter += 1
        image, mask = image.to(DEVICE), mask.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(image)
        loss = criterion(outputs.squeeze(1), mask)
        pred = torch.sigmoid(outputs).squeeze(1)
        for j in range(len(pred)):
            score_counter += 1
            score += get_score(mask[j].detach().cpu().numpy(),
                               pred[j].detach().cpu().numpy(),
                               metric=metric)
        train_running_loss += loss.item()
        loss.backward()
        optimizer.step()

        copy_state_dict = copy_model.state_dict()
        for k, v in model.state_dict().items():
            copy_state_dict[k] = running_coef * copy_state_dict[k] + (
                1 - running_coef) * v
        copy_model.load_state_dict(copy_state_dict)
    train_loss = train_running_loss / counter
    score = score / score_counter
    return train_loss, score