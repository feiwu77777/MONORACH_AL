import collections
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
import torch.optim as optim
import os
from torchvision import transforms as T
from torchvision.transforms import InterpolationMode as Inter
import copy
import json
import shutil
import pickle

from utils import set_random
from training_utils import train_validate_v2, validate
from model_v2 import deeplabv3_resnet50
from dataset import DataHandler, chose_labeled_auris, divide_data_split_auris
from routes import CLASS_ID_TYPE, CONTINUE_PATH, CONTINUE_FOLDER, FILE_TYPE, FRAME_KEYWORD, IMG_PATH, LAB_PATH, PRINT_PATH
from OF_query import OF_query, RAFT_query, RAFTxSim_query, density_OF_query
from other_queries import GT_query, GTxSim_query, density_entropy_query, density_query, entropy_query, random_query, similarity_query, k_means_fulldataset_center_query, k_means_entropy_query, k_means_fulldataset_entropy_query

# config
config = {
    'IMG_SIZE': 220,
    'LEARNING_RATE': 1e-4,
    'MAX_PATIENCE': 20,
    'DEVICE': "cuda" if torch.cuda.is_available() else "cpu",
    'BATCH_SIZE': 4,
    'EPOCHS': 10000000,
    'MIN_ITERATION': 0,  # 150 epochs * (100) / 4
    'PATIENCE_ITER': 0,  # = max training samples / batch size
    'N_LABEL': 1,
    'RUNNING_COEF': 0.99,
    'METRIC': 'AUROC',
    'CONTINUE_TRAIN': False,
    'INIT_FRAME_PER_VIDEO': 1,
    'INIT_NUM_VIDEO': 10,
    'NUM_QUERY': 10,
    'NUM_ROUND': 5,
    'SAMPLING': 'k_means_entropy',  # "density", "entropy", "random", "OF"
    'STEP_SIZE': 1,
    'START_SEED': 0,
    'TOTAL_SEEDS': 10,
    'NB_SAVED_BEST_VAL': 2,
    'EMBEDDING_METHOD': 'simCLR', #'simCLR',
    'USE_SPHERE': False,
    'USE_KMEDIAN': False,
    'PRETRAINED_WEIGHT_PATH': '../pretrained_models/auris_seg_simCLRv4/best_train_acc.pth.tar',
}


def query_new(copy_model,
              curr_labeled,
              train_dataset,
              unlabeled_dataset,
              all_train_dataset,
              ML_preds,
              SAMPLING,
              NUM_QUERY,
              n_round,
              SEED,
              step=1,
              metric='DICE'):
    if SAMPLING == 'random':
        new_labeled = random_query(unlabeled_dataset, num_query=NUM_QUERY)
    elif SAMPLING == 'density':
        new_labeled = density_query(train_dataset,
                                    unlabeled_dataset,
                                    num_query=NUM_QUERY,
                                    n_round=n_round,
                                    SEED=SEED)
    elif SAMPLING == 'entropy':
        new_labeled = entropy_query(copy_model,
                                    unlabeled_dataset,
                                    num_query=NUM_QUERY,
                                    n_round=n_round,
                                    SEED=SEED)
    elif SAMPLING == 'density-entropy':
        new_labeled = density_entropy_query(copy_model,
                                            curr_labeled,
                                            unlabeled_dataset,
                                            num_query=NUM_QUERY,
                                            n_round=n_round,
                                            SEED=SEED)
    elif SAMPLING == 'OF':
        new_labeled = OF_query(curr_labeled,
                               train_dataset,
                               unlabeled_dataset,
                               ML_preds,
                               num_query=NUM_QUERY,
                               n_round=n_round,
                               SEED=SEED,
                               step=step,
                               metric=metric)
    elif SAMPLING == 'similarity':
        new_labeled = similarity_query(train_dataset, unlabeled_dataset,
                                       NUM_QUERY)
    elif SAMPLING == 'density-OF':
        new_labeled = density_OF_query(ML_preds,
                                       curr_labeled,
                                       train_dataset,
                                       unlabeled_dataset,
                                       num_query=NUM_QUERY,
                                       n_round=n_round,
                                       SEED=SEED,
                                       step=step,
                                       metric=metric)
    elif SAMPLING == 'GT_query':
        new_labeled = GT_query(NUM_QUERY, n_round=n_round, SEED=SEED)
    elif SAMPLING == 'GTxSim_query':
        new_labeled = GTxSim_query(train_dataset,
                                   unlabeled_dataset,
                                   NUM_QUERY,
                                   n_round=n_round,
                                   SEED=SEED)
    elif SAMPLING == 'RAFT_query':
        new_labeled = RAFT_query(curr_labeled,
                                 train_dataset,
                                 unlabeled_dataset,
                                 ML_preds,
                                 NUM_QUERY,
                                 n_round=n_round,
                                 SEED=SEED,
                                 step=step,
                                 metric=metric)
    elif SAMPLING == 'RAFTxSim_query':
        new_labeled = RAFTxSim_query(curr_labeled,
                                     train_dataset,
                                     unlabeled_dataset,
                                     ML_preds,
                                     NUM_QUERY,
                                     n_round=n_round,
                                     SEED=SEED,
                                     step=step,
                                     metric=metric)
    elif SAMPLING == 'k_means_entropy':
        new_labeled = k_means_entropy_query(copy_model,
                                            unlabeled_dataset,
                                            NUM_QUERY,
                                            config['NB_K_MEANS_CLUSTER'],
                                            n_round,
                                            SEED,
                                            smooth=1e-7,
                                            embedding_method=config['EMBEDDING_METHOD'],
                                            weight_path=config['PRETRAINED_WEIGHT_PATH'],)
    elif SAMPLING == 'k_means_fulldataset_center':
            new_labeled = k_means_fulldataset_center_query(copy_model,
                                                            train_dataset,
                                                            unlabeled_dataset,
                                                            all_train_dataset,
                                                            NUM_QUERY,
                                                            n_round,
                                                            SEED,
                                                            smooth=1e-7,
                                                            embedding_method=config['EMBEDDING_METHOD'],
                                                            weight_path=config['PRETRAINED_WEIGHT_PATH'],
                                                            sphere=config['USE_SPHERE'],
                                                            use_kmedian=config['USE_KMEDIAN'])
    elif SAMPLING == 'k_means_fulldataset_entropy':
        new_labeled = k_means_fulldataset_entropy_query(copy_model,
                                                        train_dataset,
                                                        unlabeled_dataset,
                                                        all_train_dataset,
                                                        NUM_QUERY,
                                                        n_round,
                                                        SEED,
                                                        smooth=1e-7,
                                                        embedding_method=config['EMBEDDING_METHOD'],
                                                        weight_path=config['PRETRAINED_WEIGHT_PATH'])
    return new_labeled


if __name__ == '__main__':
    if os.path.isfile(PRINT_PATH):
        os.remove(PRINT_PATH)

    if config['CONTINUE_TRAIN']:
        checkpoint = torch.load(CONTINUE_PATH)
        config['START_SEED'] = checkpoint['seed']

    for SEED in range(config['START_SEED'], config['TOTAL_SEEDS']):
        with open(PRINT_PATH, "a") as f:
            f.write(f"========== SEED: {SEED} ============\n")
        set_random(SEED)
        train_data, val_data, test_data = divide_data_split_auris(
            IMG_PATH, LAB_PATH)
        with open(PRINT_PATH, "a") as f:
            f.write(
                f"train video: {sorted(train_data.keys())}, {len(train_data)}\n" + \
                f"val video: {sorted(val_data.keys())}, {len(val_data)}\n" + \
                f"test video: {sorted(test_data.keys())}, {len(test_data)}\n")
        ### DEFINE DATA AUG AND SPLIT TRAIN SET ###
        if True:
            train_imgTrans = T.Compose([
                T.RandomResizedCrop(config['IMG_SIZE'],
                                    scale=(0.85, 1.),
                                    interpolation=Inter.BILINEAR),
                T.RandomHorizontalFlip(p=0.5),
            ])
            train_labelTrans = T.Compose([
                T.RandomResizedCrop(config['IMG_SIZE'],
                                    scale=(0.85, 1.),
                                    interpolation=Inter.NEAREST),
                T.RandomHorizontalFlip(p=0.5),
            ])
            test_imgTrans = T.Compose([
                T.Resize((config['IMG_SIZE'], config['IMG_SIZE']),
                         interpolation=Inter.BILINEAR),
            ])
            test_labelTrans = T.Compose([
                T.Resize((config['IMG_SIZE'], config['IMG_SIZE']),
                         interpolation=Inter.NEAREST)
            ])
            config['PATIENCE_ITER'] = 0
            # chose frames per video to begin training
            # all_frames = []
            # for k, v in train_data.items():
            #     for v2 in v:
            #         all_frames.append((k, v2[1][-9:-4]))
            #     config['PATIENCE_ITER'] += len(v)
            # np.random.shuffle(all_frames)
            # curr_labeled = collections.defaultdict(list)
            # for i, (k, v) in enumerate(all_frames):
            #     curr_labeled[k].append(v)
            #     if i == config['INIT_FRAME_NB'] - 1:
            #         break
            # all_frames = None
            # for k, v in curr_labeled.items():
            #     curr_labeled[k] = sorted(v)
            # with open(PRINT_PATH, "a") as f:
            #     f.write(f"curr labeled: {curr_labeled}\n")

            curr_labeled = collections.defaultdict(list)
            ind_keyword = len(LAB_PATH + CLASS_ID_TYPE)
            for i, (class_ID, data_paths) in enumerate(train_data.items()):
                config['PATIENCE_ITER'] += len(data_paths)
                if config['INIT_FRAME_PER_VIDEO'] == 2:
                    inds = [
                        int(0.25 * len(data_paths)),
                        int(0.75 * len(data_paths))
                    ]
                elif config['INIT_FRAME_PER_VIDEO'] == 1:
                    inds = [int(0.5 * len(data_paths))]
                elif config['INIT_FRAME_PER_VIDEO'] == 5:
                    inds = [
                        0,
                        int(0.25 * len(data_paths)),
                        int(0.50 * len(data_paths)),
                        int(0.75 * len(data_paths)),
                        len(data_paths) - 1
                    ]
                if config["NUM_ROUND"] == 1:
                    inds = np.arange(len(data_paths))
                L = []
                for ind in inds:
                    mask_path = data_paths[ind][1]
                    number = mask_path[ind_keyword +
                                       len(FRAME_KEYWORD):-len(FILE_TYPE)]
                    L.append(number)
                curr_labeled[class_ID] = L
                if i == config['INIT_NUM_VIDEO'] - 1 and config['NUM_ROUND'] != 1:
                    break
        config[
            'PATIENCE_ITER'] = config['PATIENCE_ITER'] // config['BATCH_SIZE']
        start_round = 0
        test_scores = []
        train_scores = []
        if config['CONTINUE_TRAIN']:
            start_round = checkpoint['round']
            curr_labeled = checkpoint['curr_labeled']

        for n_round in range(start_round, config['NUM_ROUND']):
            with open(PRINT_PATH, "a") as f:
                f.write(f"======== ROUND: {n_round}, SEED: {SEED} ========\n")
            set_random(SEED)  # to have the same model parameters

            ### SET UP TRAIN DATASET ###
            if True:
                labeled_train, unlabeled_train, all_train = chose_labeled_auris(
                    train_data, labeled=curr_labeled)

                train_dataset = DataHandler(data_path=labeled_train,
                                            img_trans=train_imgTrans,
                                            label_trans=train_labelTrans)
                train_dataloader = DataLoader(
                    train_dataset,
                    batch_size=config['BATCH_SIZE'],
                    shuffle=True,
                    drop_last=True)
                train_dataset_noAug = DataHandler(data_path=labeled_train,
                                                img_trans=test_imgTrans,
                                                label_trans=test_labelTrans)
                all_train_dataset = DataHandler(data_path=all_train,
                                                img_trans=test_imgTrans,
                                                label_trans=test_labelTrans)
                all_train_dataloader = DataLoader(
                    all_train_dataset,
                    batch_size=config['BATCH_SIZE'],
                    shuffle=False)
                val_dataset = DataHandler(data_path=val_data,
                                          img_trans=test_imgTrans,
                                          label_trans=test_labelTrans)
                val_dataloader = DataLoader(val_dataset,
                                            batch_size=config['BATCH_SIZE'],
                                            shuffle=False)
                test_dataset = DataHandler(data_path=test_data,
                                           img_trans=test_imgTrans,
                                           label_trans=test_labelTrans)
                test_dataloader = DataLoader(test_dataset,
                                             batch_size=config['BATCH_SIZE'],
                                             shuffle=False)
                with open(PRINT_PATH, "a") as f:
                    f.write(
                        f"sampling is: {config['SAMPLING']}\n" + \
                        f"metric is: {config['METRIC']}\n" + \
                        f"number of labeled train frames: {len(train_dataset.data_pool)}\n" + \
                        f"number of train frames: {len(all_train_dataset.data_pool)}\n" + \
                        f"number of val frames: {len(val_dataset.data_pool)}\n" + \
                        f"number of test frames: {len(test_dataset.data_pool)}\n" + \
                        f"first dataset sample: {train_dataset.data_pool[0][0]}\n")

            ### DEFINE MODEL ###
            if True:
                model = deeplabv3_resnet50(pretrained=True,
                                           num_classes=1).to(config['DEVICE'])
                with open(PRINT_PATH, "a") as f:
                    f.write(
                        f"model param: {next(model.parameters())[0, 0, 0, 0]}\n"
                    )

                if n_round > 0:
                    if SEED == 0 and n_round <= config['NB_SAVED_BEST_VAL'] + 1:
                        model.load_state_dict(
                            torch.load(
                                CONTINUE_FOLDER +
                                f'best_val_SEED={SEED}_round={n_round - 1}.pth'
                            ))
                        with open(PRINT_PATH, "a") as f:
                            f.write(
                                f"Loaded: best_val_SEED={SEED}_round={n_round - 1}.pth\n"
                            )
                    else:
                        model.load_state_dict(
                            torch.load(CONTINUE_FOLDER + f'best_val.pth'))
                        with open(PRINT_PATH, "a") as f:
                            f.write(f"Loaded: best_val.pth\n")

                copy_model = copy.deepcopy(model).to(config['DEVICE'])
                copy_model.eval()
                optimizer = optim.Adam(model.parameters(),
                                       lr=config['LEARNING_RATE'])
                criterion = nn.BCEWithLogitsLoss()

            start_epoch = 0
            best_val_score = 0
            patience = 0
            curr_iter = 0
            nb_iter = 0

            ### resume training ###
            if config['CONTINUE_TRAIN']:
                best_val_score = checkpoint['best_val_score']
                start_epoch = checkpoint['epoch']
                patience = checkpoint['patience']
                curr_iter = checkpoint['curr_iter']
                nb_iter = checkpoint['nb_iter']
                # resume dataloader
                with open(PRINT_PATH, "a") as f:
                    f.write(f"=== RESTORING DATALOADER\n")
                # resume_dataloader(model, train_dataloader, start_epoch)
                model.load_state_dict(checkpoint['state_dict'])
                copy_model.load_state_dict(checkpoint['copy_state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer'])
                with open(PRINT_PATH, "a") as f:
                    f.write(
                        f"----- model loaded at epoch {start_epoch}-----\n" + \
                        f"----- curr iter is {curr_iter}-----\n"
                        f"----- total iter is {nb_iter}-----\n"
                        f"----- patience is {patience}-----\n"
                        f"----- best val score is {best_val_score}-----\n"
                    )
                config['CONTINUE_TRAIN'] = False
                checkpoint = None

            ### TRAIN THE MODEL ###
            val_scores = []
            shuffle_seed = (11 * SEED +
                            n_round) * config['EPOCHS']
            for epoch in range(start_epoch, config['EPOCHS']):
                model.train()
                break_ = False
                print_first_sample = False
                # a unique seed for every triplet of (SEED, n_round, epoch)
                set_random(shuffle_seed + epoch)
                # num of batches
                for i, (image, mask, names) in enumerate(train_dataloader):

                    if patience == config['MAX_PATIENCE']:
                        with open(PRINT_PATH, "a") as f:
                            f.write(
                                f"=== MAX PATIENCE REACHED, best val score: {best_val_score}\n"
                            )
                        break_ = True
                        break

                    if print_first_sample == True and i == 0:
                        print_first_sample = False
                        with open(PRINT_PATH, "a") as f:
                            f.write(
                                f"=== first sample at epoch {epoch + 1}: {image[0, 0, 120, 120]}\n"
                            )

                    image, mask = image.to(config['DEVICE']), mask.to(
                        config['DEVICE'])
                    optimizer.zero_grad()
                    outputs = model(image)
                    loss = criterion(outputs.squeeze(1), mask)
                    loss.backward()
                    optimizer.step()

                    copy_state_dict = copy_model.state_dict()
                    for k, v in model.state_dict().items():
                        copy_state_dict[k] = config[
                            'RUNNING_COEF'] * copy_state_dict[k] + (
                                1 - config['RUNNING_COEF']) * v
                    copy_model.load_state_dict(copy_state_dict)

                    nb_iter += 1
                    curr_iter += 1

                    if curr_iter >= config['PATIENCE_ITER']:
                        curr_iter = 0
                        _, val_score = validate(copy_model,
                                                val_dataloader,
                                                criterion,
                                                metric='DICE')
                        val_scores.append(val_score)
                        np.save(
                            f'results/val_scores_SEED={SEED}_round={n_round}.npy',
                            val_scores)
                        with open(PRINT_PATH, "a") as f:
                            f.write(
                                f"----- Epoch {epoch + 1}, {nb_iter} iter:  Val score is {val_score}\n"
                            )

                        if val_score > best_val_score:
                            with open(PRINT_PATH, "a") as f:
                                f.write(
                                    f"Best Val changed from {best_val_score} to {val_score}\n"
                                )
                            best_val_score = val_score
                            patience = 0

                            if SEED == 0 and n_round <= config[
                                    'NB_SAVED_BEST_VAL']:
                                if not os.path.exists(CONTINUE_FOLDER):
                                    os.mkdir(CONTINUE_FOLDER)
                                torch.save(
                                    copy_model.state_dict(), CONTINUE_FOLDER +
                                    f'best_val_SEED={SEED}_round={n_round}.pth'
                                )
                            else:
                                if not os.path.exists(CONTINUE_FOLDER):
                                    os.mkdir(CONTINUE_FOLDER)
                                torch.save(copy_model.state_dict(),
                                           CONTINUE_FOLDER + f'best_val.pth')

                        elif val_score <= best_val_score:
                            patience += 1
                            with open(PRINT_PATH, "a") as f:
                                f.write(f"Patience: {patience}\n")

                        ### SAVE MODELS ###
                        if True:
                            state = {
                                'round': n_round,
                                'seed': SEED,
                                'epoch': epoch + 1,
                                'patience': patience,
                                'curr_iter': curr_iter,
                                'nb_iter': nb_iter,
                                'curr_labeled': curr_labeled,
                                'best_val_score': best_val_score,
                                'state_dict': model.state_dict(),
                                'copy_state_dict': copy_model.state_dict(),
                                'optimizer': optimizer.state_dict(),
                            }
                            if not os.path.exists(CONTINUE_FOLDER):
                                os.mkdir(CONTINUE_FOLDER)
                            torch.save(state, CONTINUE_PATH)
                            shutil.copyfile(CONTINUE_PATH, f'./checkpoints/continue_SEED={SEED}.pth')
                if break_:
                    break

            ### test evaluation with the best val model
            if True:
                if SEED == 0 and n_round <= config['NB_SAVED_BEST_VAL']:
                    copy_model.load_state_dict(
                        torch.load(
                            CONTINUE_FOLDER +
                            f'best_val_SEED={SEED}_round={n_round}.pth'))
                else:
                    copy_model.load_state_dict(
                        torch.load(CONTINUE_FOLDER + f'best_val.pth'))

                _, test_score = validate(copy_model,
                                         test_dataloader,
                                         criterion,
                                         metric='DICE')

                train_score, ML_preds = train_validate_v2(
                    copy_model,
                    all_train_dataloader,
                    SEED=SEED,
                    n_round=n_round,
                    metric1='DICE',
                    metric2=config['METRIC'],
                    save="OF" in config['SAMPLING'])

                test_scores.append(test_score)
                train_scores.append(train_score)
                np.save(f'results/test_score_SEED={SEED}.npy', test_scores)
                with open(f'results/train_score_SEED={SEED}.pickle',
                          'wb') as f:
                    pickle.dump(train_scores, f)

                with open(PRINT_PATH, "a") as f:
                    f.write(
                        f"=== END OF ROUND {n_round}, test score: {test_score}, train score: {train_score['all']}\n"
                    )

            ### QUERY NEW LABELS ###
            if config['NUM_ROUND'] > 1:
                unlabeled_dataset = DataHandler(data_path=unlabeled_train,
                                                img_trans=test_imgTrans,
                                                label_trans=test_labelTrans)
                with open(PRINT_PATH, "a") as f:
                    f.write(
                        f"====== unlabeled dataset length: {len(unlabeled_dataset)}\n"
                    )

                new_labeled = query_new(copy_model,
                                        curr_labeled,
                                        train_dataset_noAug,
                                        unlabeled_dataset,
                                        all_train_dataset,
                                        ML_preds,
                                        SAMPLING=config['SAMPLING'],
                                        NUM_QUERY=config['NUM_QUERY'],
                                        n_round=n_round,
                                        SEED=SEED,
                                        step=config['STEP_SIZE'],
                                        metric=config['METRIC'])

                count_dict = {k: len(v) for k, v in new_labeled.items()}
                example_frames = next(iter(new_labeled.values()))
                example_frames = sorted(example_frames, key=lambda x: int(x))
                with open(PRINT_PATH, "a") as f:
                    f.write(
                        f"-- number of video sampled: {len([k for k,v in count_dict.items() if v !=0])}\n" + \
                        f"-- sampled frames: {example_frames}\n"
                        f"-- number of frames per video: {count_dict}\n"
                    )
                with open(
                        f'results/new_labeled_SEED={SEED}_round={n_round}.json',
                        'w') as f:
                    json.dump(new_labeled, f)
                with open(
                        f'results/curr_labeled_SEED={SEED}_round={n_round}.json',
                        'w') as f:
                    json.dump(curr_labeled, f)

                for k, v in new_labeled.items():
                    curr_labeled[k] = curr_labeled[k] + v
