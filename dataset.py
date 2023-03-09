from routes import CLASS_ID_CUT, CLASS_ID_TYPE, FILE_TYPE, FRAME_KEYWORD, IMG_PATH, LAB_PATH
from torch.utils.data import Dataset
import os
import numpy as np
import torch
from PIL import Image
import collections
from torchvision import transforms as T
from torchvision.transforms import InterpolationMode as Inter

LABEL_TO_COLORS = {'background': [0, 0, 0],
                    'REBUS Sheath': [0, 255, 0],
                    'needle clear sheath': [255, 0, 0],
                    'forceps sheath': [125, 0, 255],
                    'needle tip': [255, 125, 0],
                    'needle blue sheath': [255, 0, 125],
                    'needle brush': [255, 125, 125],
                    'forceps clamp': [0, 0, 255],
                    'REBUS Probe': [255, 255, 0],
                    'forceps blue': [0, 125, 255]}

class DataHandler(Dataset):
    def __init__(self, data_path, img_trans=None, label_trans=None):
        self.data_path = data_path
        self.data_pool = np.concatenate(list(data_path.values()), axis=0)
        self.img_trans = img_trans
        self.label_trans = label_trans

    def get_mask(self, class_ind, i):
        assert i < len(self.data_path[class_ind]) - 1
        left_mask = self.data_path[class_ind][i][1]
        right_mask = self.data_path[class_ind][i + 1][1]

        return left_mask, right_mask

    def open_path(self, img_path, mask_path, toTensor=True):
        x = Image.open(img_path)
        y = Image.open(mask_path).convert('L')

        rnd_state = torch.random.get_rng_state()
        if self.img_trans is not None:
            torch.random.set_rng_state(rnd_state)
            tensor_trans = T.ToTensor()
            x = self.img_trans(x)
            if toTensor:
                x = tensor_trans(x)
            else:
                x = np.array(x)

        if self.label_trans is not None:
            torch.random.set_rng_state(rnd_state)
            y = self.label_trans(y)
            y = np.array(y)
            # y = y != 0
            if toTensor:
                y = torch.Tensor(y)
                y = y.float()
            else:
                y = y.astype('float')

        return x, y

    def __getitem__(self, index):
        x, y = self.open_path(self.data_pool[index][0],
                              self.data_pool[index][1])
        name = self.data_pool[index][1]
        name = name[len(LAB_PATH):-len(FILE_TYPE)]

        return x, y, name

    def __len__(self):
        return len(self.data_pool)


class VAE_DataHandler(Dataset):
    def __init__(self, data_path):
        self.data_path = data_path
        img_trans = T.Compose(
            [T.Resize((192, 192), interpolation=Inter.BILINEAR),
             T.ToTensor()])
        self.img_trans = img_trans

    def __getitem__(self, index):
        x = Image.open(self.data_path[index])

        if self.img_trans is not None:
            x = self.img_trans(x)

        name = self.data_path[index]
        name = name[len(IMG_PATH):-len(FILE_TYPE)]
        return x, name

    def __len__(self):
        return len(self.data_path)


class CustomDataLoader():
    def __init__(self, dataset, batch_size, shuffle=False, drop_last=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.loader_inds = np.arange(len(dataset))

    def __iter__(self):
        batch_x = []
        batch_y = []
        batch_names = []

        if self.shuffle:
            np.random.shuffle(self.loader_inds)

        for i in self.loader_inds:
            x, y, name = self.dataset[i]
            batch_x.append(x)
            batch_y.append(y)
            batch_names.append(name)

            if len(batch_x) == self.batch_size:
                yield torch.stack(batch_x), torch.stack(batch_y), batch_names
                batch_x = []
                batch_y = []
                batch_names = []

        if not self.drop_last and len(batch_x) > 0:
            yield torch.stack(batch_x), torch.stack(batch_y), batch_names

    def __len__(self):
        return len(self.dataset)


# def divide_data_split_auris(img_path, lab_path, num_val=20):
#     data_split = {
#         '69-00': '',
#         '73-00': '',
#         '73-01': '',
#         '43-00': '',
#         '73-02': '',
#         '75-01': '',
#         '42-00': '',
#         '42-01': '',
#         '77-02': '',
#         '75-00': '',
#         '48-01': '',
#         '47-00': '',
#         '45-00': '',
#         '77-00': '',
#         '48-00': '',
#         '15-02': 'train',
#         '12-00': '',
#         '15-00': 'train',
#         '04-03': 'train',
#         '12-01': 'train',
#         '04-07': 'train',
#         '13-01': 'train',
#         '03-06': 'train',
#         '15-03': 'train',
#         '15-01': 'train',
#         '04-05': 'train',
#         '04-02': 'train',
#         '01-03': 'train',
#         '04-04': 'train',
#         '01-07': 'train',
#         '12-03': 'train',
#         '04-00': 'train',
#         '13-00': 'train',
#         '02-00': 'train',
#         '01-05': 'train',
#         '03-04': 'train',
#         '03-05': 'train',
#         '01-04': 'train',
#         '03-02': 'train',
#         '03-00': 'train',
#         '01-00': 'train',
#         '03-07': 'train',
#         '15-04': 'train',
#         '04-01': 'train',
#         '04-06': 'train',
#         '01-06': 'train',
#         '02-01': 'train',
#         '03-01': 'train',
#         '12-02': 'train',
#         '12-04': 'train',
#         '13-02': 'train',
#         '13-04': 'train',
#         '03-03': 'train',
#         '13-03': 'train',
#         '01-01': 'train',
#         '15-05': 'train',
#         '01-02': 'train',
#         '09-00': 'test',
#         '09-02': 'test',
#         '09-01': 'test',
#         '08-03': 'test',
#         '10-02': 'test',
#         '11-04': 'test',
#         '08-01': 'test',
#         '08-06': 'test',
#         '11-03': 'test',
#         '10-03': 'test',
#         '10-00': 'test',
#         '11-00': 'test',
#         '08-04': 'test',
#         '08-05': 'test',
#         '10-06': 'test',
#         '10-01': 'test',
#         '08-02': 'test',
#         '10-05': 'test',
#         '11-01': 'test',
#         '08-00': 'test',
#         '10-04': 'test',
#         '08-08': 'test',
#         '10-07': 'test',
#         '11-02': 'test',
#         '08-07': 'test',
#         '08-09': 'test'
#     }

#     imgs_path = sorted(os.listdir(img_path))
#     labels_path = sorted(os.listdir(lab_path))

#     all_imgs_train = {}
#     all_imgs_test = []
#     for name in imgs_path:
#         if name[0] == '.':
#             continue

#         if name in data_split and data_split[name] != '':
#             img_files = []
#             files = [f for f in os.listdir(img_path + name) if f[0] != '.']
#             files = sorted(files, key=lambda x: int(x[5:-4]))
#             for file in files:
#                 if file[0] == '.':
#                     continue
#                 img_files.append(img_path + name + '/' + file)

#             if data_split[name] == 'train':
#                 all_imgs_train[name] = img_files
#             elif data_split[name] == 'test':
#                 all_imgs_test.append(img_files)

#     all_labels_train = {}
#     all_labels_test = []
#     for name in labels_path:
#         if name[0] == '.':
#             continue
#         if name in data_split and data_split[name] != '':
#             label_files = []
#             files = [f for f in os.listdir(lab_path + name) if f[0] != '.']
#             files = sorted(files, key=lambda x: int(x[5:-4]))
#             for file in files:
#                 if file[0] == '.':
#                     continue
#                 label_files.append(lab_path + name + '/' + file)

#             if data_split[name] == 'train':
#                 all_labels_train[name] = label_files
#             elif data_split[name] == 'test':
#                 all_labels_test.append(label_files)

#     assert len(all_labels_train) == len(all_imgs_train)
#     assert len(all_labels_test) == len(all_imgs_test)

#     train_keys = collections.defaultdict(list)
#     for k in list(all_labels_train.keys()):
#         train_keys[k[:2]].append(k)
#     train_keys = [(k, v) for k, v in train_keys.items()]
#     np.random.shuffle(train_keys)

#     train_data = {}
#     val_data = {}
#     val_count = 0
#     for i, (vid_id, seg_ids) in enumerate(train_keys):
#         if val_count < num_val:
#             for seg_id in seg_ids:
#                 L = []
#                 for j, frame in enumerate(all_imgs_train[seg_id]):
#                     L.append((frame, all_labels_train[seg_id][j]))
#                 val_data[seg_id] = np.array(L)
#                 val_count += 1
#         else:
#             for seg_id in seg_ids:
#                 L = []
#                 for j, frame in enumerate(all_imgs_train[seg_id]):
#                     L.append((frame, all_labels_train[seg_id][j]))
#                 train_data[seg_id] = np.array(L)

#     test_data = {}
#     ind = len(lab_path + CLASS_ID_TYPE)
#     for i, frames in enumerate(all_imgs_test):
#         L = []
#         for j, frame in enumerate(frames):
#             L.append((frame, all_labels_test[i][j]))
#         test_data[frame[ind - len(CLASS_ID_TYPE):ind -
#                         len(CLASS_ID_CUT)]] = np.array(L)
#     return train_data, val_data, test_data


def divide_data_split_auris(img_path, lab_path, num_val=15):
    data_split = {
        '69-00': '',
        '73-00': '',
        '73-01': '',
        '43-00': '',
        '73-02': '',
        '75-01': '',
        '42-00': '',
        '42-01': '',
        '77-02': '',
        '75-00': '',
        '48-01': '',
        '47-00': '',
        '45-00': '',
        '77-00': '',
        '48-00': '',

        '01-00': 'train',
        '01-01': 'train',
        '01-02': 'train',
        '01-03': 'train',
        '01-04': 'train',
        '01-05': '',
        '01-06': 'train',
        '01-07': 'train',

        '02-00': 'train',
        '02-01': 'train',

        '03-00': 'train',
        '03-01': 'train',
        '03-02': 'train',
        '03-03': 'train',
        '03-04': 'train',
        '03-05': 'train',
        '03-06': 'train',
        '03-07': 'train',

        '04-00': '',
        '04-01': '',
        '04-02': '',
        '04-03': '',
        '04-04': 'train',
        '04-05': '',
        '04-06': 'train',
        '04-07': 'train',

        '12-00': '',
        '12-01': '',
        '12-02': '',
        '12-03': 'train',
        '12-04': '',

        '13-00': 'train',
        '13-01': '',
        '13-02': 'train',
        '13-03': 'train',
        '13-04': 'train',

        '15-00': 'train',
        '15-01': 'train',
        '15-02': 'train',
        '15-03': '',
        '15-04': 'train',
        '15-05': 'train',

        '09-00': 'test',
        '09-02': 'test',
        '09-01': 'test',
        '08-03': 'test',
        '10-02': 'test',
        '11-04': 'test',
        '08-01': 'test',
        '08-06': 'test',
        '11-03': 'test',
        '10-03': 'test',
        '10-00': 'test',
        '11-00': 'test',
        '08-04': 'test',
        '08-05': 'test',
        '10-06': 'test',
        '10-01': 'test',
        '08-02': 'test',
        '10-05': 'test',
        '11-01': 'test',
        '08-00': 'test',
        '10-04': 'test',
        '08-08': 'test',
        '10-07': 'test',
        '11-02': 'test',
        '08-07': 'test',
        '08-09': 'test'
    }

    imgs_path = sorted(os.listdir(img_path))
    labels_path = sorted(os.listdir(lab_path))

    all_imgs_train = []
    all_imgs_test = []
    for name in imgs_path:
        if name[0] == '.':
            continue

        if name in data_split and data_split[name] != '':
            img_files = []
            files = [f for f in os.listdir(img_path + name) if f[0] != '.']
            files = sorted(files, key=lambda x: int(x[5:-4]))
            for file in files:
                if file[0] == '.':
                    continue
                img_files.append(img_path + name + '/' + file)

            if data_split[name] == 'train':
                all_imgs_train.append(img_files)
            elif data_split[name] == 'test':
                all_imgs_test.append(img_files)

    all_labels_train = []
    all_labels_test = []
    for name in labels_path:
        if name[0] == '.':
            continue
        if name in data_split and data_split[name] != '':
            label_files = []
            files = [f for f in os.listdir(lab_path + name) if f[0] != '.']
            files = sorted(files, key=lambda x: int(x[5:-4]))
            for file in files:
                if file[0] == '.':
                    continue
                label_files.append(lab_path + name + '/' + file)

            if data_split[name] == 'train':
                all_labels_train.append(label_files)
            elif data_split[name] == 'test':
                all_labels_test.append(label_files)

    assert len(all_labels_train) == len(all_imgs_train)
    assert len(all_labels_test) == len(all_imgs_test)

    indexes = np.arange(len(all_labels_train))
    np.random.shuffle(indexes)

    train_data = {}
    val_data = {}
    ind = len(lab_path + CLASS_ID_TYPE)
    for i, n in enumerate(indexes):
        if i < num_val:
            L = []
            for j, frame in enumerate(all_imgs_train[n]):
                L.append((frame, all_labels_train[n][j]))
            val_data[frame[ind - len(CLASS_ID_TYPE):ind -
                        len(CLASS_ID_CUT)]] = np.array(L)
        else:
            L = []
            for j, frame in enumerate(all_imgs_train[n]):
                L.append((frame, all_labels_train[n][j]))
            train_data[frame[ind - len(CLASS_ID_TYPE):ind -
                        len(CLASS_ID_CUT)]] = np.array(L)

    test_data = {}
    for i, frames in enumerate(all_imgs_test):
        L = []
        for j, frame in enumerate(frames):
            L.append((frame, all_labels_test[i][j]))
        test_data[frame[ind - len(CLASS_ID_TYPE):ind -
                        len(CLASS_ID_CUT)]] = np.array(L)
    return train_data, val_data, test_data

def chose_labeled_auris(train_data, labeled):
    labeled_train = collections.defaultdict(list)
    unlabeled_train = collections.defaultdict(list)
    all_train = collections.defaultdict(list)
    ind_keyword = len(LAB_PATH + CLASS_ID_TYPE)
    for k, v in train_data.items():
        for i in range(len(v)):
            number = v[i][1][ind_keyword + len(FRAME_KEYWORD):-len(FILE_TYPE)]
            if k in labeled and number in labeled[k]:
                labeled_train[k].append(v[i])
            else:
                unlabeled_train[k].append(v[i])
            all_train[k].append(v[i])

    return labeled_train, unlabeled_train, all_train