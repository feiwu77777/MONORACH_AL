import torch
from torchvision.models import resnet34
import numpy as np
from sklearn.metrics import roc_curve
from sklearn.metrics import auc as auc_score
from sklearn.utils.class_weight import compute_class_weight
import torch.optim as optim
from torch.utils.data import DataLoader

from dataset import DataHandler


class Strategy:
    def __init__(self, X, Y, X_test, Y_test, idxs_lb, args, SEED=0):
        self.X = X
        self.Y = Y
        self.X_test = X_test
        self.Y_test = Y_test
        self.idxs_lb = idxs_lb
        self.args = args
        self.n_pool = len(Y)
        use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if use_cuda else "cpu")
        self.SEED = SEED

    def query(self, n):
        pass

    def update(self, idxs_lb):
        self.idxs_lb = idxs_lb

    def _train(self, epoch, loader_tr, optimizer):
        self.clf.train()
        P = torch.zeros(len(self.Y), dtype=torch.long)
        L = torch.zeros(len(self.Y), dtype=torch.long)

        for batch_idx, (x, y, idxs) in enumerate(loader_tr):
            x, y = x.to(self.device), y.to(self.device)
            optimizer.zero_grad()
            out = self.clf(x)

            loss = F.binary_cross_entropy_with_logits(
                    out.squeeze(1), y.float())

            loss.backward()
            optimizer.step()

        return P, L

    def train(self, round):
        print('======================')
        print('ROUND: ', round)
        self.test_preds = []
        self.test_label = []

        n_epoch = self.args['n_epoch']
        start_epoch = 1
        self.clf = resnet34(num_classes=1, pretrained=False)
        self.clf = self.clf.to(self.device)
        optimizer = optim.Adam(self.clf.parameters(),
                               **self.args['optimizer_args'])

        idxs_train = np.arange(self.n_pool)[self.idxs_lb]

        dataset = DataHandler(self.X[idxs_train],
                              self.Y[idxs_train],
                              transform=self.args['train_transform'],
                              pos_ind=self.pos_ind)
        loader_tr = DataLoader(dataset,
                               shuffle=True,
                               **self.args['loader_tr_args'])

        print('length of dataset: ', len(dataset))
        print('length of dataloader: ', len(loader_tr), 'batch size: ',
              self.args['loader_tr_args']['batch_size'])

        print('==============')
        for epoch in range(start_epoch, n_epoch + 1):
            if epoch % 10 == 0:
                print('epoch: ', epoch)
            train_pred, train_label = self._train(epoch, loader_tr, optimizer)
            pred, label = self.predict(self.X_test, self.Y_test)
            self.test_preds.append(np.array(pred))
            self.test_label.append(np.array(label))
            np.save(
                f'results/test_preds_round={round}_seed={self.SEED}_startEp={start_epoch}.npy',
                self.test_preds)
            np.save(
                f'results/test_labels_round={round}_seed={self.SEED}_startEp={start_epoch}.npy',
                self.test_label)

            if epoch % 10 == 0:
                fpr, tpr, _ = roc_curve(np.array(label), np.array(pred))
                roc_auc = auc_score(fpr, tpr)
                print('ROC SCORE: ', roc_auc)

            #### save checkpoint #####
            state = {
                'epoch': epoch + 1,
                'state_dict': self.clf.state_dict(),
                'optimizer': optimizer.state_dict(),
                'seed': self.SEED,
                'idx_lb': self.idxs_lb,
                'round': round
            }
            torch.save(state, 'checkpoints/continue.pth')

    def predict(self, X, Y):
        dataset = DataHandler(X, Y, transform=self.args['test_transform'])
        loader_te = DataLoader(dataset,
                               shuffle=False,
                               **self.args['loader_te_args'])

        self.clf.eval()

        if self.binary:
            P = torch.zeros(len(Y))
            L = torch.zeros(len(Y))
        else:
            P = torch.zeros(len(Y), dtype=torch.long)
            L = torch.zeros(len(Y), dtype=torch.long)

        with torch.no_grad():
            for x, y, idxs in loader_te:
                x, y = x.to(self.device), y.to(self.device)
                out = self.clf(x)

                if self.binary:
                    P[idxs] = torch.sigmoid(out).squeeze(1).cpu()
                    #P[idxs] = torch.tanh(out).squeeze(1).cpu()
                    L[idxs] = y.float().detach().cpu()
                else:
                    pred = out.max(1)[1]
                    P[idxs] = pred.cpu()
                    L[idxs] = y.detach().cpu()

        return P, L