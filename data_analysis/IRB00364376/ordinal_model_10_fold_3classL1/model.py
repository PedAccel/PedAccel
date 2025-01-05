# %%
from scipy.io import loadmat, savemat
import torch
import torch.nn as nn
import torch.nn.functional as F
import seaborn as sns
import numpy as np
import torch.optim as optim
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import metrics
from torch.utils.data.sampler import WeightedRandomSampler

#%%
class ordinal_regression(nn.Module):
    def __init__(self, ndim, ydim):
        super(ordinal_regression, self).__init__()
        self.w = nn.Linear(ndim, 1, bias=False)
        self.bias = nn.Parameter(torch.zeros((1, ydim)))
        self.ydim = ydim
   
    def forward(self, X):
        X = torch.tensor(X, dtype=torch.float32)
        y_hat = self.w(X)
        y_hat = F.sigmoid(y_hat.repeat(1,self.ydim) + self.bias)
        return y_hat
   
    def predict(self, X):
        X = torch.tensor(X, dtype=torch.float32)
        with torch.no_grad():
            y_hat = self.w(X)
            y_hat = F.sigmoid(y_hat.repeat(1,self.ydim) + self.bias)
            return y_hat.numpy()      
#%%
def training(model, X, y, samples_weight, max_iter):
    learning_rate = 0.0001
    epsilon = 1e-7
    criterion = torch.nn.BCELoss()
    l1_mag = 0.00
    opt_j     = optim.Adam(model.parameters(), lr=learning_rate)
    y_train = torch.Tensor(y)
    X_train = X
    loss_criterion = []
    loss_sparsity = []
    old_loss = 0
    for i in range(max_iter):
        opt_j.zero_grad() # Setting our stored gradients equal to zero
        sampler = list(WeightedRandomSampler(samples_weight, len(samples_weight), replacement = True))
        X_data = X_train[sampler, :]
        y_data = y_train[sampler]
        # X_data = X_train
        # y_data = y_train
        outputs = model(X_data)
        loss_c = criterion(outputs, y_data)# + l1_mag*torch.abs(model.w.weight).sum()
        loss_criterion.append(loss_c.detach().item())
        loss = loss_c
        loss.backward() # Computes the gradient of the given tensor w.r.t. the weights/bias
        opt_j.step() # Updates weights and biases with the optimizer (SGD)
        if abs(old_loss - loss) < epsilon:
            break
        old_loss = loss.detach().item()
        
    return loss_criterion, loss_sparsity, model.w.weight.clone().detach().numpy()

# %%
for repeat in range(10):
    Acc = []
    F1 = []
    Sens = []
    Spec = []
    Kappa = []
    y_true = []
    y_pred = []
    X_wave = []
    test_ids = []
    importances = []

    for fold in range(10):
        data_folder = f'cv{repeat}/fold{fold}.mat'
        X_train = (loadmat(data_folder)['X_train'])
        y_train = (loadmat(data_folder)['y_train']) 
        X_test = (loadmat(data_folder)['X_test'])  
        x_wave = (loadmat(data_folder)['X_test_wave']) 
        y_test = (loadmat(data_folder)['y_test'])
        test_id = (loadmat(data_folder)['test_ids'])
        y_test = y_test.sum(axis = 1)

        y_train_label = y_train.sum(axis = 1)
        class_sample_count = np.array([len(np.where(y_train_label==t)[0]) for t in np.unique(y_train_label)])
        weight = 1. / class_sample_count
        samples_weight = np.array([weight[t] for t in y_train_label])
        samples_weight = torch.from_numpy(samples_weight)

        # training
        model = ordinal_regression(X_train.shape[1], y_train.shape[1])
        loss_c, loss_s,weights = training(model, X_train.astype('double'), y_train.astype('int16'), samples_weight,  10000)
        importances.append(np.squeeze(weights))

        with torch.no_grad():
            y_prob_fixed = model(torch.Tensor(X_test))
            #Y_hat = (cpu(y_j_hat).data.numpy()> ordinal_thres).cumprod(axis = 1).sum(axis = 1)
            y_te = (y_prob_fixed.detach().numpy() > 0.5).cumprod(axis = 1).sum(axis = 1)
            y_te[y_te < 0] = 0

            # class_sample_count = np.array([len(np.where(y_test==t)[0]) for t in np.unique(y_test)])
            # weight = 1. / class_sample_count
            # samples_weight = np.array([weight[t] for t in y_test])

            f1 = metrics.f1_score(y_test, y_te, average='weighted')
            accuracy = metrics.balanced_accuracy_score(y_test, y_te)
            kappa = metrics.cohen_kappa_score(y_test, y_te)
            sensitivity = metrics.recall_score(y_test, y_te, average='macro')
            specs = []
            for i in range(1, 3):
                prec, recall,_, _ = metrics.precision_recall_fscore_support(y_test==i, y_te==i, pos_label=True, average=None)
                specs.append(recall[0])
            specificity = sum(specs)/len(specs)
            Y = y_test
            Y_PRED = y_te
            y_true += np.squeeze(Y).tolist()
            y_pred += np.squeeze(Y_PRED).tolist()
            test_ids += np.squeeze(test_id).tolist()
        
        X_wave.append(x_wave)
        Acc.append(accuracy)
        Kappa.append(kappa)
        Sens.append(sensitivity)
        Spec.append(specificity)
        F1.append(f1)

    X_wave = np.concatenate(X_wave, axis = 0)
    folder = "results/"
    filename = folder + (f"cv{repeat}.mat")
    savemat(filename,  {'Acc':Acc, 'Spec':Spec, 'Sens':Sens, 'Kappa':Kappa, 'F1':F1, 'y_true':y_true, 'y_pred':y_pred, 'importance':np.vstack(importances).mean(axis = 0), 'X_wave':X_wave, 'test_ids':test_ids})
    # %%
