# Author Yaoyao Bao
# yaoyaobao@mail.ecust.edu.cn

from src.feat_engineering.feat_engineering_abstract import FeatEngineeringExtract
from sklearn.preprocessing import StandardScaler
import pandas as pd
import torch
from torch.optim.adam import Adam
from torch.optim import lr_scheduler
from torch.utils.data import TensorDataset, DataLoader


MAX_VAL = 1e10


class MaxCorr(FeatEngineeringExtract):
    def __init__(self, n_components):
        """
        mahalanobis metric learning that maximize the correlation coefficient
        between the distances in feat space and label space

        Parameters
        ----------
        :param n_components: the dimension of the map space (usually smaller than original dimension)
        """
        self.n_comp = n_components
        self.scaler = StandardScaler()
        self.trans_mat = None

    def fit(self, feat, label, **fit_params):
        """
        fit the model with feat and label, which means to calculate the trans_matrix
        :param feat: original features, pd.DataFrame or np.array
        :param label: original labels, pd.DataFrame or np.array
        :param fit_params: params for learning process, i.e. {"batch_size": 30,
                                                              "init_lr": 0.001,
                                                              "max_iterations": 50,
                                                              "early_stop_cnt": 5,
                                                              "l1_norm": 0,
                                                              "l2_norm": 0,
                                                              "verbose": None}
        :return:
        """
        batch_size = fit_params.get("batch_size", 30)
        init_lr = fit_params.get("init_lr", 0.001)
        max_iterations = fit_params.get("max_iterations", 50)
        early_stop_cnt = fit_params.get("early_stop", 5)
        l1_norm = fit_params.get("l1_norm", 0)
        l2_norm = fit_params.get("l2_norm", 0)
        verbose = fit_params.get("verbose", None)

        feat = self.scaler.fit_transform(feat)
        if type(label) == pd.DataFrame:
            label = label.values

        feat_tensor = torch.from_numpy(feat).float()
        label_tensor = torch.from_numpy(label).float()

        dataset = TensorDataset(feat_tensor, label_tensor)
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)

        # random initialize the transform matrix
        trans_mat = torch.randn(self.n_comp, feat_tensor.shape[1]).requires_grad_(True)
        optimizer = Adam([{"params": [trans_mat], "initial_lr": init_lr}], lr=init_lr, weight_decay=l2_norm)
        ls = lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=len(data_loader), eta_min=init_lr/100)

        best_loss, early_stop = MAX_VAL, 0
        for epoch in range(max_iterations):
            loss_tmp, cnt = 0, 0
            for bat_idx, (bat_x, bat_y) in enumerate(data_loader):

                dis_mat = self._self_dis(bat_y)
                dis_x_map = self._self_dis(bat_x.mm(trans_mat.T))
                loss = - ((dis_x_map - dis_x_map.mean()) * (
                        dis_mat - dis_mat.mean())).mean() / dis_x_map.std() / dis_mat.std()
                loss += abs(trans_mat).sum() * l1_norm
                loss_tmp += loss
                cnt += 1

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if verbose is not None:
                if epoch % verbose == 0:
                    print("Epoch: {} | Batch: {} | Loss: {}".format(epoch, epoch, float(loss_tmp / cnt)))

            if loss_tmp / cnt < best_loss:
                best_loss = loss_tmp / cnt
                early_stop = 0
            else:
                early_stop += 1
                if early_stop > early_stop_cnt:
                    break
        self.trans_mat = (trans_mat.T / (trans_mat.T ** 2).sum(0) ** 0.5).T.clone().detach()
        return self

    def transform(self, feat):
        """
        linear map the feat into the label correlative space
        :param feat: features to transform, pd.DataFrame or np.array
        :return: transformed features
        """
        feat = self.scaler.transform(feat)
        feat_tensor = torch.from_numpy(feat).float()
        feat_trans = feat_tensor.mm(self.trans_mat.T).data.numpy()
        return feat_trans

    @staticmethod
    def _self_dis(mat: torch.FloatTensor):
        """
        calculate the distances between those samples in mat
        :param mat: two dimensional tensor
        :return: distance matrix
        """
        n_samp, n_dim = mat.shape
        YY = (mat ** 2).sum(1).repeat((n_samp, 1))
        XY = mat.mm(mat.T)
        dis_matrix = YY + YY.T - 2 * XY
        return dis_matrix




