from torchmin import minimize
from sklearn.base import BaseEstimator, RegressorMixin
import torch
import numpy as np


class IteratedKulsif(BaseEstimator, RegressorMixin):

    def __init__(self,  kernel, rbf_gam, reg_par=1e-3, niter=1):
        super(IteratedKulsif, self).__init__()
        self.reg_par = reg_par
        self.kernel = kernel
        self.niter = niter
        self.rbf_gam = rbf_gam

        # Dummy initialization required for sklearn score function, is overwritten during optimization
        self.alphas = torch.zeros(size=(1, 2))
        self.gram = torch.zeros(size=(2, 2))
        self.source_X, self.target_X = 0., 0.
        self.nsamples_source, self.nsamples_target = 1, 1
        # estimate dist_2 / dist_1 or target / source

    def fit(self, X, y):
        self.curr_iter = 0
        source_X, target_X = X[X[:, -1] == 1][:, :-1], X[X[:, -1] == 0][:, :-1]
        self.nsamples_source, self.nsamples_target = torch.tensor(source_X.shape[0]).to('cuda'), torch.tensor(target_X.shape[0]).to('cuda')
        # 1D data is already wrapped in 2D array, no need to reshape
        self.gram = torch.from_numpy(1 + self.kernel(X=np.concatenate((source_X, target_X), axis=0), gamma=self.rbf_gam)).to('cuda')
        K11 = self.gram[:self.nsamples_source, :self.nsamples_source]
        K12 = self.gram[:self.nsamples_source, self.nsamples_source:]
        self.source_X, self.target_X = source_X, target_X
        # First iteration
        if self.curr_iter == 0:
            ones_trg = torch.ones(self.nsamples_target).T.to('cuda')
            reg = torch.tensor(self.reg_par).to('cuda')
            self.lhs = K11 / self.nsamples_source + reg * torch.eye(self.nsamples_source).to('cuda')
            self.rhs = - K12.double() @ ones_trg.double() / (
                        self.nsamples_source.double() * self.nsamples_target.double() * reg.double())
            self.lhs_inv = torch.linalg.inv(self.lhs)
            alphas = (self.lhs_inv @ self.rhs).reshape(-1, 1)
            betas = ones_trg.reshape(-1, 1) / (self.nsamples_target * reg)
            self.alphas_betas = torch.cat((alphas, betas), dim=0)
            self.curr_iter += 1
        # Further iterations
        it = torch.tensor(1).to('cuda')
        if self.curr_iter == 1 and self.curr_iter < self.niter:
            it = torch.tensor(self.curr_iter)
            rhs = reg.double() * alphas.reshape(-1).double() - K12.double() @ ones_trg.T.double() * ((it + 1).double() / (
                        self.nsamples_source.double() * self.nsamples_target.double() * reg.double()))
            alphas = (self.lhs_inv @ rhs).reshape(-1, 1)
            betas += ones_trg.reshape(-1, 1) / (self.nsamples_target * reg)
            self.alphas_betas = torch.cat((alphas, betas), dim=0)
        while self.curr_iter > 1 and self.curr_iter < self.niter:
            rhs = (reg * alphas.reshape(-1) - K12 @ ones_trg.T * ((self.curr_iter + 1) /
                   (self.nsamples_source * self.nsamples_target * reg)))
            alphas = (self.lhs_inv @ rhs).reshape(-1, 1)
            betas += ones_trg.reshape(-1, 1) / (self.nsamples_target * reg)
            self.alphas_betas = torch.cat((alphas, betas), dim=0)
            self.curr_iter += 1
            it += 1

        del self.lhs_inv
        del self.lhs
        del self.rhs
        del ones_trg
        if self.curr_iter >= 2:
            del rhs
        del alphas
        del betas
        del K11
        del K12
        torch.cuda.empty_cache()
        return self

    def predict(self, dset):
        if len(dset.shape) == 1:
            dset = dset.reshape(-1, 1)
        else:
            dset = dset[:, :-1]
        gram = torch.from_numpy(1 + self.kernel(X=np.concatenate((self.source_X, self.target_X), axis=0), Y=dset, gamma=self.rbf_gam)).to('cuda')
        preds = (self.alphas_betas * gram).sum(axis=0)

        del gram
        torch.cuda.empty_cache()
        return preds.cpu().numpy()

    def loss_regularized(self, alphas):
        weighted_gram_mat = alphas.reshape(1, -1) * self.gram
        inner_sums = weighted_gram_mat.sum(axis=1)
        loss_P = 1 / self.nsamples_target.double() * (-inner_sums[self.nsamples_source:]).sum()
        loss_Q = 1 / self.nsamples_source.double() * (1 / 2 * inner_sums[:self.nsamples_source] ** 2).sum()
        loss = loss_P + loss_Q + self.reg_par / 2 * alphas.T @ self.gram @ alphas
        del self.gram
        del weighted_gram_mat
        torch.cuda.empty_cache()
        return loss

    def score(self, X, y, sample_weight=None):
        return -self.loss_regularized(self.alphas_betas).cpu().numpy()


class IterCpeExp(BaseEstimator, RegressorMixin):

    def __init__(self, kernel, rbf_gam, reg_par=1e-3, niter=1):
        super(IterCpeExp, self).__init__()
        self.reg_par = reg_par
        self.curr_iter = 0
        self.kernel = kernel
        self.rbf_gam = rbf_gam
        self.niter = niter

        # Dummy initialization required for sklearn score function, is overwritten during optimization
        self.alphas = torch.zeros(size=(1, 2))
        self.gram = torch.zeros(size=(2, 2))
        self.source_X, self.target_X = 0., 0.
        self.nsamples_source, self.nsamples_target = 1, 1
        # estimate dist_2 / dist_1 or target / source

    def loss_regularized(self, alphas):
        weighted_gram_mat = alphas * self.gram
        inner_sums = weighted_gram_mat.sum(axis=1)
        loss_P = 1 / self.nsamples_target * torch.exp(-inner_sums[:self.nsamples_target]).sum()
        loss_Q = 1 / self.nsamples_source * torch.exp(inner_sums[self.nsamples_target:]).sum()
        loss = (loss_P.float() + loss_Q.float() + torch.tensor(self.reg_par / 2 * (alphas - self.alphas_prev)).float() @ self.gram.float()
                @ (alphas - self.alphas_prev).T.float())
        del weighted_gram_mat
        torch.cuda.empty_cache()
        return loss

    def fit(self, X, y):
        source_X, target_X = X[X[:, -1] == 1][:, :-1], X[X[:, -1] == 0][:, :-1]
        self.nsamples_source, self.nsamples_target = source_X.shape[0], target_X.shape[0]
        self.gram = torch.from_numpy(1 + self.kernel(X=np.concatenate((target_X, source_X), axis=0), gamma=self.rbf_gam)).to('cuda')
        self.source_X, self.target_X = source_X, target_X
        self.alphas = torch.zeros(self.nsamples_source + self.nsamples_target).to('cuda')
        self.initi = torch.zeros(self.nsamples_source + self.nsamples_target).to('cuda')
        for it in range(int(self.niter)):
            self.alphas_prev = self.alphas
            self.alphas = minimize(fun=self.loss_regularized, x0=self.initi, method='cg',
                                            max_iter=1).x.T
            self.curr_iter += 1
        del self.initi
        torch.cuda.empty_cache()
        return self

    def predict(self, dset):
        dset = dset[:, :-1]
        gram = torch.from_numpy(
            1 + self.kernel(X=np.concatenate((self.target_X, self.source_X), axis=0), Y=dset, gamma=self.rbf_gam)).to(
            'cuda')
        preds = torch.exp((self.alphas.reshape(-1, 1) * gram).sum(dim=0) * 2)
        del gram
        torch.cuda.empty_cache()
        return preds.cpu().numpy()

    def score(self, X, y, sample_weight=None):
        loss = -self.loss_regularized(self.alphas).cpu().numpy()
        del self.gram
        del self.alphas_prev
        torch.cuda.empty_cache()
        return loss


class IterLogi(BaseEstimator, RegressorMixin):

    def __init__(self, kernel, rbf_gam, reg_par=1e-3, niter=1):
        super(IterLogi, self).__init__()
        self.reg_par = reg_par
        self.curr_iter = 0
        self.kernel = kernel
        self.rbf_gam = rbf_gam
        self.niter = niter

        # Dummy initialization required for sklearn score function, is overwritten during optimization
        self.alphas = torch.zeros(size=(1, 2))
        self.gram = torch.zeros(size=(2, 2))
        self.source_X, self.target_X = 0., 0.
        self.nsamples_source, self.nsamples_target = 1, 1
        # estimate dist_2 / dist_1 or target / source

    def loss_regularized(self, alphas):
        weighted_gram_mat = alphas * self.gram
        inner_sums = weighted_gram_mat.sum(axis=1)
        loss_P = 1 / self.nsamples_target * torch.log(1. + torch.exp(-inner_sums[:self.nsamples_target])).sum()
        loss_Q = 1 / self.nsamples_source * torch.log(1. + torch.exp(inner_sums[self.nsamples_target:])).sum()
        loss = (loss_P.float() + loss_Q.float() + torch.tensor(self.reg_par / 2 * (alphas - self.alphas_prev)).float() @ self.gram.float()
                @ (alphas - self.alphas_prev).T.float())
        del weighted_gram_mat
        torch.cuda.empty_cache()
        return loss

    def fit(self, X, y):
        source_X, target_X = X[X[:, -1] == 1][:, :-1], X[X[:, -1] == 0][:, :-1]
        self.nsamples_source, self.nsamples_target = source_X.shape[0], target_X.shape[0]
        self.gram = torch.from_numpy(1 + self.kernel(X=np.concatenate((target_X, source_X), axis=0), gamma=self.rbf_gam)).to('cuda')
        self.source_X, self.target_X = source_X, target_X
        self.alphas = torch.zeros(self.nsamples_source + self.nsamples_target).to('cuda')
        self.initi = torch.zeros(self.nsamples_source + self.nsamples_target).to('cuda')
        for it in range(int(self.niter)):
            self.alphas_prev = self.alphas
            self.alphas = minimize(fun=self.loss_regularized, x0=self.initi, method='cg',
                                   max_iter=1).x.T
            self.curr_iter += 1
        del self.initi
        torch.cuda.empty_cache()
        return self

    def predict(self, dset):
        dset = dset[:, :-1]
        gram = torch.from_numpy(
            1 + self.kernel(X=np.concatenate((self.target_X, self.source_X), axis=0), Y=dset, gamma=self.rbf_gam)).to(
            'cuda')
        preds = torch.exp((self.alphas.reshape(-1, 1) * gram).sum(dim=0))
        del gram
        torch.cuda.empty_cache()
        return preds.cpu().numpy()

    def score(self, X, y, sample_weight=None):
        loss = -self.loss_regularized(self.alphas).cpu().numpy()
        del self.gram
        del self.alphas_prev
        torch.cuda.empty_cache()
        return loss


class IterSquare(BaseEstimator, RegressorMixin):

    def __init__(self, kernel, rbf_gam, reg_par=1e-3, niter=1):
        super(IterSquare, self).__init__()
        self.reg_par = reg_par
        self.curr_iter = 0
        self.kernel = kernel
        self.rbf_gam = rbf_gam
        self.niter = niter

        # Dummy initialization required for sklearn score function, is overwritten during optimization
        self.alphas = torch.zeros(size=(1, 2))
        self.gram = torch.zeros(size=(2, 2))
        self.source_X, self.target_X = 0., 0.
        self.nsamples_source, self.nsamples_target = 1, 1
        # estimate dist_2 / dist_1 or target / source

    def loss_regularized(self, alphas):
        weighted_gram_mat = alphas * self.gram
        inner_sums = weighted_gram_mat.sum(axis=1)
        loss_P = 1 / self.nsamples_target * ((1. - inner_sums[:self.nsamples_target]) ** 2).sum()
        loss_Q = 1 / self.nsamples_source * ((1. + inner_sums[self.nsamples_target:]) ** 2).sum()
        loss = (loss_P.float() + loss_Q.float() + torch.tensor(
            self.reg_par / 2 * (alphas - self.alphas_prev)).float() @ self.gram.float()
                @ (alphas - self.alphas_prev).T.float())
        del weighted_gram_mat
        torch.cuda.empty_cache()
        return loss

    def fit(self, X, y):
        source_X, target_X = X[X[:, -1] == 1][:, :-1], X[X[:, -1] == 0][:, :-1]
        self.nsamples_source, self.nsamples_target = source_X.shape[0], target_X.shape[0]
        self.gram = torch.from_numpy(1 + self.kernel(X=np.concatenate((target_X, source_X), axis=0), gamma=self.rbf_gam)).to('cuda')
        self.source_X, self.target_X = source_X, target_X
        self.alphas = torch.zeros(self.nsamples_source + self.nsamples_target).to('cuda')
        self.initi = torch.zeros(self.nsamples_source + self.nsamples_target).to('cuda')
        for it in range(int(self.niter)):
            self.alphas_prev = self.alphas
            self.alphas = minimize(fun=self.loss_regularized, x0=self.initi, method='cg',
                                   max_iter=1).x.T
            self.curr_iter += 1
        del self.initi
        torch.cuda.empty_cache()
        return self

    def predict(self, dset):
        dset = dset[:, :-1]
        gram = torch.from_numpy(
            1 + self.kernel(X=np.concatenate((self.target_X, self.source_X), axis=0), Y=dset, gamma=self.rbf_gam)).to(
            'cuda')
        f = (self.alphas.reshape(-1, 1) * gram).sum(axis=0)
        preds = (2. * f - 1.) / (2. - 2. * f)
        del gram
        torch.cuda.empty_cache()
        return preds.cpu().numpy()

    def score(self, X, y, sample_weight=None):
        loss = -self.loss_regularized(self.alphas).cpu().numpy()
        del self.gram
        del self.alphas_prev
        torch.cuda.empty_cache()
        return loss
