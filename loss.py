from typing import Any, Dict, List, Tuple, Union
import torch
from torch import distributions, nn
import torch.nn.functional as F
from sklearn.base import BaseEstimator
from pytorch_forecasting.data.encoders import TorchNormalizer, softplus_inv
from pytorch_forecasting.metrics.base_metrics import MultivariateDistributionLoss
from pytorch_forecasting.metrics import MultivariateNormalDistributionLoss
from general_lr_multivariate_normal import GeneralLowRankMultivariateNormal
import numpy as np
import math

from pyro.distributions import MultivariateStudentT


def toeplitz(c, r):
    vals = torch.cat((r, c[1:].flip(0)))
    shape = len(c), len(r)
    i, j = torch.ones(*shape).nonzero().T
    return vals[j-i].reshape(*shape)


class MultivariateStudentTDistributionLoss(MultivariateDistributionLoss):
    """
    Multivariate low-rank normal distribution loss.

    Use this loss to make out of a DeepAR model a DeepVAR network.
    """

    distribution_class = MultivariateStudentT

    def __init__(
        self,
        name: str = None,
        quantiles: List[float] = [0.02, 0.1, 0.25, 0.5, 0.75, 0.9, 0.98],
        reduction: str = "mean",
        rank: int = 10,
        sigma_init: float = 1.0,
        sigma_minimum: float = 1e-3,
    ):
        """
        Initialize metric

        Args:
            name (str): metric name. Defaults to class name.
            quantiles (List[float], optional): quantiles for probability range.
                Defaults to [0.02, 0.1, 0.25, 0.5, 0.75, 0.9, 0.98].
            reduction (str, optional): Reduction, "none", "mean" or "sqrt-mean". Defaults to "mean".
            rank (int): rank of low-rank approximation for covariance matrix. Defaults to 10.
            sigma_init (float, optional): default value for diagonal covariance. Defaults to 1.0.
            sigma_minimum (float, optional): minimum value for diagonal covariance. Defaults to 1e-3.
        """
        super().__init__(name=name, quantiles=quantiles, reduction=reduction)
        self.rank = rank
        self.sigma_minimum = sigma_minimum
        self.sigma_init = sigma_init
        self.distribution_arguments = list(range(3 + rank))

        # determine bias
        self._diag_bias: float = (
            softplus_inv(torch.tensor(self.sigma_init) ** 2).item() if self.sigma_init > 0.0 else 0.0
        )
        # determine normalizer to bring unscaled diagonal close to 1.0
        self._cov_factor_scale: float = np.sqrt(self.rank)

    def map_x_to_distribution(self, x: torch.Tensor) -> distributions.Normal:
        x = x.permute(1, 0, 2) # (Q, B, 2+2+R)

        cov = x[..., 5:]@x[..., 5:].mT + torch.diag_embed(x[..., 3]) # (Q, B, B)
        scale_tril = torch.cholesky(cov) # (Q, B, B)

        distr = self.distribution_class(
            df=x[..., 4].mean(-1), # (Q, B)
            loc=x[..., 2], # (Q, B)
            scale_tril=scale_tril  # (Q, B, B) lower triangular matrix with positive diagonal entries
        )

        # scaler = AffineTransform(loc=x[0, :, 0], scale=x[0, :, 1], event_dim=1)
        # if self._transformation is None:
        #     return TransformedDistribution(distr, [scaler])
        # else:
        #     return distributions.TransformedDistribution(
        #         distr, [scaler, TorchNormalizer.get_transform(self._transformation)["inverse_torch"]]
        #     )

        scaler = distributions.AffineTransform(loc=x[0, :, 0], scale=x[0, :, 1], event_dim=1)
        if self._transformation is None:
            return distributions.TransformedDistribution(distr, [scaler])
        else:
            return distributions.TransformedDistribution(
                distr, [scaler, TorchNormalizer.get_transform(self._transformation)["inverse_torch"]]
            )

    def rescale_parameters(
        self, parameters: torch.Tensor, target_scale: torch.Tensor, encoder: BaseEstimator
    ) -> torch.Tensor:
        self._transformation = encoder.transformation

        # scale
        loc = parameters[..., 0].unsqueeze(-1)  # (B, Q, 1)
        scale = F.softplus(parameters[..., 1].unsqueeze(-1) + self._diag_bias) + self.sigma_minimum**2  # (B, Q, 1)
        df = F.softplus(parameters[..., 2].unsqueeze(-1)) # (B, Q, 1)

        cov_factor = parameters[..., 3:] / self._cov_factor_scale  # (B, Q, R)
        return torch.concat([target_scale.unsqueeze(1).expand(-1, loc.size(1), -1), loc, scale, df, cov_factor], dim=-1)


class BatchMGD_Kernel(MultivariateDistributionLoss):
    """
    Multivariate low-rank normal distribution loss.

    Use this loss to make out of a DeepAR model a DeepVAR network.

    Uses multiple covariance matrix for each rank
    """

    distribution_class = distributions.LowRankMultivariateNormal

    def __init__(
        self,
        name: str = None,
        quantiles: List[float] = [0.02, 0.1, 0.25, 0.5, 0.75, 0.9, 0.98],
        reduction: str = "mean",
        rank: int = 10,
        sigma_init: float = 1.0,
        sigma_minimum: float = 1e-3,
        n_layer: int = 1,
        D: int = 12,
        K_r: int = 4,  # number of mixture
        K_d: int = 1,
        delta_l: float = 1.0,  # lengthscale step size
        train_l: bool = False,
        lr: float = 1e-03,  # learning rate for loss params
        wd: float = 1e-08,
        reg_w: float = 1.0,
        static: bool = False,
        static_l: bool = True,  # whether make static length_scale learnable
        l: int = 1.0,  # static length_scale
        # batch_size: int = None,
    ):
        super().__init__(name=name, quantiles=quantiles, reduction=reduction)
        self.rank = rank
        self.sigma_minimum = sigma_minimum
        self.sigma_init = sigma_init
        self.distribution_arguments = list(range(2 + rank))

        # determine bias
        self._diag_bias: float = (
            softplus_inv(torch.tensor(self.sigma_init) ** 2).item() if self.sigma_init > 0.0 else 0.0
        )
        # determine normalizer to bring unscaled diagonal close to 1.0
        self._cov_factor_scale: float = np.sqrt(self.rank)

        self.training_distribution = GeneralLowRankMultivariateNormal
        self.predictive_distribution = distributions.MultivariateNormal

        self.batch_cov_horizon = D
        self.K_r = K_r
        self.K_d = K_d
        self.static = static
        self.static_l = static_l
        self.n_layer = n_layer
        self.lr = lr
        self.wd = wd
        self.reg_w = reg_w

        # define kernel distance and identity component I_D
        self.dist = nn.Parameter(torch.range(0, self.batch_cov_horizon-1), requires_grad=False)
        self.kernel_eye = nn.Parameter(torch.eye(self.batch_cov_horizon), requires_grad=False)

        # define I_R and I_B
        self.eye_r = nn.Parameter(torch.eye(self.rank), requires_grad=False) if self.K_r > 1 else None
        self.eye_b = nn.Parameter(torch.eye(self.batch_size), requires_grad=False) if self.K_d > 1 else None

        if self.K_r > 1 or self.K_d > 1:
            self.c_list = nn.ParameterList([nn.Parameter(torch.tensor(i+delta_l), requires_grad=train_l) for i in range(max(self.K_r, self.K_d)-1)])
        else:
            self.c_list = nn.ParameterList([nn.Parameter(torch.tensor(float(l)), requires_grad=not self.static_l)])
            if self.static:  # static adjustment with identity matrix
                self.sigma = nn.Parameter(torch.rand(1), requires_grad=True)  # TODO: change to randn, randn is worse than rand

    def map_x_to_distribution(self, x: torch.Tensor) -> distributions.Normal:
        x = x.permute(1, 0, 2)
        distr = self.distribution_class(
            loc=x[..., 2],
            cov_factor=x[..., 4:],
            cov_diag=x[..., 3],
        )
        scaler = distributions.AffineTransform(loc=x[0, :, 0], scale=x[0, :, 1], event_dim=1)
        if self._transformation is None:
            return distributions.TransformedDistribution(distr, [scaler])
        else:
            return distributions.TransformedDistribution(
                distr, [scaler, TorchNormalizer.get_transform(self._transformation)["inverse_torch"]]
            )

    def map_x_to_training_distribution(self, x: torch.Tensor) -> distributions.Normal:
        mixture_weights = x[..., len(self.distribution_arguments)+2:]
        x = x[..., :len(self.distribution_arguments)+2]
        x = x.permute(1, 0, 2)
        corr_mat_r = self.get_corr(mixture_weights[..., :self.K_r], self.K_r) if self.K_r > 1 else None
        corr_mat_d = self.get_corr(mixture_weights[..., -self.K_d:], self.K_d) if self.K_d > 1 else None

        ################### Method 1 ####################
        loc = x[..., 2].flatten().unsqueeze(0)
        cov_factor = torch.block_diag(*x[..., 4:]).unsqueeze(0)
        cov_diag = x[..., 3].flatten().unsqueeze(0)

        distr = self.training_distribution(
            loc=loc,  # (1, DB)
            cov_factor=cov_factor,  # (1, DB, DR)
            cov_diag=cov_diag,  # (1, DB)
            corr_mat=corr_mat_r,  # (D, D)
            corr_eye=self.eye_r,  # (R, R)
            reg_w=self.reg_w
        )
        scaler = distributions.AffineTransform(loc=x[..., 0].flatten(), scale=x[..., 1].flatten(), event_dim=1)

        ################### Method 2 ####################
        # loc = x[..., 2]
        # cov_factor = x[..., 4:]
        # cov_diag = x[..., 3]
        # distr = self.training_distribution(
        #     loc=loc,  # (D, B)
        #     cov_factor=cov_factor,  # (D, B, R)
        #     cov_diag=cov_diag,  # (D, B)
        #     corr_mat_r=corr_mat_r,  # (D, D)
        #     corr_mat_d=corr_mat_d,  # (D, D)
        #     eye_r=self.eye_r,  # (R, R)
        #     eye_b=self.eye_b  # (B, B)
        # )
        # scaler = distributions.AffineTransform(loc=x[0, : ,0], scale=x[0, :, 1], event_dim=1)
        #################################################

        if self._transformation is None:
            return distributions.TransformedDistribution(distr, [scaler])
        else:
            return distributions.TransformedDistribution(
                distr, [scaler, TorchNormalizer.get_transform(self._transformation)["inverse_torch"]]
            )

    def map_x_to_predictive_distribution(self, x: torch.Tensor, mu: torch.Tensor, cov: torch.Tensor) -> distributions.Normal:
        x = x.permute(1, 0, 2)
        distr = self.predictive_distribution(loc=mu, covariance_matrix=cov)

        scaler = distributions.AffineTransform(loc=x[0, :, 0], scale=x[0, :, 1], event_dim=1)
        if self._transformation is None:
            return distributions.TransformedDistribution(distr, [scaler])
        else:
            return distributions.TransformedDistribution(
                distr, [scaler, TorchNormalizer.get_transform(self._transformation)["inverse_torch"]]
            )

    def sample(self, y_pred, n_samples: int, mu=None, sigma=None) -> torch.Tensor:
        """
        Sample from distribution.

        Args:
            y_pred: prediction output of network (shape batch_size x n_timesteps x n_paramters)
            n_samples (int): number of samples to draw

        Returns:
            torch.Tensor: tensor with samples  (shape batch_size x n_timesteps x n_samples)
        """
        if mu is None:
            dist = self.map_x_to_distribution(y_pred)
        else:
            dist = self.map_x_to_predictive_distribution(y_pred, mu, sigma)
        samples = dist.sample((n_samples,)).permute(
            2, 1, 0
        )  # returned as (n_samples, n_timesteps, batch_size), so reshape to (batch_size, n_timesteps, n_samples)
        return samples

    def rescale_parameters(
        self, parameters: torch.Tensor, target_scale: torch.Tensor, encoder: BaseEstimator
    ) -> torch.Tensor:
        self._transformation = encoder.transformation

        # scale
        loc = parameters[..., 0].unsqueeze(-1)
        scale = F.softplus(parameters[..., 1].unsqueeze(-1) + self._diag_bias) + self.sigma_minimum**2

        cov_factor = parameters[..., 2:] / self._cov_factor_scale
        return torch.concat([target_scale.unsqueeze(1).expand(-1, loc.size(1), -1), loc, scale, cov_factor], dim=-1)

    def kernel_fun(self, l):
        return torch.exp(-self.dist**2/F.relu(l)**2)  #TODO: relu will have problem for learnable lengthscale

    def get_kmat(self, l):
        dist = self.kernel_fun(l)
        kmat = toeplitz(dist, dist)
        return kmat

    def get_corr(self, mixture_weights: torch.Tensor = None, K: int = None):
        corr_list = [self.get_kmat(torch.clamp(self.c_list[i], max=8.0)) for i in range(K-1)]

        if K > 1:
            corr_list += [self.kernel_eye]
            corr = sum([corr_list[i]*mixture_weights[:,-1,i].mean() for i in range(len(corr_list))])
        else:
            if self.static:
                sigma = F.sigmoid(self.sigma)
                corr = (1-sigma)*corr_list[0] + sigma*self.kernel_eye
            else:
                corr = (1-mixture_weights[:,-1:])*corr_list[0].unsqueeze(0).repeat_interleave(mixture_weights.shape[0], 0) + mixture_weights[:,-1:]*self.kernel_eye

        corr.view(-1, self.batch_cov_horizon * self.batch_cov_horizon)[:, ::self.batch_cov_horizon + 1] += self.sigma_minimum**2
        return corr

    def loss(self, y_pred: torch.Tensor, y_actual: torch.Tensor) -> torch.Tensor:
        """
        loss function: BatchCovLoss
        y_pred: (batch_size (N), Q, n_params), params for normed data
        y_actual: (batch_size (N), Q), this comes from dataloader y of (x, y), in original scale
        """
        N = y_pred.shape[1]//self.batch_cov_horizon
        y_pred = y_pred[:,:self.batch_cov_horizon*N]
        y_actual = y_actual[:,:self.batch_cov_horizon*N]
        y_actual = y_actual.reshape(y_actual.shape[0], N, self.batch_cov_horizon)

        if self.static:
            y_pred = y_pred.reshape(y_pred.shape[0], N, self.batch_cov_horizon, -1)
            loss = []
            for i in range(y_pred.shape[1]):
                loss.append(-self.map_x_to_training_distribution(y_pred[:, i]).log_prob(y_actual[:, i]).unsqueeze(-1))
            loss = torch.cat(loss, dim=-1)
        else:
            y_pred = y_pred.reshape(y_pred.shape[0], N, self.batch_cov_horizon, -1)
            loss = []
            for i in range(y_pred.shape[1]):
                # loss.append(-self.map_x_to_training_distribution(y_pred[:, i]).log_prob(y_actual[:, i].T))  # for method 1
                loss.append(-self.map_x_to_training_distribution(y_pred[:, i]).log_prob(y_actual[:, i].T.flatten().unsqueeze(0)))
            loss = torch.cat(loss, dim=0).sum()*y_actual.size(0)

        return loss.sum()


class BatchMGD_AR(BatchMGD_Kernel, MultivariateDistributionLoss):
    """
    Multivariate low-rank normal distribution loss.

    Use this loss to make out of a DeepAR model a DeepVAR network.

    Uses multiple covariance matrix for each rank
    """

    distribution_class = distributions.LowRankMultivariateNormal

    def __init__(
        self,
        name: str = None,
        quantiles: List[float] = [0.02, 0.1, 0.25, 0.5, 0.75, 0.9, 0.98],
        reduction: str = "mean",
        rank: int = 10,
        sigma_init: float = 1.0,
        sigma_minimum: float = 1e-3,
        n_layer: int = 1,
        D: int = 12,
        K_r: int = 4,  # AR order for r process
        K_d: int = 1,  # AR order for \epsilon process
        lr: float = 1e-03,  # learning rate for loss params
        wd: float = 1e-08,
        reg_w: float = 0.1,
        static: bool = False,
    ):
        super().__init__(name=name, quantiles=quantiles, reduction=reduction)
        self.rank = rank
        self.sigma_minimum = sigma_minimum
        self.sigma_init = sigma_init
        self.distribution_arguments = list(range(2 + rank))

        # determine bias
        self._diag_bias: float = (
            softplus_inv(torch.tensor(self.sigma_init) ** 2).item() if self.sigma_init > 0.0 else 0.0
        )
        # determine normalizer to bring unscaled diagonal close to 1.0
        self._cov_factor_scale: float = np.sqrt(self.rank)

        self.training_distribution = GeneralLowRankMultivariateNormal
        self.predictive_distribution = distributions.MultivariateNormal

        self.batch_cov_horizon = D
        self.K_r = K_r
        self.K_d = K_d
        self.static = static
        self.n_layer = n_layer
        self.lr = lr
        self.wd = wd
        self.reg_w = reg_w

        self.kernel_eye = nn.Parameter(torch.eye(self.batch_cov_horizon), requires_grad=False)
        # define I_R and I_B
        self.eye_r = nn.Parameter(torch.eye(self.rank), requires_grad=False) if self.K_r > 1 else None


    def get_kmat(self, ar_coef):
        order = ar_coef.shape[0]
        rhos = [ar_coef[0]/ar_coef[0]]
        if order == 1:
            for i in range(self.batch_cov_horizon-1):
                rhos.append(ar_coef[0]*rhos[-1])
        elif order == 2:
            for i in range(self.batch_cov_horizon-1):
                if i == 0:
                    rhos.append(ar_coef[0]/(1-ar_coef[1]))
                else:
                    rhos.append(ar_coef[0]*rhos[-1] + ar_coef[1]*rhos[-2])
        elif order == 3:
            for i in range(self.batch_cov_horizon-1):
                if i == 0:
                    rhos.append((ar_coef[0] + ar_coef[1]*ar_coef[2])/(1-ar_coef[1]-ar_coef[2]*(ar_coef[0] + ar_coef[2])))
                elif i == 1:
                    rhos.append(ar_coef[1] + (ar_coef[0] + ar_coef[2])*rhos[-1])
                else:
                    rhos.append(ar_coef[0]*rhos[-1] + ar_coef[1]*rhos[-2] + ar_coef[2]*rhos[-3])
        else:
            raise ValueError("AR(p) order should be less than 4")

        rhos = torch.stack(rhos)
        kmat = toeplitz(rhos, rhos)

        return kmat

    def get_corr(self, mixture_weights: torch.Tensor = None, K: int = None):
        # w, ar_coef = mixture_weights[:,-1].mean(0)[0].abs(), mixture_weights[:,-1].mean(0)[1:]
        # w, ar_coef = mixture_weights[:,-1].mean(0)[0], mixture_weights[:,-1].mean(0)[1:]
        ar_coef = mixture_weights[:,-1].mean(0)
        corr = self.get_kmat(ar_coef)
        # corr = (1-w)*corr + w*self.kernel_eye
        corr.view(-1, self.batch_cov_horizon * self.batch_cov_horizon)[:, ::self.batch_cov_horizon + 1] += self.sigma_minimum**2
        return corr

    def loss(self, y_pred: torch.Tensor, y_actual: torch.Tensor) -> torch.Tensor:
        """
        loss function: BatchCovLoss
        y_pred: (batch_size (N), Q, n_params), params for normed data
        y_actual: (batch_size (N), Q), this comes from dataloader y of (x, y), in original scale
        """
        N = y_pred.shape[1]//self.batch_cov_horizon
        y_pred = y_pred[:,:self.batch_cov_horizon*N]
        y_actual = y_actual[:,:self.batch_cov_horizon*N]
        y_actual = y_actual.reshape(y_actual.shape[0], N, self.batch_cov_horizon)

        if self.static:
            y_pred = y_pred.reshape(y_pred.shape[0], N, self.batch_cov_horizon, -1)
            loss = []
            for i in range(y_pred.shape[1]):
                loss.append(-self.map_x_to_training_distribution(y_pred[:, i]).log_prob(y_actual[:, i]).unsqueeze(-1))
            loss = torch.cat(loss, dim=-1)
        else:
            y_pred = y_pred.reshape(y_pred.shape[0], N, self.batch_cov_horizon, -1)
            loss = []
            for i in range(y_pred.shape[1]):
                if self.reg_w != 0:
                    loss.append(-self.map_x_to_training_distribution(y_pred[:, i]).log_prob(y_actual[:, i].T.flatten().unsqueeze(0)) + self.reg_w*torch.pow(y_pred[:,i,...,len(self.distribution_arguments)+2:], 2).sum())
                else:
                    loss.append(-self.map_x_to_training_distribution(y_pred[:, i]).log_prob(y_actual[:, i].T.flatten().unsqueeze(0)))
                # loss.append(-self.map_x_to_training_distribution(y_pred[:, i]).log_prob(y_actual[:, i].T.flatten().unsqueeze(0)))
            loss = torch.cat(loss, dim=0).sum()*y_actual.size(0)

        return loss.sum()


class BatchMGD_Toeplitz(MultivariateDistributionLoss):
    distribution_class = distributions.LowRankMultivariateNormal

    def __init__(
        self,
        name: str = None,
        quantiles: List[float] = [0.02, 0.1, 0.25, 0.5, 0.75, 0.9, 0.98],
        reduction: str = "mean",
        rank: int = 10,
        sigma_init: float = 1.0,
        sigma_minimum: float = 1e-3,
        D: int = 12,
        K: int = 1,  # number of mixturex
        lr: float = 0.001,  # individual learning rate
        static: bool = True,
        static_l: bool = True,
    ):
        super().__init__(name=name, quantiles=quantiles, reduction=reduction)
        self.rank = rank
        self.sigma_minimum = sigma_minimum
        self.sigma_init = sigma_init
        self.distribution_arguments = list(range(2 + rank))

        self.batch_distribution = distributions.MultivariateNormal

        self.lr = lr
        self.batch_cov_horizon = D
        self.K = K
        self.static = static
        self.static_l = static_l

        if self.K > 1:  # dynamic mixture
            self.c_list = nn.ParameterList([nn.Parameter(torch.randn(2*D-1), requires_grad=True) for i in range(K)])
        else:
            self.c_list = nn.ParameterList([nn.Parameter(torch.randn(2*D-1), requires_grad=not self.static_l)])

            if self.static:  # static adjustment with identity matrix
                self.sigma = nn.Parameter(torch.rand(1), requires_grad=True)  # TODO: change to randn, randn is worse than rand

        # determine bias
        self._diag_bias: float = (
            softplus_inv(torch.tensor(self.sigma_init) ** 2).item() if self.sigma_init > 0.0 else 0.0
        )
        # determine normalizer to bring unscaled diagonal close to 1.0
        self._cov_factor_scale: float = np.sqrt(self.rank)

    def map_x_to_distribution(self, x: torch.Tensor) -> distributions.Normal:
        x = x.permute(1, 0, 2)
        distr = self.distribution_class(
            loc=x[..., 2],
            cov_factor=x[..., 4:],
            cov_diag=x[..., 3],
        )
        scaler = distributions.AffineTransform(loc=x[0, :, 0], scale=x[0, :, 1], event_dim=1)
        if self._transformation is None:
            return distributions.TransformedDistribution(distr, [scaler])
        else:
            return distributions.TransformedDistribution(
                distr, [scaler, TorchNormalizer.get_transform(self._transformation)["inverse_torch"]]
            )

    def map_x_to_distribution_batch(self, x: torch.Tensor, mixture_weights: torch.Tensor = None) -> distributions.Normal:
        corr = self.get_corr(mixture_weights)

        x = x.permute(1, 0, 2)   # (B, D, (4+R)) >> (D, B, (4+R))

        cov_mat = torch.block_diag(*x[..., 4:])@torch.kron(corr, torch.eye(self.rank, device=corr.device))@torch.block_diag(*x[..., 4:].mT) + torch.block_diag(*torch.diag_embed(x[..., 3]))

        distr = self.batch_distribution(loc=x[..., 2].flatten().unsqueeze(0), covariance_matrix=cov_mat.unsqueeze(0))

        scaler = distributions.AffineTransform(loc=x[..., 0].flatten(), scale=x[..., 1].flatten(), event_dim=1)
        if self._transformation is None:
            return distributions.TransformedDistribution(distr, [scaler])
        else:
            return distributions.TransformedDistribution(
                distr, [scaler, TorchNormalizer.get_transform(self._transformation)["inverse_torch"]]
            )

    def map_x_to_distribution_cov(self, x: torch.Tensor, mu: torch.Tensor, cov: torch.Tensor) -> distributions.Normal:
        x = x.permute(1, 0, 2)
        distr = self.batch_distribution(loc=mu, covariance_matrix=cov)

        scaler = distributions.AffineTransform(loc=x[0, :, 0], scale=x[0, :, 1], event_dim=1)
        if self._transformation is None:
            return distributions.TransformedDistribution(distr, [scaler])
        else:
            return distributions.TransformedDistribution(
                distr, [scaler, TorchNormalizer.get_transform(self._transformation)["inverse_torch"]]
            )

    def sample(self, prediction_parameters, n_samples: int, mu=None, sigma=None) -> torch.Tensor:
        """
        Sample from distribution.

        Args:
            y_pred: prediction output of network (shape batch_size x n_timesteps x n_paramters)
            n_samples (int): number of samples to draw

        Returns:
            torch.Tensor: tensor with samples  (shape batch_size x n_timesteps x n_samples)
        """
        if mu is None:
            dist = self.map_x_to_distribution(prediction_parameters)
        else:
            dist = self.map_x_to_distribution_cov(prediction_parameters, mu, sigma)
        samples = dist.sample((n_samples,)).permute(
            2, 1, 0
        )  # returned as (n_samples, n_timesteps, batch_size), so reshape to (batch_size, n_timesteps, n_samples)
        return samples

    def rescale_parameters(
        self, parameters: torch.Tensor, target_scale: torch.Tensor, encoder: BaseEstimator
    ) -> torch.Tensor:
        self._transformation = encoder.transformation

        # scale
        loc = parameters[..., 0].unsqueeze(-1)
        scale = F.softplus(parameters[..., 1].unsqueeze(-1) + self._diag_bias) + self.sigma_minimum**2

        cov_factor = parameters[..., 2:] / self._cov_factor_scale
        return torch.concat([target_scale.unsqueeze(1).expand(-1, loc.size(1), -1), loc, scale, cov_factor], dim=-1)

    def get_kmat(self, c):
        c = torch.fft.irfft(torch.mul(torch.conj(torch.fft.rfft(c)), torch.fft.rfft(c)))
        c = c[:self.batch_cov_horizon]
        c = c/c.max()
        corr = toeplitz(c, c)
        return corr

    def get_corr(self, mixture_weights: torch.Tensor = None):
        corr_list = [self.get_kmat(self.c_list[i]) for i in range(self.K)]

        if self.K > 1:
            corr = sum([corr_list[i]*mixture_weights[:,-1,i].mean() for i in range(len(corr_list))])
        else:
            if self.static:
                sigma = F.sigmoid(self.sigma)
                corr = (1-sigma)*corr_list[0] + sigma*self.identity
            else:
                corr = (1-mixture_weights[:,-1:])*corr_list[0].unsqueeze(0).repeat_interleave(mixture_weights.shape[0], 0) + mixture_weights[:,-1:]*self.identity

        return corr

    def loss(self, y_pred: torch.Tensor, y_actual: torch.Tensor) -> torch.Tensor:
        """
        loss function: BatchCovLoss
        y_pred: (batch_size (N), Q, n_params), params for normed data
        y_actual: (batch_size (N), Q), this comes from dataloader y of (x, y), in original scale
        """
        N = y_pred.shape[1]//self.batch_cov_horizon
        y_pred = y_pred[:,:self.batch_cov_horizon*N]
        y_actual = y_actual[:,:self.batch_cov_horizon*N]
        y_actual = y_actual.reshape(y_actual.shape[0], N, self.batch_cov_horizon)

        if self.static:
            y_pred = y_pred.reshape(y_pred.shape[0], N, self.batch_cov_horizon, -1)
            loss = []
            for i in range(y_pred.shape[1]):
                loss.append(-self.map_x_to_distribution_batch(y_pred[:, i]).log_prob(y_actual[:, i]).unsqueeze(-1))
            loss = torch.cat(loss, dim=-1)
        else:
            mixture_weights = y_pred[..., -self.K:]
            y_pred = y_pred[..., :-self.K]

            mixture_weights = mixture_weights[:,:self.batch_cov_horizon*N]
            mixture_weights = mixture_weights.reshape(mixture_weights.shape[0], N, self.batch_cov_horizon, -1)
            y_pred = y_pred.reshape(y_pred.shape[0], N, self.batch_cov_horizon, -1)

            loss = []
            for i in range(y_pred.shape[1]):
                loss.append(-self.map_x_to_distribution_batch(y_pred[:, i], mixture_weights[:, i]).log_prob(y_actual[:, i].T.flatten().unsqueeze(0)))
            loss = torch.cat(loss, dim=0).sum()*y_actual.size(0)

        return loss.sum()


