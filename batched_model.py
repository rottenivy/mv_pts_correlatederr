from typing import Any, Callable, Dict, List, Tuple, Union

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from tqdm.autonotebook import tqdm
import matplotlib.pyplot as plt

from pytorch_forecasting import DeepAR
from pytorch_forecasting.models.base_model import AutoRegressiveBaseModelWithCovariates
from pytorch_forecasting.models.nn import HiddenState
from pytorch_forecasting.models.base_model import _torch_cat_na, _concatenate_output
from pytorch_forecasting.metrics.base_metrics import DistributionLoss
from pytorch_forecasting.metrics import MultiLoss
from pytorch_forecasting.utils import apply_to_list, to_list

from pytorch_forecasting.optim import Ranger
from torch.optim.lr_scheduler import LambdaLR, ReduceLROnPlateau

from pytorch_forecasting.data import TimeSeriesDataSet
from pytorch_forecasting.metrics import (
    MAE,
    MAPE,
    MASE,
    RMSE,
    SMAPE,
    DistributionLoss,
    Metric,
    MultiLoss,
    MultivariateDistributionLoss,
    NormalDistributionLoss,
)
from pytorch_forecasting.optim import Ranger
from pytorch_forecasting.utils import (
    apply_to_list,
    create_mask,
    move_to_device,
    to_list,
)

from model import ARTransformer


class BatchedEstimator(AutoRegressiveBaseModelWithCovariates):
    def __init__(
        self,
        **kwargs
    ):
        super().__init__(**kwargs)

        if self.loss.name == 'BatchMGD_Kernel':
            if self.loss.K_r > 1:
                self.mixture_projector_r = nn.Sequential(nn.Linear(self.hparams.hidden_size, self.loss.K_r), nn.Softmax(dim=-1)) if self.loss.n_layer == 1 else nn.Sequential(nn.Linear(self.hparams.hidden_size, int(self.hparams.hidden_size/2)), nn.ELU(), nn.Linear(int(self.hparams.hidden_size/2), self.loss.K_r), nn.Softmax(dim=-1))
            if self.loss.K_d > 1:
                self.mixture_projector_d = nn.Sequential(nn.Linear(self.hparams.hidden_size, self.loss.K_d), nn.Softmax(dim=-1)) if self.loss.n_layer == 1 else nn.Sequential(nn.Linear(self.hparams.hidden_size, int(self.hparams.hidden_size/2)), nn.ELU(), nn.Linear(int(self.hparams.hidden_size/2), self.loss.K_d), nn.Softmax(dim=-1))
        elif self.loss.name == 'BatchMGD_AR':
            if self.loss.K_r > 1:
                self.mixture_projector_r = nn.Sequential(nn.Linear(self.hparams.hidden_size, self.loss.K_r-1), nn.Tanh()) if self.loss.n_layer == 1 else nn.Sequential(nn.Linear(self.hparams.hidden_size, int(self.hparams.hidden_size/2)), nn.ELU(), nn.Linear(int(self.hparams.hidden_size/2), self.loss.K_r-1), nn.Tanh())
            if self.loss.K_d > 1:
                self.mixture_projector_d = nn.Sequential(nn.Linear(self.hparams.hidden_size, self.loss.K_d-1),  nn.Tanh()) if self.loss.n_layer == 1 else nn.Sequential(nn.Linear(self.hparams.hidden_size, int(self.hparams.hidden_size/2)), nn.ELU(), nn.Linear(int(self.hparams.hidden_size/2), self.loss.K_d-1),  nn.Tanh())

    def configure_optimizers(self):
            # either set a schedule of lrs or find it dynamically
            if self.hparams.optimizer_params is None:
                optimizer_params = {}
            else:
                optimizer_params = self.hparams.optimizer_params
            # set optimizer
            lrs = self.hparams.learning_rate
            if isinstance(lrs, (list, tuple)):
                lr = lrs[0]
            else:
                lr = lrs

            # assign parameter groups
            params = list(self.named_parameters())

            # grouped_parameters = [
            # {"params": [p for n, p in params if n.split('.')[0] in ['loss', 'mixture_projector_r']], 'lr': self.loss.lr, 'weight_decay': self.loss.wd},  # TODO add mixture projecter to this
            # {"params": [p for n, p in params if n.split('.')[0] not in ['loss', 'mixture_projector_r']], 'lr': lr, 'weight_decay': self.hparams.weight_decay}]

            grouped_parameters = [
            {"params": [p for n, p in params if n.split('.')[0] in ['loss']], 'lr': self.loss.lr, 'weight_decay': self.loss.wd},  # TODO add mixture projecter to this
            {"params": [p for n, p in params if n.split('.')[0] not in ['loss']], 'lr': lr, 'weight_decay': self.hparams.weight_decay}]
            
            if callable(self.optimizer):
                try:
                    optimizer = self.optimizer(
                        grouped_parameters, lr=lr, weight_decay=self.hparams.weight_decay, **optimizer_params
                    )
                except TypeError:  # in case there is no weight decay
                    optimizer = self.optimizer(grouped_parameters, lr=lr, **optimizer_params)
            elif self.hparams.optimizer == "adam":
                optimizer = torch.optim.Adam(
                    grouped_parameters, lr=lr, weight_decay=self.hparams.weight_decay, **optimizer_params
                )
            elif self.hparams.optimizer == "adamw":
                optimizer = torch.optim.AdamW(
                    grouped_parameters, lr=lr, weight_decay=self.hparams.weight_decay, **optimizer_params
                )
            elif self.hparams.optimizer == "ranger":
                optimizer = Ranger(grouped_parameters, lr=lr, weight_decay=self.hparams.weight_decay, **optimizer_params)
            elif self.hparams.optimizer == "sgd":
                optimizer = torch.optim.SGD(
                    grouped_parameters, lr=lr, weight_decay=self.hparams.weight_decay, **optimizer_params
                )
            elif hasattr(torch.optim, self.hparams.optimizer):
                try:
                    optimizer = getattr(torch.optim, self.hparams.optimizer)(
                        grouped_parameters, lr=lr, weight_decay=self.hparams.weight_decay, **optimizer_params
                    )
                except TypeError:  # in case there is no weight decay
                    optimizer = getattr(torch.optim, self.hparams.optimizer)(grouped_parameters, lr=lr, **optimizer_params)
            else:
                raise ValueError(f"Optimizer of self.hparams.optimizer={self.hparams.optimizer} unknown")

            # set scheduler
            if isinstance(lrs, (list, tuple)):  # change for each epoch
                # normalize lrs
                lrs = np.array(lrs) / lrs[0]
                scheduler_config = {
                    "scheduler": LambdaLR(optimizer, lambda epoch: lrs[min(epoch, len(lrs) - 1)]),
                    "interval": "epoch",
                    "frequency": 1,
                    "strict": False,
                }
            elif self.hparams.reduce_on_plateau_patience is None:
                scheduler_config = {}
            else:  # find schedule based on validation loss
                scheduler_config = {
                    "scheduler": ReduceLROnPlateau(
                        optimizer,
                        mode="min",
                        factor=1.0 / self.hparams.reduce_on_plateau_reduction,
                        patience=self.hparams.reduce_on_plateau_patience,
                        cooldown=self.hparams.reduce_on_plateau_patience,
                        min_lr=self.hparams.reduce_on_plateau_min_lr,
                    ),
                    "monitor": "val_loss",  # Default: val_loss
                    "interval": "epoch",
                    "frequency": 1,
                    "strict": False,
                }

            return {"optimizer": optimizer, "lr_scheduler": scheduler_config}

    def get_dynamic_weights(self, decoder_output: torch.Tensor):
        """weights for r and e have separate projectors"""
        if self.loss.K_r > 1 and self.loss.K_d > 1:
            mixture_weights_r = self.mixture_projector_r(decoder_output)
            mixture_weights_d = self.mixture_projector_d(decoder_output)
            mixture_weights = torch.cat([mixture_weights_r, mixture_weights_d], dim=-1)
        elif self.loss.K_r > 1:
            mixture_weights = self.mixture_projector_r(decoder_output)
        elif self.loss.K_d > 1:
            mixture_weights = self.mixture_projector_d(decoder_output)

        return mixture_weights


class BatchedPredictor(AutoRegressiveBaseModelWithCovariates):
    def get_cond_cov(self, current_decoder_output, x, n_samples):
        N = x.shape[2]
        # N = prediction_params_all.shape[0]
        # x = prediction_params_all.permute(1, 2, 0, 3)

        mixture_weights = self.get_dynamic_weights(current_decoder_output)  # (B*n_sample, 1, H) -> (B*n_sample, 1, 2)
        if self.loss.K_r > 1 and self.loss.K_d > 1:
            corr_mat_r = self.loss.get_corr(mixture_weights[..., :self.loss.K_r], self.loss.K_r)
            corr_mat_d = self.loss.get_corr(mixture_weights[..., -self.loss.K_d:], self.loss.K_d)
        elif self.loss.K_r > 1:
            corr_mat_r = self.loss.get_corr(mixture_weights, self.loss.K_r)
        elif self.loss.K_d > 1:
            corr_mat_d = self.loss.get_corr(mixture_weights, self.loss.K_d)

        cov_fac = torch.stack([torch.block_diag(*x[i,...,4:]) for i in range(n_samples)])
        diag_fac = torch.stack([torch.block_diag(*torch.diag_embed(x[i,...,3])) for i in range(n_samples)])

        D_inv = torch.diag_embed((1/x[:,:-1,...,3]).flatten(start_dim=1))
        A = cov_fac[:, :(self.loss.batch_cov_horizon-1)*N, :(self.loss.batch_cov_horizon-1)*self.loss.rank]
        C_inv = torch.kron(torch.inverse(corr_mat_r[:self.loss.batch_cov_horizon-1, :self.loss.batch_cov_horizon-1]).contiguous(), self.loss.eye)
        cov_11_inv = D_inv - D_inv@A@torch.inverse(C_inv+A.mT@D_inv@A)@A.mT@D_inv

        Sigma = cov_fac@torch.kron(corr_mat_r, self.loss.eye)@cov_fac.mT + diag_fac
        cov_21 = Sigma[..., -N:, :-N]

        return mixture_weights, cov_21, cov_11_inv

    def get_cond_cov_fast(self, current_decoder_output, x, n_samples):
        mixture_weights = self.get_dynamic_weights(current_decoder_output)  # (B*n_sample, 1, H) -> (B*n_sample, 1, 2)
        
        if self.loss.K_r > 1 and self.loss.K_d > 1:
            corr_mat_r = self.loss.get_corr(mixture_weights[..., :self.loss.K_r], self.loss.K_r)
            corr_mat_d = self.loss.get_corr(mixture_weights[..., -self.loss.K_d:], self.loss.K_d)
        elif self.loss.K_r > 1:
            corr_mat_r = self.loss.get_corr(mixture_weights, self.loss.K_r)
        elif self.loss.K_d > 1:
            corr_mat_d = self.loss.get_corr(mixture_weights, self.loss.K_d)

        if self.loss.K_r > 1 and self.loss.K_d > 1:
            D_inv = torch.kron(torch.linalg.inv(corr_mat_d[:self.batch_cov_horizon-1, :self.batch_cov_horizon-1]).contiguous(), self.eye_b)@torch.diag_embed((1/x[:,:-1,...,3]).flatten(start_dim=1))
            
            A = cov_fac[:, :(self.batch_cov_horizon-1)*N, :(self.batch_cov_horizon-1)*self.rank]
            C_inv = torch.kron(torch.inverse(corr_mat_r[:self.batch_cov_horizon-1, :self.batch_cov_horizon-1]).contiguous(), self.eye_r)
            cov_11_inv = D_inv - D_inv@A@torch.inverse(C_inv+A.mT@D_inv@A)@A.mT@D_inv

            Sigma = cov_fac@torch.kron(corr_mat_r, self.eye_r)@cov_fac.mT + diag_fac@torch.kron(corr_mat_d, self.eye_b)
        elif self.loss.K_r > 1:
            # calculate cov_11_inv
            # calculate capacitance_tril_11
            A11, D11 = x[:,:-1,...,4:], x[:,:-1,...,3]
            A11t_D11inv = A11.mT / D11.unsqueeze(-2)

            AtDinvA11 = torch.cat([torch.block_diag(*torch.matmul(A11t_D11inv[i], A11[i])).unsqueeze(0) for i in range(n_samples)])

            Lcap11 = torch.linalg.cholesky(torch.kron(torch.linalg.inv(corr_mat_r[:-1, :-1]).contiguous(), self.loss.eye_r) + AtDinvA11)  # (n_sample, DR, DR)

            A11t_D11inv = torch.cat([torch.block_diag(*A11t_D11inv[i]).unsqueeze(0) for i in range(n_samples)])

            Lcapt_At_Dinv_11 = torch.linalg.inv(Lcap11)@A11t_D11inv
            cov_11_inv = -Lcapt_At_Dinv_11.mT@Lcapt_At_Dinv_11  # the memory bottleneck, (n_sample, (D-1)B, (D-1)B)
            m = cov_11_inv.shape[-1]
            cov_11_inv.view(-1, m * m)[:, ::m + 1] += 1/D11.flatten(start_dim=1)

            # calculate cov_21
            cov_21 = torch.cat([corr_mat_r[-1-i, 0]*x[:,-1,...,4:]@x[:,i,...,4:].mT for i in range(self.loss.batch_cov_horizon-1)], dim=-1)
        elif self.loss.K_d > 1:
            D_inv = torch.kron(torch.linalg.inv(corr_mat_d[:self.batch_cov_horizon-1, :self.batch_cov_horizon-1]).contiguous(), self.eye_b)@torch.diag_embed((1/x[:,:-1,...,3]).flatten(start_dim=1))

            A = cov_fac[:, :(self.batch_cov_horizon-1)*N, :(self.batch_cov_horizon-1)*self.rank]
            m = A.size(-1)
            capacitance = A.mT@D_inv@A
            capacitance.view(-1, m * m)[:, ::m + 1] += 1
            cov_11_inv = D_inv - D_inv@A@torch.inverse(capacitance)@A.mT@D_inv

            Sigma = cov_fac@cov_fac.mT + diag_fac@torch.kron(corr_mat_d, self.eye_b)

        return mixture_weights, cov_21, cov_11_inv

    def output_to_prediction(
        self,
        normalized_prediction_parameters: torch.Tensor,
        target_scale: Union[List[torch.Tensor], torch.Tensor],
        n_samples: int = 1,
        pre_normed_prediction_params: torch.Tensor = None,
        x_1: torch.Tensor = None,  # pre_normed_outputs
        current_decoder_output: torch.Tensor = None,
        **kwargs,
        ) -> Tuple[Union[List[torch.Tensor], torch.Tensor], torch.Tensor]:
        
        single_prediction = to_list(normalized_prediction_parameters)[0].ndim == 2
        
        if single_prediction:  # add time dimension as it is expected
            normalized_prediction_parameters = apply_to_list(normalized_prediction_parameters, lambda x: x.unsqueeze(1))
        # transform into real space
        
        prediction_parameters = self.transform_output(
            prediction=normalized_prediction_parameters, target_scale=target_scale, **kwargs
        )

        ######################## faster GPR #####################
        pre_prediction_parameters = self.transform_output(
        prediction=pre_normed_prediction_params, target_scale=target_scale, **kwargs
        )
        prediction_params_all = torch.cat([pre_prediction_parameters, prediction_parameters], dim=1)
        prediction_params_all = prediction_params_all.reshape(prediction_params_all.shape[0]//n_samples, n_samples, *prediction_params_all.shape[1:])
        x_1 = x_1.reshape(x_1.shape[0]//n_samples, n_samples, *x_1.shape[1:]).permute(1, 2, 0, 3)

        mu_1 = prediction_params_all[..., :-1, 2:3].permute(1, 2, 0, 3)
        mu_2 = prediction_params_all[..., -1:, 2:3].permute(1, 2, 0, 3).flatten(start_dim=1, end_dim=2)
        # use all avaiable observations
        epsilon = (x_1 - mu_1).flatten(start_dim=1, end_dim=2)

        x = prediction_params_all.permute(1, 2, 0, 3)
        N = x.shape[2]

        # calculate cov_22
        cov_22 = x[:,-1,...,4:]@x[:,-1,...,4:].mT
        cov_22.view(-1, N * N)[:, ::N + 1] += x[:,-1,...,3]

        if self.wReg:
            mixture_weights, cov_21, cov_11_inv = self.get_cond_cov_fast(current_decoder_output, x, n_samples)

            mu_21 = mu_2 + cov_21@cov_11_inv@epsilon
            sigma_21 = cov_22 - cov_21@cov_11_inv@cov_21.mT

            valid = torch.linalg.cholesky_ex(sigma_21).info.eq(0)
            if not torch.all(valid):
                if sigma_21[valid].shape[0] >= sigma_21[~valid].shape[0]:
                    sigma_21[~valid] = sigma_21[valid][:sigma_21[~valid].shape[0]]
                else:
                    mu_21 = mu_2
                    sigma_21 = cov_22
        else:
            mu_21 = mu_2
            sigma_21 = cov_22

        ######################## faster GPR #####################
        # todo: handle classification
        # sample value(s) from distribution and  select first sample
        if isinstance(self.loss, DistributionLoss) or (
            isinstance(self.loss, MultiLoss) and isinstance(self.loss[0], DistributionLoss)
        ):
            # todo: handle mixed losses
            if n_samples > 1:
                prediction_parameters = apply_to_list(
                    prediction_parameters, lambda x: x.reshape(int(x.size(0) / n_samples), n_samples, -1)
                )
                prediction = self.loss.sample(prediction_parameters, 1, mu_21.squeeze(-1), sigma_21)
                prediction = apply_to_list(prediction, lambda x: x.reshape(x.size(0) * n_samples, 1, -1))
            else:
                prediction = self.loss.sample(normalized_prediction_parameters, 1)

        else:
            prediction = prediction_parameters
        # normalize prediction prediction
        normalized_prediction = self.output_transformer.transform(prediction, target_scale=target_scale)
        if isinstance(normalized_prediction, list):
            input_target = torch.cat(normalized_prediction, dim=-1)
        else:
            input_target = normalized_prediction  # set next input target to normalized prediction

        # remove time dimension
        if single_prediction:
            prediction = apply_to_list(prediction, lambda x: x.squeeze(1))  # predictive samples in the original scale 
            input_target = apply_to_list(input_target, lambda x: x.squeeze(1))  # scaled predictive samples
            mixture_weights = apply_to_list(mixture_weights, lambda x: x.squeeze(1))
            prediction_parameters = apply_to_list(prediction_parameters, lambda x: x.squeeze(1))

        if self.loss.static:
            return prediction, input_target, torch.ones_like(prediction, device=prediction.device)
        else:
            return prediction, input_target, mixture_weights, prediction_parameters

    def predict(
        self,
        data: Union[DataLoader, pd.DataFrame, TimeSeriesDataSet],
        mode: Union[str, Tuple[str, str]] = "prediction",
        return_index: bool = False,
        return_decoder_lengths: bool = False,
        batch_size: int = 64,
        num_workers: int = 0,
        fast_dev_run: bool = False,
        show_progress_bar: bool = False,
        return_x: bool = False,
        return_w: bool = False,
        return_actual: bool = False,
        return_param: bool = False,
        return_scale: bool = False,
        mode_kwargs: Dict[str, Any] = None,
        **kwargs,
    ):
        """
        Run inference / prediction.

        Args:
            dataloader: dataloader, dataframe or dataset
            mode: one of "prediction", "quantiles", or "raw", or tuple ``("raw", output_name)`` where output_name is
                a name in the dictionary returned by ``forward()``
            return_index: if to return the prediction index (in the same order as the output, i.e. the row of the
                dataframe corresponds to the first dimension of the output and the given time index is the time index
                of the first prediction)
            return_decoder_lengths: if to return decoder_lengths (in the same order as the output
            batch_size: batch size for dataloader - only used if data is not a dataloader is passed
            num_workers: number of workers for dataloader - only used if data is not a dataloader is passed
            fast_dev_run: if to only return results of first batch
            show_progress_bar: if to show progress bar. Defaults to False.
            return_x: if to return network inputs (in the same order as prediction output)
            mode_kwargs (Dict[str, Any]): keyword arguments for ``to_prediction()`` or ``to_quantiles()``
                for modes "prediction" and "quantiles"
            **kwargs: additional arguments to network's forward method

        Returns:
            output, x, index, decoder_lengths: some elements might not be present depending on what is configured
                to be returned
        """
        # convert to dataloader
        if isinstance(data, pd.DataFrame):
            data = TimeSeriesDataSet.from_parameters(self.dataset_parameters, data, predict=True)
        if isinstance(data, TimeSeriesDataSet):
            dataloader = data.to_dataloader(batch_size=batch_size, train=False, num_workers=num_workers)
        else:
            dataloader = data

        # mode kwargs default to None
        if mode_kwargs is None:
            mode_kwargs = {}

        # ensure passed dataloader is correct
        assert isinstance(dataloader.dataset, TimeSeriesDataSet), "dataset behind dataloader mut be TimeSeriesDataSet"

        # prepare model
        self.eval()  # no dropout, etc. no gradients

        # run predictions
        output = []
        decode_lenghts = []
        x_list = []
        index = []
        w_list = []
        param_list = []
        actual = []
        target_scale = []
        progress_bar = tqdm(desc="Predict", unit=" batches", total=len(dataloader), disable=not show_progress_bar)
        with torch.no_grad():
            for x, y in dataloader:
                # move data to appropriate device
                data_device = x["encoder_cont"].device
                if data_device != self.device:
                    x = move_to_device(x, self.device)
                    y = move_to_device(y, self.device)

                # make prediction
                out, w, pred_param = self(x, **kwargs)  # raw output is dictionary

                lengths = x["decoder_lengths"]
                if return_decoder_lengths:
                    decode_lenghts.append(lengths)
                nan_mask = create_mask(lengths.max(), lengths)
                if isinstance(mode, (tuple, list)):
                    if mode[0] == "raw":
                        out = out[mode[1]]
                    else:
                        raise ValueError(
                            f"If a tuple is specified, the first element must be 'raw' - got {mode[0]} instead"
                        )
                elif mode == "prediction":
                    out = self.to_prediction(out, **mode_kwargs)
                    # mask non-predictions
                    if isinstance(out, (list, tuple)):
                        out = [
                            o.masked_fill(nan_mask, torch.tensor(float("nan"))) if o.dtype == torch.float else o
                            for o in out
                        ]
                    elif out.dtype == torch.float:  # only floats can be filled with nans
                        out = out.masked_fill(nan_mask, torch.tensor(float("nan")))
                elif mode == "quantiles":
                    out = self.to_quantiles(out, **mode_kwargs)
                    # mask non-predictions
                    if isinstance(out, (list, tuple)):
                        out = [
                            o.masked_fill(nan_mask.unsqueeze(-1), torch.tensor(float("nan")))
                            if o.dtype == torch.float
                            else o
                            for o in out
                        ]
                    elif out.dtype == torch.float:
                        out = out.masked_fill(nan_mask.unsqueeze(-1), torch.tensor(float("nan")))
                elif mode == "raw":
                    pass
                else:
                    raise ValueError(f"Unknown mode {mode} - see docs for valid arguments")

                out = move_to_device(out, device="cpu")

                output.append(out)
                if return_x:
                    x = move_to_device(x, "cpu")
                    x_list.append(x)
                if return_index:
                    index.append(dataloader.dataset.x_to_index(x))
                if return_w:
                    w = move_to_device(w, "cpu")
                    w_list.append(w)
                if return_actual:
                    actual.append(move_to_device(y[0], "cpu"))
                if return_param:
                    pred_param = move_to_device(pred_param, "cpu")
                    param_list.append(pred_param)
                if return_scale:
                    target_scale.append(move_to_device(x['target_scale'], "cpu"))
                progress_bar.update()
                if fast_dev_run:
                    break

        # concatenate output (of different batches)
        if isinstance(mode, (tuple, list)) or mode != "raw":
            if isinstance(output[0], (tuple, list)) and len(output[0]) > 0 and isinstance(output[0][0], torch.Tensor):
                output = [_torch_cat_na([out[idx] for out in output]) for idx in range(len(output[0]))]
            else:
                output = _torch_cat_na(output)
        elif mode == "raw":
            output = _concatenate_output(output)

        # generate output
        if return_x or return_index or return_decoder_lengths or return_w or return_param or return_actual:
            output = [output]
        if return_x:
            output.append(_concatenate_output(x_list))
        if return_index:
            output.append(pd.concat(index, axis=0, ignore_index=True))
        if return_decoder_lengths:
            output.append(torch.cat(decode_lenghts, dim=0))
        if return_actual:
            output.append(torch.cat(actual))
        if return_w:
            output.append(_concatenate_output(w_list))
        if return_param:
            output.append(_concatenate_output(param_list))
        if return_scale:
            output.append(torch.cat(target_scale))
        return output

    def plot_prediction(
        self,
        x: Dict[str, torch.Tensor],
        out: Dict[str, torch.Tensor],
        idx: int = 0,
        add_loss_to_title: Union[Metric, torch.Tensor, bool] = False,
        show_future_observed: bool = True,
        ax=None,
        quantiles_kwargs: Dict[str, Any] = {},
        prediction_kwargs: Dict[str, Any] = {},
    ) -> plt.Figure:
        """
        Plot prediction of prediction vs actuals

        Args:
            x: network input
            out: network output
            idx: index of prediction to plot
            add_loss_to_title: if to add loss to title or loss function to calculate. Can be either metrics,
                bool indicating if to use loss metric or tensor which contains losses for all samples.
                Calcualted losses are determined without weights. Default to False.
            show_future_observed: if to show actuals for future. Defaults to True.
            ax: matplotlib axes to plot on
            quantiles_kwargs (Dict[str, Any]): parameters for ``to_quantiles()`` of the loss metric.
            prediction_kwargs (Dict[str, Any]): parameters for ``to_prediction()`` of the loss metric.

        Returns:
            matplotlib figure
        """
        if isinstance(self.loss, DistributionLoss):
            prediction_kwargs.setdefault("use_metric", False)
            quantiles_kwargs.setdefault("use_metric", False)
        
        # all true values for y of the first sample in batch
        encoder_targets = to_list(x["encoder_target"])
        decoder_targets = to_list(x["decoder_target"])

        y_raws = to_list(out["prediction"])  # raw predictions - used for calculating loss
        y_hats = to_list(self.to_prediction(out, **prediction_kwargs))
        y_quantiles = to_list(self.to_quantiles(out, **quantiles_kwargs))

        # for each target, plot
        figs = []
        for y_raw, y_hat, y_quantile, encoder_target, decoder_target in zip(
            y_raws, y_hats, y_quantiles, encoder_targets, decoder_targets
        ):

            y_all = torch.cat([encoder_target[idx], decoder_target[idx]])
            max_encoder_length = x["encoder_lengths"].max()
            y = torch.cat(
                (
                    y_all[: x["encoder_lengths"][idx]],
                    y_all[max_encoder_length : (max_encoder_length + x["decoder_lengths"][idx])],
                ),
            )
            # move predictions to cpu
            y_hat = y_hat.detach().cpu()[idx, : x["decoder_lengths"][idx]]
            y_quantile = y_quantile.detach().cpu()[idx, : x["decoder_lengths"][idx]]
            y_raw = y_raw.detach().cpu()[idx, : x["decoder_lengths"][idx]]

            # move to cpu
            y = y.detach().cpu()
            # create figure
            if ax is None:
                fig, ax = plt.subplots()
            else:
                fig = ax.get_figure()
            n_pred = y_hat.shape[0]
            x_obs = np.arange(-(y.shape[0] - n_pred), 0)
            x_pred = np.arange(n_pred)
            prop_cycle = iter(plt.rcParams["axes.prop_cycle"])
            obs_color = next(prop_cycle)["color"]
            pred_color = next(prop_cycle)["color"]
            # plot observed history
            if len(x_obs) > 0:
                if len(x_obs) > 1:
                    plotter = ax.plot
                else:
                    plotter = ax.scatter
                plotter(x_obs, y[:-n_pred], label="observed", c=obs_color)
            if len(x_pred) > 1:
                plotter = ax.plot
            else:
                plotter = ax.scatter

            # plot observed prediction
            if show_future_observed:
                plotter(x_pred, y[-n_pred:], label=None, c=obs_color)

            # plot prediction
            plotter(x_pred, y_hat, label="predicted", c=pred_color)

            # plot predicted quantiles
            plotter(x_pred, y_quantile[:, y_quantile.shape[1] // 2], c=pred_color, alpha=0.15)
            for i in range(y_quantile.shape[1] // 2):
                if len(x_pred) > 1:
                    ax.fill_between(x_pred, y_quantile[:, i], y_quantile[:, -i - 1], alpha=0.15, fc=pred_color)
                else:
                    quantiles = torch.tensor([[y_quantile[0, i]], [y_quantile[0, -i - 1]]])
                    ax.errorbar(
                        x_pred,
                        y[[-n_pred]],
                        yerr=quantiles - y[-n_pred],
                        c=pred_color,
                        capsize=1.0,
                    )

            if add_loss_to_title is not False:
                if isinstance(add_loss_to_title, bool):
                    loss = self.loss
                elif isinstance(add_loss_to_title, torch.Tensor):
                    loss = add_loss_to_title.detach()[idx].item()
                elif isinstance(add_loss_to_title, Metric):
                    loss = add_loss_to_title
                else:
                    raise ValueError(f"add_loss_to_title '{add_loss_to_title}'' is unkown")
                if isinstance(loss, MASE):
                    loss_value = loss(y_raw[None], (y[-n_pred:][None], None), y[:n_pred][None])
                elif isinstance(loss, Metric):
                    try:
                        loss_value = loss(y_raw[None], (y[-n_pred:][None], None))
                    except Exception:
                        loss_value = "-"
                else:
                    loss_value = loss
                ax.set_title(f"Loss {loss_value}")
            ax.set_xlabel("Time index")
            figs.append(fig)

        # return multiple of target is a list, otherwise return single figure
        if isinstance(x["encoder_target"], (tuple, list)):
            return figs
        else:
            return fig


class BatchDeepAREstimator(BatchedEstimator, DeepAR):
    """
    1-step DeepAR Model, with conditional sampling
    """
    def decode_all(
        self,
        x: torch.Tensor,
        hidden_state: HiddenState,
        lengths: torch.Tensor = None,
    ):
        decoder_output, hidden_state = self.rnn(x, hidden_state, lengths=lengths, enforce_sorted=False)
        if isinstance(self.hparams.target, str):  # single target
            output = self.distribution_projector(decoder_output)
        else:
            output = [projector(decoder_output) for projector in self.distribution_projector]
        return output, decoder_output, hidden_state

    def decode(
        self,
        input_vector: torch.Tensor,
        target_scale: torch.Tensor,
        decoder_lengths: torch.Tensor,
        hidden_state: HiddenState,
        n_samples: int = None,
    ) -> Tuple[torch.Tensor, bool]:
        """
        Decode hidden state of RNN into prediction. If n_smaples is given,
        decode not by using actual values but rather by
        sampling new targets from past predictions iteratively
        """
        if n_samples is None:
            """
            output: the output features h_t from the last layer of the LSTM, for each t: (N, L, H_out)
            h_n: the final hidden state for each element in the sequence: (num_layers, N, H_out)
            c_n: the final cell state for each element in the sequence: (num_layers, N, H_cell)
            """
            output, decoder_output, _ = self.decode_all(input_vector, hidden_state, lengths=decoder_lengths)
            output = self.transform_output(output, target_scale=target_scale)
        else:
            # run in eval, i.e. simulation mode
            target_pos = self.target_positions
            lagged_target_positions = self.lagged_target_positions
            # repeat for n_samples
            input_vector = input_vector.repeat_interleave(n_samples, 0)
            hidden_state = self.rnn.repeat_interleave(hidden_state, n_samples)
            target_scale = apply_to_list(target_scale, lambda x: x.repeat_interleave(n_samples, 0))

            # define function to run at every decoding step
            def decode_one(
                idx,
                lagged_targets,
                hidden_state,
            ):
                x = input_vector[:, [idx]]
                x[:, 0, target_pos] = lagged_targets[-1]
                for lag, lag_positions in lagged_target_positions.items():
                    if idx > lag:
                        x[:, 0, lag_positions] = lagged_targets[-lag]
                prediction, _, hidden_state = self.decode_all(x, hidden_state)
                prediction = apply_to_list(prediction, lambda x: x[:, 0])  # select first time step
                return prediction, hidden_state

            # make predictions which are fed into next step
            output = self.decode_autoregressive(
                decode_one,
                first_target=input_vector[:, 0, target_pos],
                first_hidden_state=hidden_state,
                target_scale=target_scale,
                n_decoder_steps=input_vector.size(1),
                n_samples=n_samples,
            )
            # reshape predictions for n_samples:
            # from n_samples * batch_size x time steps to batch_size x time steps x n_samples
            output = apply_to_list(output, lambda x: x.reshape(-1, n_samples, input_vector.size(1)).permute(0, 2, 1))
        return output, decoder_output

    def forward(self, x: Dict[str, torch.Tensor], n_samples: int = None) -> Dict[str, torch.Tensor]:
        """
        Forward network
        """
        outputs = []
        decoder_outputs = []
        for i in range(x["decoder_lengths"][0]):
            x["encoder_cat"] = torch.cat([x["encoder_cat"][:,i:], x["decoder_cat"][:,:i]], dim=1)
            x["encoder_cont"] = torch.cat([x["encoder_cont"][:,i:], x["decoder_cont"][:,:i]], dim=1)
            hidden_state = self.encode(x) # (num_layer, batch_size, H)
            # decode
            input_vector = self.construct_input_vector(
                x["decoder_cat"][:,i:i+1],
                x["decoder_cont"][:,i:i+1],
                one_off_target=x["encoder_cont"][
                    torch.arange(x["encoder_cont"].size(0), device=x["encoder_cont"].device),
                    x["encoder_lengths"] - 1,
                    self.target_positions.unsqueeze(-1),
                ].T.contiguous(),
            )  # (batch_size (N in DeepAR), Q, H)

            if self.training:
                assert n_samples is None, "cannot sample from decoder when training"
            output, decoder_output = self.decode(
                input_vector,
                decoder_lengths=torch.ones_like(x["decoder_lengths"], dtype=torch.int),
                target_scale=x["target_scale"],
                hidden_state=hidden_state,
                n_samples=n_samples,
            ) # (batch_size (N in DeepAR), Q, dist_proj (loc_scaler, scale_scaler, loc, scale)
            # return relevant part
            outputs.append(output)
            decoder_outputs.append(decoder_output)
        output = torch.cat(outputs, dim=1)
        decoder_output = torch.cat(decoder_outputs, dim=1)

        if not self.loss.static:
            # self.loss.epoch = self.current_epoch
            mixture_weights = self.get_dynamic_weights(decoder_output)
            output = torch.cat([output, mixture_weights], dim=-1)

        return self.to_network_output(prediction=output)


class BatchDeepARPredictor(BatchedPredictor, BatchDeepAREstimator):
    """
    1-step DeepAR Model, with conditional sampling
    """
    def decode_autoregressive(
        self,
        decode_one: Callable,
        first_target: Union[List[torch.Tensor], torch.Tensor],
        first_hidden_state: Any,
        target_scale: Union[List[torch.Tensor], torch.Tensor],
        n_decoder_steps: int,
        n_samples: int = 1,
        pre_normed_outputs: torch.Tensor = None,
        pre_normed_prediction_params: torch.Tensor = None,
        **kwargs,
    ) -> Union[List[torch.Tensor], torch.Tensor]:
        # make predictions which are fed into next step
        output = []
        weights = []
        pred_params = []
        current_target = first_target
        current_hidden_state = first_hidden_state

        normalized_output = [first_target]

        # pre_normed_prediction_params = []
        for idx in range(n_decoder_steps):
            # get lagged targets
            normed_prediction_params, current_decoder_output, current_hidden_state = decode_one(
                idx, lagged_targets=normalized_output, hidden_state=current_hidden_state, **kwargs
            )  # current target: (N*n_sample, (mu, sigma))

            prediction, current_target, mixture_weights, prediction_parameters = self.output_to_prediction(
                    normed_prediction_params, target_scale=target_scale, n_samples=n_samples, pre_normed_prediction_params=pre_normed_prediction_params, x_1=pre_normed_outputs,
                    current_decoder_output=current_decoder_output
                )

            # save normalized output for lagged targets
            normalized_output.append(current_target)

            pre_normed_outputs = torch.cat([pre_normed_outputs[:,1:], current_target.unsqueeze(1)], dim=1)
            pre_normed_prediction_params = torch.cat([pre_normed_prediction_params[:,1:], normed_prediction_params.unsqueeze(1)], dim=1)

            output.append(prediction)
            weights.append(mixture_weights)
            pred_params.append(prediction_parameters)

        if isinstance(self.hparams.target, str):
            output = torch.stack(output, dim=1)
            weights = torch.stack(weights, dim=1)
            pred_params = torch.stack(pred_params, dim=1)
        else:
            # for multi-targets
            output = [torch.stack([out[idx] for out in output], dim=1) for idx in range(len(self.target_positions))]
        return output, weights, pred_params

    def encode(self, x: Dict[str, torch.Tensor]) -> HiddenState:
        """
        Encode sequence into hidden state
        """
        # encode using rnn
        assert x["encoder_lengths"].min() > 0
        encoder_lengths = x["encoder_lengths"] - 1
        input_vector = self.construct_input_vector(x["encoder_cat"], x["encoder_cont"])
        encoder_output, hidden_state = self.rnn(
            input_vector, lengths=encoder_lengths, enforce_sorted=False
        )  # second ouput is not needed (hidden state)
        if isinstance(self.hparams.target, str):  # single target
            output = self.distribution_projector(encoder_output)
        else:
            output = [projector(encoder_output) for projector in self.distribution_projector]
        return output, hidden_state

    def decode(
        self,
        input_vector: torch.Tensor,
        target_scale: torch.Tensor,
        decoder_lengths: torch.Tensor,
        hidden_state: HiddenState,
        n_samples: int = None,
        encoder_output: torch.Tensor = None,
        encoder_dist_params: torch.Tensor = None
    ) -> Tuple[torch.Tensor, bool]:
        """
        Decode hidden state of RNN into prediction. If n_smaples is given,
        decode not by using actual values but rather by
        sampling new targets from past predictions iteratively
        """
        if n_samples is None:  # trainig
            output, decoder_output, _ = self.decode_all(input_vector, hidden_state, lengths=decoder_lengths)  # use real values as decoder input
            output = self.transform_output(output, target_scale=target_scale)
            weights = self.get_dynamic_weights(decoder_output)
            return output, weights, None
        else:  # validating and testing
            # run in eval, i.e. simulation mode
            target_pos = self.target_positions
            lagged_target_positions = self.lagged_target_positions
            # repeat for n_samples
            input_vector = input_vector.repeat_interleave(n_samples, 0)  # (batch_size*n_sample, prediction_length, n_features)
            hidden_state = self.rnn.repeat_interleave(hidden_state, n_samples)
            target_scale = apply_to_list(target_scale, lambda x: x.repeat_interleave(n_samples, 0))

            encoder_output = encoder_output.repeat_interleave(n_samples, 0)
            encoder_dist_params = encoder_dist_params.repeat_interleave(n_samples, 0)

            # define function to run at every decoding step
            def decode_one(
                idx,
                lagged_targets,
                hidden_state,
            ):
                x = input_vector[:, [idx]]
                x[:, 0, target_pos] = lagged_targets[-1]
                for lag, lag_positions in lagged_target_positions.items():
                    if idx > lag:
                        x[:, 0, lag_positions] = lagged_targets[-lag]
                prediction, decoder_output, hidden_state = self.decode_all(x, hidden_state)  # prediction: (batch_size*n_sample, 1, dist_parameters), hidden_state: (n_layers, batch_size*n_sample, hidden_size)
                prediction = apply_to_list(prediction, lambda x: x[:, 0])  # select first time step
                return prediction, decoder_output, hidden_state

            # make predictions which are fed into next step
            output, weights, pred_params = self.decode_autoregressive(
                decode_one,
                first_target=input_vector[:, 0, target_pos],
                first_hidden_state=hidden_state,
                target_scale=target_scale,
                n_decoder_steps=input_vector.size(1),
                n_samples=n_samples,
                pre_normed_outputs=encoder_output,
                pre_normed_prediction_params=encoder_dist_params,
            )
            # reshape predictions for n_samples:
            # from n_samples * batch_size x time steps to batch_size x time steps x n_samples
            output = apply_to_list(output, lambda x: x.reshape(-1, n_samples, input_vector.size(1)).permute(0, 2, 1))
            weights = apply_to_list(weights, lambda x: x.reshape(-1, n_samples, input_vector.size(1), weights.shape[-1]).permute(0, 2, 3, 1))
            return output, weights, pred_params

    def forward(self, x: Dict[str, torch.Tensor], n_samples: int = None) -> Dict[str, torch.Tensor]:
        """
        Forward network
        """
        encoder_dist_params, hidden_state = self.encode(x) # (num_layer, batch_size, H)
        # decode
        input_vector = self.construct_input_vector(
            x["decoder_cat"],
            x["decoder_cont"],
            one_off_target=x["encoder_cont"][
                torch.arange(x["encoder_cont"].size(0), device=x["encoder_cont"].device),
                x["encoder_lengths"] - 1,
                self.target_positions.unsqueeze(-1),
            ].T.contiguous(),
        )  # (batch_size (N in DeepAR), Q, H)

        if self.training:
            assert n_samples is None, "cannot sample from decoder when training"
        output, weights, pred_params = self.decode(
            input_vector,
            decoder_lengths=x["decoder_lengths"],
            target_scale=x["target_scale"],
            hidden_state=hidden_state,
            n_samples=n_samples,
            encoder_output=x["encoder_cont"][:,-self.loss.batch_cov_horizon+1:, :1],
            encoder_dist_params=encoder_dist_params[:,-self.loss.batch_cov_horizon+1:],
        ) # (batch_size (N in DeepAR), Q, dist_proj (loc_scaler, scale_scaler, loc, scale)

        encoder_dist_params = self.transform_output(
        prediction=encoder_dist_params, target_scale=x["target_scale"]
        )

        # return relevant part
        if pred_params is None:
            return self.to_network_output(prediction=output), self.to_network_output(prediction=weights)
        else:
            return self.to_network_output(prediction=output), self.to_network_output(prediction=weights[...,0]), self.to_network_output(prediction=torch.cat([encoder_dist_params, pred_params[:,:,0]], dim=1))


class BatchGPTEstimator(BatchedEstimator, ARTransformer):
    """
    1-step DeepAR Model, with conditional sampling
    """
    def decode_all(
        self,
        input_vector: torch.Tensor,
        decoder_length: torch.Tensor = None,
    ):
        src = self.add_input_vector(input_vector)
        # src = self.pos_encoder(input_vector.permute(1,0,2)).permute(1,0,2)
        decoder_output = self.rnn(src, mask=nn.Transformer.generate_square_subsequent_mask(src.shape[1]).to(src.device))
        decoder_output = decoder_output[:,-decoder_length:]
        if isinstance(self.hparams.target, str):  # single target
            output = self.distribution_projector(decoder_output)
        else:
            output = [projector(decoder_output) for projector in self.distribution_projector]
        return output, decoder_output

    def decode(
        self,
        input_vector: torch.Tensor,
        decoder_length: int,
        target_scale: torch.Tensor,
        n_samples: int = None,
    ) -> Tuple[torch.Tensor, bool]:
        """
        Decode hidden state of RNN into prediction. If n_smaples is given,
        decode not by using actual values but rather by
        sampling new targets from past predictions iteratively
        """
        if n_samples is None:
            output, decoder_output = self.decode_all(input_vector, decoder_length)
            output = self.transform_output(output, target_scale=target_scale)
        else:
            # run in eval, i.e. simulation mode
            target_pos = self.target_positions
            lagged_target_positions = self.lagged_target_positions
            # repeat for n_samples
            input_vector = input_vector.repeat_interleave(n_samples, 0)
            target_scale = apply_to_list(target_scale, lambda x: x.repeat_interleave(n_samples, 0))

            # define function to run at every decoding step
            def decode_one(
                idx,
                lagged_targets,
                decoder_length
            ):
                x = input_vector[:, :decoder_length+idx]
                lagged_targets = torch.stack(lagged_targets, dim=1)
                x[:, decoder_length-1:, target_pos] = lagged_targets
                for lag, lag_positions in lagged_target_positions.items():
                    if idx > lag:
                        x[:, 0, lag_positions] = lagged_targets[-lag]
                prediction = self.decode_all(x, lagged_targets.shape[1])
                prediction = apply_to_list(prediction, lambda x: x[:, -1])  # select first time step
                return prediction

            # make predictions which are fed into next step
            output = self.decode_autoregressive(
                decode_one,
                first_target=input_vector[:, -decoder_length, target_pos],
                target_scale=target_scale,
                n_decoder_steps=decoder_length,
                n_samples=n_samples,
            )
            # reshape predictions for n_samples:
            # from n_samples * batch_size x time steps to batch_size x time steps x n_samples
            output = apply_to_list(output, lambda x: x.reshape(-1, n_samples, decoder_length).permute(0, 2, 1))
        return output, decoder_output

    def forward(self, x: Dict[str, torch.Tensor], n_samples: int = None) -> Dict[str, torch.Tensor]:
        """
        Forward network
        """
        # # encoder_output = self.encode(x)
        # # encoder_length = encoder_output.shape[1]

        # x_cat = torch.cat([x["encoder_cat"], x["decoder_cat"]], dim=1)
        # x_cont = torch.cat([x["encoder_cont"], x["decoder_cont"]], dim=1)

        # input_vector = self.construct_input_vector(x_cat, x_cont)
        # decoder_length = int(x["decoder_lengths"].max())

        # if self.training:
        #     assert n_samples is None, "cannot sample from decoder when training"

        # output, decoder_output = self.decode(
        #     input_vector,  # (B, P+Q, H)
        #     decoder_length=decoder_length,
        #     target_scale=x["target_scale"],  # (B, 2)
        #     n_samples=n_samples,
        # )

        outputs = []
        decoder_outputs = []
        for i in range(x["decoder_lengths"][0]):
            x_cat = torch.cat([x["encoder_cat"][:,i:], x["decoder_cat"][:,:i+1]], dim=1)
            x_cont = torch.cat([x["encoder_cont"][:,i:], x["decoder_cont"][:,:i+1]], dim=1)

            input_vector = self.construct_input_vector(x_cat, x_cont)

            if self.training:
                assert n_samples is None, "cannot sample from decoder when training"

            output, decoder_output = self.decode(
                input_vector,  # (B, P+Q, H)
                decoder_length=1,
                target_scale=x["target_scale"],  # (B, 2)
                n_samples=n_samples,
            )

            outputs.append(output)
            decoder_outputs.append(decoder_output)

        output = torch.cat(outputs, dim=1)
        decoder_output = torch.cat(decoder_outputs, dim=1)

        if not self.loss.static:
            mixture_weights = self.get_dynamic_weights(decoder_output)
            output = torch.cat([output, mixture_weights], dim=-1)

        return self.to_network_output(prediction=output)


class BatchGPTPredictor(BatchedPredictor, BatchGPTEstimator):
    """
    1-step DeepAR Model, with conditional sampling
    """
    def decode_autoregressive(
        self,
        decode_one: Callable,
        first_target: Union[List[torch.Tensor], torch.Tensor],
        target_scale: Union[List[torch.Tensor], torch.Tensor],
        n_decoder_steps: int,
        n_samples: int = 1,
        pre_normed_outputs: torch.Tensor = None,
        **kwargs,
    ) -> Union[List[torch.Tensor], torch.Tensor]:

        # make predictions which are fed into next step
        output = []
        weights = []
        pred_params = []
        normalized_output = [first_target]
        for idx in range(n_decoder_steps):
            # get lagged targets
            normed_prediction_params, current_decoder_output = decode_one(
                idx, lagged_targets=normalized_output, decoder_length=n_decoder_steps, **kwargs
            )

            # get prediction and its normalized version for the next step
            prediction, current_target, mixture_weights, prediction_parameters = self.output_to_prediction(
                normalized_prediction_parameters=normed_prediction_params[:, -1],
                target_scale=target_scale,
                n_samples=n_samples, pre_normed_prediction_params=normed_prediction_params[:, :-1][:,-self.loss.batch_cov_horizon+1:],
                x_1=pre_normed_outputs,
                current_decoder_output=current_decoder_output
            )
            # save normalized output for lagged targets
            normalized_output.append(current_target)
            # set output to unnormalized samples, append each target as n_batch_samples x n_random_samples

            pre_normed_outputs = torch.cat([pre_normed_outputs[:,1:], current_target.unsqueeze(1)], dim=1)

            output.append(prediction)
            weights.append(mixture_weights)
            pred_params.append(prediction_parameters)

        if isinstance(self.hparams.target, str):
            output = torch.stack(output, dim=1)
            weights = torch.stack(weights, dim=1)
            pred_params = torch.stack(pred_params, dim=1)
        else:
            # for multi-targets
            output = [torch.stack([out[idx] for out in output], dim=1) for idx in range(len(self.target_positions))]
        return output, weights, pred_params

    def decode_all(
        self,
        input_vector: torch.Tensor,
        decoder_length: int,
    ):
        src = self.add_input_vector(input_vector)
        # src = self.pos_encoder(input_vector.permute(1,0,2)).permute(1,0,2)
        decoder_output = self.rnn(src, mask=nn.Transformer.generate_square_subsequent_mask(src.shape[1]).to(src.device))
        if isinstance(self.hparams.target, str):  # single target
            output = self.distribution_projector(decoder_output)
        else:
            output = [projector(decoder_output) for projector in self.distribution_projector]
        return output, decoder_output[:,-decoder_length:]

    def decode(
        self,
        input_vector: torch.Tensor,
        decoder_length: int,
        target_scale: torch.Tensor,
        n_samples: int = None,
        encoder_output: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, bool]:
        """
        Decode hidden state of RNN into prediction. If n_smaples is given,
        decode not by using actual values but rather by
        sampling new targets from past predictions iteratively
        """
        if n_samples is None:
            output, decoder_output = self.decode_all(input_vector, decoder_length)
            output = self.transform_output(output, target_scale=target_scale)
            weights = self.get_dynamic_weights(decoder_output)
            return output, weights, None
        else:
            # run in eval, i.e. simulation mode
            target_pos = self.target_positions
            lagged_target_positions = self.lagged_target_positions
            # repeat for n_samples
            input_vector = input_vector.repeat_interleave(n_samples, 0)
            target_scale = apply_to_list(target_scale, lambda x: x.repeat_interleave(n_samples, 0))

            encoder_output = encoder_output.repeat_interleave(n_samples, 0)

            # define function to run at every decoding step
            def decode_one(
                idx,
                lagged_targets,
                decoder_length
            ):
                x = input_vector[:, :decoder_length+idx]
                # lagged_targets = torch.stack(lagged_targets, dim=1)
                x[:, decoder_length-1:, target_pos] = torch.stack(lagged_targets, dim=1)
                for lag, lag_positions in lagged_target_positions.items():
                    if idx > lag:
                        x[:, 0, lag_positions] = lagged_targets[-lag]
                # prediction, decoder_output = self.decode_all(x, lagged_targets.shape[1])
                prediction, decoder_output = self.decode_all(x, len(lagged_targets))
                return prediction, decoder_output[:, -1:]

            # make predictions which are fed into next step
            output, weights, pred_params = self.decode_autoregressive(
                decode_one,
                first_target=input_vector[:, -decoder_length, target_pos],
                target_scale=target_scale,
                n_decoder_steps=decoder_length,
                n_samples=n_samples,
                pre_normed_outputs=encoder_output,
            )
            output = apply_to_list(output, lambda x: x.reshape(-1, n_samples, decoder_length).permute(0, 2, 1))
            weights = apply_to_list(weights, lambda x: x.reshape(-1, n_samples, decoder_length, weights.shape[-1]).permute(0, 2, 3, 1))
            return output, weights, pred_params

    def forward(self, x: Dict[str, torch.Tensor], n_samples: int = None) -> Dict[str, torch.Tensor]:
        """
        Forward network
        """
        x_cat = torch.cat([x["encoder_cat"], x["decoder_cat"]], dim=1)
        x_cont = torch.cat([x["encoder_cont"], x["decoder_cont"]], dim=1)

        input_vector = self.construct_input_vector(x_cat, x_cont)
        decoder_length = int(x["decoder_lengths"].max())

        src = self.add_input_vector(input_vector[:,:-decoder_length])
        # src = self.pos_encoder(input_vector.permute(1,0,2)).permute(1,0,2)
        encoder_output = self.rnn(src, mask=nn.Transformer.generate_square_subsequent_mask(src.shape[1]).to(src.device))
        if isinstance(self.hparams.target, str):  # single target
            encoder_dist_params = self.distribution_projector(encoder_output)
        else:
            encoder_dist_params = [projector(encoder_output) for projector in self.distribution_projector]

        if self.training:
            assert n_samples is None, "cannot sample from decoder when training"

        output, weights, pred_params = self.decode(
            input_vector,
            decoder_length=decoder_length,
            target_scale=x["target_scale"],
            n_samples=n_samples,
            encoder_output=x["encoder_cont"][:,-self.loss.batch_cov_horizon+1:, :1],
        )

        encoder_dist_params = self.transform_output(
        prediction=encoder_dist_params, target_scale=x["target_scale"]
        )

        if pred_params is None:
            return self.to_network_output(prediction=output), self.to_network_output(prediction=weights)
        else:
            return self.to_network_output(prediction=output), self.to_network_output(prediction=weights[...,0]), self.to_network_output(prediction=torch.cat([encoder_dist_params, pred_params[:,:,0]], dim=1))

    # def predict(
    #     self,
    #     data: Union[DataLoader, pd.DataFrame, TimeSeriesDataSet],
    #     mode: Union[str, Tuple[str, str]] = "prediction",
    #     return_index: bool = False,
    #     return_decoder_lengths: bool = False,
    #     batch_size: int = 64,
    #     num_workers: int = 0,
    #     fast_dev_run: bool = False,
    #     show_progress_bar: bool = False,
    #     return_x: bool = False,
    #     return_w: bool = False,
    #     return_actual: bool = False,
    #     return_param: bool = False,
    #     mode_kwargs: Dict[str, Any] = None,
    #     **kwargs,
    # ):
    #     """
    #     Run inference / prediction.

    #     Args:
    #         dataloader: dataloader, dataframe or dataset
    #         mode: one of "prediction", "quantiles", or "raw", or tuple ``("raw", output_name)`` where output_name is
    #             a name in the dictionary returned by ``forward()``
    #         return_index: if to return the prediction index (in the same order as the output, i.e. the row of the
    #             dataframe corresponds to the first dimension of the output and the given time index is the time index
    #             of the first prediction)
    #         return_decoder_lengths: if to return decoder_lengths (in the same order as the output
    #         batch_size: batch size for dataloader - only used if data is not a dataloader is passed
    #         num_workers: number of workers for dataloader - only used if data is not a dataloader is passed
    #         fast_dev_run: if to only return results of first batch
    #         show_progress_bar: if to show progress bar. Defaults to False.
    #         return_x: if to return network inputs (in the same order as prediction output)
    #         mode_kwargs (Dict[str, Any]): keyword arguments for ``to_prediction()`` or ``to_quantiles()``
    #             for modes "prediction" and "quantiles"
    #         **kwargs: additional arguments to network's forward method

    #     Returns:
    #         output, x, index, decoder_lengths: some elements might not be present depending on what is configured
    #             to be returned
    #     """
    #     # convert to dataloader
    #     if isinstance(data, pd.DataFrame):
    #         data = TimeSeriesDataSet.from_parameters(self.dataset_parameters, data, predict=True)
    #     if isinstance(data, TimeSeriesDataSet):
    #         dataloader = data.to_dataloader(batch_size=batch_size, train=False, num_workers=num_workers)
    #     else:
    #         dataloader = data

    #     # mode kwargs default to None
    #     if mode_kwargs is None:
    #         mode_kwargs = {}

    #     # ensure passed dataloader is correct
    #     assert isinstance(dataloader.dataset, TimeSeriesDataSet), "dataset behind dataloader mut be TimeSeriesDataSet"

    #     # prepare model
    #     self.eval()  # no dropout, etc. no gradients

    #     # run predictions
    #     output = []
    #     decode_lenghts = []
    #     x_list = []
    #     index = []
    #     w_list = []
    #     param_list = []
    #     actual = []
    #     progress_bar = tqdm(desc="Predict", unit=" batches", total=len(dataloader), disable=not show_progress_bar)
    #     with torch.no_grad():
    #         for x, y in dataloader:
    #             # move data to appropriate device
    #             data_device = x["encoder_cont"].device
    #             if data_device != self.device:
    #                 x = move_to_device(x, self.device)
    #                 y = move_to_device(y, self.device)

    #             # make prediction
    #             out, w, pred_param = self(x, **kwargs)  # raw output is dictionary

    #             lengths = x["decoder_lengths"]
    #             if return_decoder_lengths:
    #                 decode_lenghts.append(lengths)
    #             nan_mask = create_mask(lengths.max(), lengths)
    #             if isinstance(mode, (tuple, list)):
    #                 if mode[0] == "raw":
    #                     out = out[mode[1]]
    #                 else:
    #                     raise ValueError(
    #                         f"If a tuple is specified, the first element must be 'raw' - got {mode[0]} instead"
    #                     )
    #             elif mode == "prediction":
    #                 out = self.to_prediction(out, **mode_kwargs)
    #                 # mask non-predictions
    #                 if isinstance(out, (list, tuple)):
    #                     out = [
    #                         o.masked_fill(nan_mask, torch.tensor(float("nan"))) if o.dtype == torch.float else o
    #                         for o in out
    #                     ]
    #                 elif out.dtype == torch.float:  # only floats can be filled with nans
    #                     out = out.masked_fill(nan_mask, torch.tensor(float("nan")))
    #             elif mode == "quantiles":
    #                 out = self.to_quantiles(out, **mode_kwargs)
    #                 # mask non-predictions
    #                 if isinstance(out, (list, tuple)):
    #                     out = [
    #                         o.masked_fill(nan_mask.unsqueeze(-1), torch.tensor(float("nan")))
    #                         if o.dtype == torch.float
    #                         else o
    #                         for o in out
    #                     ]
    #                 elif out.dtype == torch.float:
    #                     out = out.masked_fill(nan_mask.unsqueeze(-1), torch.tensor(float("nan")))
    #             elif mode == "raw":
    #                 pass
    #             else:
    #                 raise ValueError(f"Unknown mode {mode} - see docs for valid arguments")

    #             out = move_to_device(out, device="cpu")

    #             output.append(out)
    #             actual.append(y[0])
    #             if return_x:
    #                 x = move_to_device(x, "cpu")
    #                 x_list.append(x)
    #             if return_index:
    #                 index.append(dataloader.dataset.x_to_index(x))
    #             if return_w:
    #                 w_list.append(w)
    #             if return_param:
    #                 param_list.append(pred_param)
    #             progress_bar.update()
    #             if fast_dev_run:
    #                 break

    #     # concatenate output (of different batches)
    #     if isinstance(mode, (tuple, list)) or mode != "raw":
    #         if isinstance(output[0], (tuple, list)) and len(output[0]) > 0 and isinstance(output[0][0], torch.Tensor):
    #             output = [_torch_cat_na([out[idx] for out in output]) for idx in range(len(output[0]))]
    #         else:
    #             output = _torch_cat_na(output)
    #     elif mode == "raw":
    #         output = _concatenate_output(output)

    #     actual = torch.cat(actual)

    #     # generate output
    #     if return_x or return_index or return_decoder_lengths or return_w or return_param or return_actual:
    #         output = [output]
    #     if return_x:
    #         output.append(_concatenate_output(x_list))
    #     if return_index:
    #         output.append(pd.concat(index, axis=0, ignore_index=True))
    #     if return_decoder_lengths:
    #         output.append(torch.cat(decode_lenghts, dim=0))
    #     if return_actual:
    #         output.append(actual)
    #     if return_w:
    #         output.append(_concatenate_output(w_list))
    #     if return_param:
    #         output.append(_concatenate_output(param_list))
    #     return output

    # def plot_prediction(
    #     self,
    #     x: Dict[str, torch.Tensor],
    #     out: Dict[str, torch.Tensor],
    #     idx: int = 0,
    #     add_loss_to_title: Union[Metric, torch.Tensor, bool] = False,
    #     show_future_observed: bool = True,
    #     ax=None,
    #     quantiles_kwargs: Dict[str, Any] = {},
    #     prediction_kwargs: Dict[str, Any] = {},
    # ) -> plt.Figure:
    #     """
    #     Plot prediction of prediction vs actuals

    #     Args:
    #         x: network input
    #         out: network output
    #         idx: index of prediction to plot
    #         add_loss_to_title: if to add loss to title or loss function to calculate. Can be either metrics,
    #             bool indicating if to use loss metric or tensor which contains losses for all samples.
    #             Calcualted losses are determined without weights. Default to False.
    #         show_future_observed: if to show actuals for future. Defaults to True.
    #         ax: matplotlib axes to plot on
    #         quantiles_kwargs (Dict[str, Any]): parameters for ``to_quantiles()`` of the loss metric.
    #         prediction_kwargs (Dict[str, Any]): parameters for ``to_prediction()`` of the loss metric.

    #     Returns:
    #         matplotlib figure
    #     """
    #     if isinstance(self.loss, DistributionLoss):
    #         prediction_kwargs.setdefault("use_metric", False)
    #         quantiles_kwargs.setdefault("use_metric", False)
        
    #     # all true values for y of the first sample in batch
    #     encoder_targets = to_list(x["encoder_target"])
    #     decoder_targets = to_list(x["decoder_target"])

    #     y_raws = to_list(out["prediction"])  # raw predictions - used for calculating loss
    #     y_hats = to_list(self.to_prediction(out, **prediction_kwargs))
    #     y_quantiles = to_list(self.to_quantiles(out, **quantiles_kwargs))

    #     # for each target, plot
    #     figs = []
    #     for y_raw, y_hat, y_quantile, encoder_target, decoder_target in zip(
    #         y_raws, y_hats, y_quantiles, encoder_targets, decoder_targets
    #     ):

    #         y_all = torch.cat([encoder_target[idx], decoder_target[idx]])
    #         max_encoder_length = x["encoder_lengths"].max()
    #         y = torch.cat(
    #             (
    #                 y_all[: x["encoder_lengths"][idx]],
    #                 y_all[max_encoder_length : (max_encoder_length + x["decoder_lengths"][idx])],
    #             ),
    #         )
    #         # move predictions to cpu
    #         y_hat = y_hat.detach().cpu()[idx, : x["decoder_lengths"][idx]]
    #         y_quantile = y_quantile.detach().cpu()[idx, : x["decoder_lengths"][idx]]
    #         y_raw = y_raw.detach().cpu()[idx, : x["decoder_lengths"][idx]]

    #         # move to cpu
    #         y = y.detach().cpu()
    #         # create figure
    #         if ax is None:
    #             fig, ax = plt.subplots()
    #         else:
    #             fig = ax.get_figure()
    #         n_pred = y_hat.shape[0]
    #         x_obs = np.arange(-(y.shape[0] - n_pred), 0)
    #         x_pred = np.arange(n_pred)
    #         prop_cycle = iter(plt.rcParams["axes.prop_cycle"])
    #         obs_color = next(prop_cycle)["color"]
    #         pred_color = next(prop_cycle)["color"]
    #         # plot observed history
    #         if len(x_obs) > 0:
    #             if len(x_obs) > 1:
    #                 plotter = ax.plot
    #             else:
    #                 plotter = ax.scatter
    #             plotter(x_obs, y[:-n_pred], label="observed", c=obs_color)
    #         if len(x_pred) > 1:
    #             plotter = ax.plot
    #         else:
    #             plotter = ax.scatter

    #         # plot observed prediction
    #         if show_future_observed:
    #             plotter(x_pred, y[-n_pred:], label=None, c=obs_color)

    #         # plot prediction
    #         plotter(x_pred, y_hat, label="predicted", c=pred_color)

    #         # plot predicted quantiles
    #         plotter(x_pred, y_quantile[:, y_quantile.shape[1] // 2], c=pred_color, alpha=0.15)
    #         for i in range(y_quantile.shape[1] // 2):
    #             if len(x_pred) > 1:
    #                 ax.fill_between(x_pred, y_quantile[:, i], y_quantile[:, -i - 1], alpha=0.15, fc=pred_color)
    #             else:
    #                 quantiles = torch.tensor([[y_quantile[0, i]], [y_quantile[0, -i - 1]]])
    #                 ax.errorbar(
    #                     x_pred,
    #                     y[[-n_pred]],
    #                     yerr=quantiles - y[-n_pred],
    #                     c=pred_color,
    #                     capsize=1.0,
    #                 )

    #         if add_loss_to_title is not False:
    #             if isinstance(add_loss_to_title, bool):
    #                 loss = self.loss
    #             elif isinstance(add_loss_to_title, torch.Tensor):
    #                 loss = add_loss_to_title.detach()[idx].item()
    #             elif isinstance(add_loss_to_title, Metric):
    #                 loss = add_loss_to_title
    #             else:
    #                 raise ValueError(f"add_loss_to_title '{add_loss_to_title}'' is unkown")
    #             if isinstance(loss, MASE):
    #                 loss_value = loss(y_raw[None], (y[-n_pred:][None], None), y[:n_pred][None])
    #             elif isinstance(loss, Metric):
    #                 try:
    #                     loss_value = loss(y_raw[None], (y[-n_pred:][None], None))
    #                 except Exception:
    #                     loss_value = "-"
    #             else:
    #                 loss_value = loss
    #             ax.set_title(f"Loss {loss_value}")
    #         ax.set_xlabel("Time index")
    #         figs.append(fig)

    #     # return multiple of target is a list, otherwise return single figure
    #     if isinstance(x["encoder_target"], (tuple, list)):
    #         return figs
    #     else:
    #         return fig

