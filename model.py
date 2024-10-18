from copy import copy, deepcopy
from typing import Any, Callable, Dict, List, Literal, Tuple, Optional, Union

import math
import numpy as np
import pandas as pd

import torch
from torch import nn
from torch.utils.data.dataloader import DataLoader
from torchmetrics import Metric as LightningMetric

from pytorch_forecasting import DeepAR
from pytorch_forecasting.models.base_model import BaseModelWithCovariates, AutoRegressiveBaseModel, AutoRegressiveBaseModelWithCovariates
from pytorch_forecasting.models.nn import HiddenState, MultiEmbedding, LSTM, get_rnn
from pytorch_forecasting.data.encoders import EncoderNormalizer, MultiNormalizer, NaNLabelEncoder
from pytorch_forecasting.data.timeseries import TimeSeriesDataSet
from pytorch_forecasting.utils import apply_to_list, to_list, create_mask, detach, integer_histogram, masked_op, padded_stack
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
    MultiHorizonMetric
)
from pytorch_forecasting.models.temporal_fusion_transformer.sub_modules import (
    AddNorm,
    GateAddNorm,
    GatedLinearUnit,
    GatedResidualNetwork,
    InterpretableMultiHeadAttention,
    VariableSelectionNetwork,
    PositionalEncoder,
)

from matplotlib import pyplot as plt


class PositionalEncoding(nn.Module):
    r"""Inject some information about the relative or absolute position of the tokens in the sequence.
        The positional encodings have the same dimension as the embeddings, so that the two can be summed.
        Here, we use sine and cosine functions of different frequencies.
    .. math:
        \text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
        \text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        \text{where pos is the word position and i is the embed idx)
    Args:
        d_model: the embed dim (required).
        dropout: the dropout value (default=0.1).
        max_len: the max. length of the incoming sequence (default=5000).
    Examples:
        >>> pos_encoder = PositionalEncoding(d_model)
    """

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        Examples:
            >>> output = pos_encoder(x)
        """

        x = x + self.pe[:x.size(0), :]  # (T, B, d_model) + (T, 1, d_model)
        return self.dropout(x)


def make_linear_layer(dim_in, dim_out):
    lin = nn.Linear(dim_in, dim_out)
    torch.nn.init.uniform_(lin.weight, -0.07, 0.07)
    torch.nn.init.zeros_(lin.bias)
    return lin


class DeepAR(DeepAR):
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


class ARTransformer(AutoRegressiveBaseModelWithCovariates):
    def __init__(
        self,
        hidden_size: int = 10,
        n_heads = 1,
        rnn_layers: int = 2,
        dropout: float = 0.1,
        static_categoricals: List[str] = [],
        static_reals: List[str] = [],
        time_varying_categoricals_encoder: List[str] = [],
        time_varying_categoricals_decoder: List[str] = [],
        categorical_groups: Dict[str, List[str]] = {},
        time_varying_reals_encoder: List[str] = [],
        time_varying_reals_decoder: List[str] = [],
        embedding_sizes: Dict[str, Tuple[int, int]] = {},
        embedding_paddings: List[str] = [],
        embedding_labels: Dict[str, np.ndarray] = {},
        x_reals: List[str] = [],
        x_categoricals: List[str] = [],
        n_validation_samples: int = None,
        n_plotting_samples: int = None,
        target: Union[str, List[str]] = None,
        target_lags: Dict[str, List[int]] = {},
        loss: DistributionLoss = None,
        logging_metrics: nn.ModuleList = None,
        **kwargs,
    ):
        """
        without positional encoding
        """
        if loss is None:
            loss = NormalDistributionLoss()
        if logging_metrics is None:
            logging_metrics = nn.ModuleList([SMAPE(), MAE(), RMSE(), MAPE(), MASE()])
        if n_plotting_samples is None:
            if n_validation_samples is None:
                n_plotting_samples = n_validation_samples
            else:
                n_plotting_samples = 100
        self.save_hyperparameters()
        # store loss function separately as it is a module
        super().__init__(loss=loss, logging_metrics=logging_metrics, **kwargs)

        self.embeddings = MultiEmbedding(
            embedding_sizes=embedding_sizes,
            embedding_paddings=embedding_paddings,
            categorical_groups=categorical_groups,
            x_categoricals=x_categoricals,
        )

        lagged_target_names = [l for lags in target_lags.values() for l in lags]
        assert set(self.encoder_variables) - set(to_list(target)) - set(lagged_target_names) == set(
            self.decoder_variables
        ) - set(lagged_target_names), "Encoder and decoder variables have to be the same apart from target variable"
        for targeti in to_list(target):
            assert (
                targeti in time_varying_reals_encoder
            ), f"target {targeti} has to be real"  # todo: remove this restriction
        assert (isinstance(target, str) and isinstance(loss, DistributionLoss)) or (
            isinstance(target, (list, tuple)) and isinstance(loss, MultiLoss) and len(loss) == len(target)
        ), "number of targets should be equivalent to number of loss metrics"

        self.cont_size = len(self.reals)
        self.cat_size = sum(self.embeddings.output_size.values())

        self.target_embed = nn.Linear(self.cont_size, self.hparams.hidden_size)
        self.covariate_embed = nn.Linear(self.cat_size, self.hparams.hidden_size)

        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_size, nhead=n_heads, dim_feedforward=hidden_size*2, dropout=dropout, batch_first=True)
        self.rnn = nn.TransformerEncoder(encoder_layer, num_layers=rnn_layers)

        # add linear layers for argument projects
        if isinstance(target, str):  # single target
            self.distribution_projector = nn.Linear(self.hparams.hidden_size, len(self.loss.distribution_arguments))
        else:  # multi target
            self.distribution_projector = nn.ModuleList(
                [nn.Linear(self.hparams.hidden_size, len(args)) for args in self.loss.distribution_arguments]
            )

    @classmethod
    def from_dataset(
        cls,
        dataset: TimeSeriesDataSet,
        allowed_encoder_known_variable_names: List[str] = None,
        **kwargs,
    ):
        """
        Create model from dataset.

        Args:
            dataset: timeseries dataset
            allowed_encoder_known_variable_names: List of known variables that are allowed in encoder, defaults to all
            **kwargs: additional arguments such as hyperparameters for model (see ``__init__()``)

        Returns:
            DeepAR network
        """
        new_kwargs = {}
        if dataset.multi_target:
            new_kwargs.setdefault("loss", MultiLoss([NormalDistributionLoss()] * len(dataset.target_names)))
        new_kwargs.update(kwargs)
        assert not isinstance(dataset.target_normalizer, NaNLabelEncoder) and (
            not isinstance(dataset.target_normalizer, MultiNormalizer)
            or all([not isinstance(normalizer, NaNLabelEncoder) for normalizer in dataset.target_normalizer])
        ), "target(s) should be continuous - categorical targets are not supported"  # todo: remove this restriction
        if isinstance(new_kwargs.get("loss", None), MultivariateDistributionLoss):
            assert (
                dataset.min_prediction_length == dataset.max_prediction_length
            ), "Multivariate models require constant prediction lenghts"
        return super().from_dataset(
            dataset, allowed_encoder_known_variable_names=allowed_encoder_known_variable_names, **new_kwargs
        )

    def construct_input_vector(
        self, x_cat: torch.Tensor, x_cont: torch.Tensor, one_off_target: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Create input vector into RNN network

        Args:
            one_off_target: tensor to insert into first position of target. If None (default), remove first time step.
        """
        # create input vector

        if len(self.categoricals) > 0:
            embeddings = self.embeddings(x_cat)
            flat_embeddings = torch.cat([emb for emb in embeddings.values()], dim=-1)
            input_vector = flat_embeddings

        if len(self.reals) > 0:
            input_vector = x_cont.clone()

        if len(self.reals) > 0 and len(self.categoricals) > 0:
            input_vector = torch.cat([x_cont, flat_embeddings], dim=-1)

        # shift target by one
        input_vector[..., self.target_positions] = torch.roll(
            input_vector[..., self.target_positions], shifts=1, dims=1
        )

        if one_off_target is not None:  # set first target input (which is rolled over)
            input_vector[:, 0, self.target_positions] = one_off_target
        else:
            input_vector = input_vector[:, 1:]
        # shift target
        return input_vector

    def add_input_vector(self, input_vector: torch.Tensor):
        return self.target_embed(input_vector[..., :self.cont_size]) + self.covariate_embed(input_vector[..., self.cont_size:])

    def encode(self, x: Dict[str, torch.Tensor]) -> HiddenState:
        r"""Encode sequence into hidden state
        Shape:
            output: (B, P-1, d_model)
        """
        # encode using rnn
        assert x["encoder_lengths"].min() > 0
        input_vector = self.construct_input_vector(x["encoder_cat"], x["encoder_cont"])  # (B, P, n_fea), (B, P, n_target) > (B, P-1, n_target+embed(n_fea))
        src = self.add_input_vector(input_vector)  # (B, P-1, n_target+embed(n_fea)) > (B, P-1, d_model)
        encoder_output = self.rnn(src, mask=nn.Transformer.generate_square_subsequent_mask(src.shape[1]).to(src.device))
        return encoder_output

    def decode_autoregressive(
        self,
        decode_one: Callable,
        first_target: Union[List[torch.Tensor], torch.Tensor],
        target_scale: Union[List[torch.Tensor], torch.Tensor],
        n_decoder_steps: int,
        n_samples: int = 1,
        **kwargs,
    ) -> Union[List[torch.Tensor], torch.Tensor]:
        """
        Shape:
            first_target: (B*n_sample, 1)
        """
        # make predictions which are fed into next step
        output = []
        normalized_output = [first_target]
        for idx in range(n_decoder_steps):
            # get lagged targets
            current_target = decode_one(
                idx, lagged_targets=normalized_output, decoder_length=n_decoder_steps, **kwargs
            )  # >(B*n_sample, 2)

            # get prediction and its normalized version for the next step
            prediction, current_target = self.output_to_prediction(
                current_target, target_scale=target_scale, n_samples=n_samples
            )
            # save normalized output for lagged targets
            normalized_output.append(current_target)
            # set output to unnormalized samples, append each target as n_batch_samples x n_random_samples

            output.append(prediction)
        if isinstance(self.hparams.target, str):
            output = torch.stack(output, dim=1)
        else:
            # for multi-targets
            output = [torch.stack([out[idx] for out in output], dim=1) for idx in range(len(self.target_positions))]
        return output

    def decode_all(
        self,
        input_vector: torch.Tensor,
        decoder_length: int,
    ):
        """
        input_vector: (B, P+Q-1, n_fea+embed(n_target))
        """
        src = self.add_input_vector(input_vector)  # (B, P+Q-1, n_fea+embed(n_target)) > (B, P+Q-1, d_model)
        decoder_output = self.rnn(src, mask=nn.Transformer.generate_square_subsequent_mask(src.shape[1]).to(src.device))  # (B, P+Q-1, d_model)
        decoder_output = decoder_output[:,-decoder_length:]  # (B, Q, d_model)
        if isinstance(self.hparams.target, str):  # single target
            output = self.distribution_projector(decoder_output)
        else:
            output = [projector(decoder_output) for projector in self.distribution_projector]
        return output

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
        Shape:
            input_vector: (B, P+Q-1, d_model)
            output: (B, Q, 4)
        """
        if n_samples is None:
            output = self.decode_all(input_vector, decoder_length)
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
                x = input_vector[:, :decoder_length+idx]  # > (B*n_sample, P-1+idx+1, d_model)
                # lagged_targets = torch.stack(lagged_targets, dim=1)  # > (B*n_sample, idx+1, 1)
                x[:, decoder_length-1:, target_pos] = torch.stack(lagged_targets, dim=1)
                for lag, lag_positions in lagged_target_positions.items():
                    if idx > lag:
                        x[:, -1, lag_positions] = lagged_targets[-lag]
                # prediction = self.decode_all(x, lagged_targets.shape[1])
                prediction = self.decode_all(x, len(lagged_targets))
                prediction = apply_to_list(prediction, lambda x: x[:, -1])  # select the last time step
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
        return output

    def forward(self, x: Dict[str, torch.Tensor], n_samples: int = None) -> Dict[str, torch.Tensor]:
        """
        Forward network
        """
        # encoder_output = self.encode(x)
        # encoder_length = encoder_output.shape[1]

        x_cat = torch.cat([x["encoder_cat"], x["decoder_cat"]], dim=1)  # >(B, P+Q, n_fea)
        x_cont = torch.cat([x["encoder_cont"], x["decoder_cont"]], dim=1)  # >(B, P+Q, n_target)

        input_vector = self.construct_input_vector(x_cat, x_cont)  # >(B, P+Q-1, n_fea+embed(n_target))
        decoder_length = int(x["decoder_lengths"].max())

        if self.training:
            assert n_samples is None, "cannot sample from decoder when training"
        output = self.decode(
            input_vector,
            decoder_length=decoder_length,
            target_scale=x["target_scale"],
            n_samples=n_samples,
        )
        # return relevant part
        return self.to_network_output(prediction=output)

    def create_log(self, x, y, out, batch_idx):
        n_samples = [self.hparams.n_validation_samples, self.hparams.n_plotting_samples][self.training]
        log = super().create_log(
            x,
            y,
            out,
            batch_idx,
            prediction_kwargs=dict(n_samples=n_samples),
            quantiles_kwargs=dict(n_samples=n_samples),
        )
        return log

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
        mode_kwargs: Dict[str, Any] = None,
        n_samples: int = 100,
    ):
        """
        predict dataloader

        Args:
            dataloader: dataloader, dataframe or dataset
            mode: one of "prediction", "quantiles", "samples" or "raw", or tuple ``("raw", output_name)`` where
                output_name is a name in the dictionary returned by ``forward()``
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
            n_samples: number of samples to draw. Defaults to 100.

        Returns:
            output, x, index, decoder_lengths: some elements might not be present depending on what is configured
                to be returned
        """
        if isinstance(mode, str):
            if mode in ["prediction", "quantiles"]:
                if mode_kwargs is None:
                    mode_kwargs = dict(use_metric=False)
                else:
                    mode_kwargs = deepcopy(mode_kwargs)
                    mode_kwargs["use_metric"] = False
            elif mode == "samples":
                mode = ("raw", "prediction")
        return super().predict(
            data=data,
            mode=mode,
            return_decoder_lengths=return_decoder_lengths,
            return_index=return_index,
            n_samples=n_samples,  # new keyword that is passed to forward function
            return_x=return_x,
            show_progress_bar=show_progress_bar,
            fast_dev_run=fast_dev_run,
            num_workers=num_workers,
            batch_size=batch_size,
            mode_kwargs=mode_kwargs,
        )

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
