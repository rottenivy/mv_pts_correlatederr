import warnings
warnings.filterwarnings("ignore")
import pickle
import argparse
import os
import yaml

import numpy as np
import pandas as pd
import torch
import matplotlib

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_forecasting import TimeSeriesDataSet
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger
from pytorch_forecasting.metrics.distributions import MultivariateNormalDistributionLoss
from pytorch_forecasting.data.encoders import (
    GroupNormalizer,
    NaNLabelEncoder,
)

from metrics import get_metrics
from model import DeepAR, ARTransformer

matplotlib.use("Agg")

parser = argparse.ArgumentParser()
parser.add_argument('--device', type=int, default=0)
parser.add_argument('--model', type=str, default="deepar")
parser.add_argument('--dataset', type=str, default="exchange_rate_nips")
parser.add_argument('--batch_size', type=int, default=20)
parser.add_argument('--hidden_size', type=int, default=40)
parser.add_argument('--prediction_horizon', type=int, default=None)
parser.add_argument('--num_repeat', type=int, default=10)
parser.add_argument('--n_heads', type=int, default=2)
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--lr', type=float, default=1e-03)
args = parser.parse_args()

pl.seed_everything(args.seed)

with open('./datasets/pred_horizon_dict.pkl', 'rb') as f:
    pred_horizon_dict = pickle.load(f)
with open('./datasets/pred_rolling_dict.pkl', 'rb') as f:
    pred_rolling_dict = pickle.load(f)
with open('./datasets/dataset_freq.pkl', 'rb') as f:
    dataset_freq_dict = pickle.load(f)

default_rolling = {'B':5, '30min':56, 'M':1, 'W':3, '5min':56, 'D':5, 'Q':1, 'H':7, 'Y':1}

args.prediction_horizon = pred_horizon_dict[args.dataset]
args.num_pred_rolling = pred_rolling_dict[args.dataset] if args.dataset in pred_rolling_dict.keys() else default_rolling[dataset_freq_dict[args.dataset]]

def main():
    ################################## Load Data ##################################
    data = pd.read_csv("./datasets/%s.csv"%(args.dataset))
    if dataset_freq_dict[args.dataset] in ['30min', '5min', 'H', 'T']:
        data['datetime'] = pd.to_datetime(data['datetime'])
        data['tod'] = (data['datetime'].values - data['datetime'].values.astype("datetime64[D]")) / np.timedelta64(1, "D")
        data['dow'] = data['datetime'].dt.weekday
        time_varying_known_cats = ['tod', 'dow']
        data = data.astype(dict(sensor=str, tod=str, dow=str))
        if dataset_freq_dict[args.dataset] == 'H':
            lags = {"value": [24, 168]}
        else:
            lags = {"value": [2, 4, 12, 24, 48]}
    elif dataset_freq_dict[args.dataset] in ['B', 'D']:
        data['datetime'] = pd.to_datetime(data['datetime'])
        data['dow'] = data['datetime'].dt.weekday
        time_varying_known_cats = ['dow']
        data = data.astype(dict(sensor=str, dow=str))
        lags = {"value": [7, 14]}
    else:
        time_varying_known_cats = []
        data = data.astype(dict(sensor=str))
        lags = {}

    ################################## Create Dataloaders ##################################
    validation_cutoff = data["time_idx"].max() - args.prediction_horizon - args.num_pred_rolling + 1
    training_cutoff = validation_cutoff - (data["time_idx"].max() - validation_cutoff)

    training = TimeSeriesDataSet(
        data[lambda x: x.time_idx <= training_cutoff],
        time_idx="time_idx",
        target="value",
        target_normalizer=GroupNormalizer(groups=["sensor"], transformation=None),
        categorical_encoders={"sensor": NaNLabelEncoder().fit(data.sensor)},
        group_ids=["sensor"],
        static_categoricals=["sensor"],
        time_varying_known_categoricals=time_varying_known_cats,
        time_varying_unknown_reals=["value"],
        lags=lags,
        min_encoder_length=args.prediction_horizon,
        max_encoder_length=args.prediction_horizon,
        min_prediction_length=args.prediction_horizon,
        max_prediction_length=args.prediction_horizon,
        allow_missing_timesteps=False,
    )

    validation = TimeSeriesDataSet.from_dataset(training, data[lambda x: x.time_idx <= validation_cutoff], min_prediction_idx=training_cutoff + 1)
    testing = TimeSeriesDataSet.from_dataset(training, data, min_prediction_idx=validation_cutoff + 1)

    train_dataloader = training.to_dataloader(
        train=True, batch_size=args.batch_size, num_workers=0, batch_sampler="synchronized"
    )
    val_dataloader = validation.to_dataloader(
        train=False, batch_size=args.batch_size, num_workers=0, batch_sampler="synchronized"
    )
    test_dataloader = testing.to_dataloader(
        train=False, batch_size=args.batch_size, num_workers=0, batch_sampler="synchronized"
    )

    ################################## Initialize Model ##################################
    early_stop_callback = EarlyStopping(monitor="val_loss", patience=10, verbose=False, mode="min")
    checkpoint_callback = ModelCheckpoint(filename='{epoch}-{val_loss:.2f}', save_top_k=1, monitor="val_loss", mode="min")

    f = open("./config/config.yaml")
    configs = yaml.load(f, Loader=yaml.Loader)
    f.close()

    log_file = "%s_baseline_B%s_Q%s_H%s_lr%s"%(args.dataset, args.batch_size, args.prediction_horizon, args.hidden_size, args.lr)
    logger = TensorBoardLogger(save_dir="logs", name=args.model, version=log_file)

    trainer = pl.Trainer(
        logger=logger,
        max_steps=configs['train']['max_steps'],
        accelerator='gpu',
        devices=[args.device],
        enable_model_summary=True,
        gradient_clip_val=configs['train']['gradient_clip_val'],
        callbacks=[early_stop_callback, checkpoint_callback],
        limit_train_batches=configs['train']['limit_train_batches'],
        enable_checkpointing=True,
        accumulate_grad_batches=16
    )

    if args.model == "deepar":
        model = DeepAR
        net = model.from_dataset(
            training,
            cell_type=configs[args.model]['cell_type'],
            hidden_size=args.hidden_size,
            rnn_layers=configs[args.model]['rnn_layers'],
            dropout=configs['train']['dropout'],
            loss=MultivariateNormalDistributionLoss(),
            optimizer="adam",
            learning_rate=args.lr,
            weight_decay=float(configs['train']['weight_decay']),
            reduce_on_plateau_patience=configs['train']['reduce_on_plateau_patience']
        )
    elif args.model == "gpt":
        model = ARTransformer
        net = model.from_dataset(
            training,
            n_heads=configs[args.model]['n_heads'],
            hidden_size=args.hidden_size,
            rnn_layers=configs[args.model]['rnn_layers'],
            dropout=configs['train']['dropout'],
            loss=MultivariateNormalDistributionLoss(),
            optimizer="adam",
            learning_rate=args.lr,
            weight_decay=float(configs['train']['weight_decay']),
            reduce_on_plateau_patience=configs['train']['reduce_on_plateau_patience']
        )
    else:
        raise Exception("Choose among deepar, lstm, and tft")

    trainer.fit(
        net,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
    )

    ################################## Evaluate Model ##################################
    best_model = model.load_from_checkpoint(checkpoint_callback.best_model_path, map_location="cuda:%s"%(args.device)).to("cuda:%s"%(args.device))

    metrics = []
    crps_mean_noagg_all, crps_noagg_all, crps_sum_noagg_all = [], [], []
    for i in range(args.num_repeat):
        raw_predictions, x = best_model.predict(test_dataloader, mode="raw", n_samples=100, show_progress_bar=True, return_x=True)
        preds = raw_predictions['prediction'].cpu()
        actuals = x['decoder_target']

        try:
            preds = preds.reshape(args.num_pred_rolling, preds.shape[0]//args.num_pred_rolling, preds.shape[1], preds.shape[2])
            actuals = actuals.reshape(args.num_pred_rolling, actuals.shape[0]//args.num_pred_rolling, actuals.shape[1])
        except:
            preds = preds.unsqueeze(0)
            actuals = actuals.unsqueeze(0)

        agg_metric, crps_mean_noagg, crps_noagg, crps_sum_noagg = get_metrics(preds, actuals)
        crps_mean_noagg_all.append(crps_mean_noagg)
        crps_noagg_all.append(crps_noagg)
        crps_sum_noagg_all.append(crps_sum_noagg)

        metrics.append(agg_metric)

    metrics = np.array(metrics)
    metrics = np.concatenate([metrics.mean(0).reshape(-1, 1), metrics.std(0).reshape(-1, 1)], axis=1)

    crps_mean_noagg = torch.stack(crps_mean_noagg_all)
    crps_noagg = torch.stack(crps_noagg_all)
    crps_sum_noagg = torch.stack(crps_sum_noagg_all)

    if not os.path.isdir("./metrics_raw/%s"%(args.model)):
        os.makedirs("./metrics_raw/%s"%(args.model))

    torch.save(crps_mean_noagg, './metrics_raw/%s/%s_crps_mean.pt'%(args.model, log_file))
    torch.save(crps_noagg, './metrics_raw/%s/%s_crps.pt'%(args.model, log_file))
    torch.save(crps_sum_noagg, './metrics_raw/%s/%s_crps_sum.pt'%(args.model, log_file))

    if not os.path.isdir("./metrics/%s"%(args.model)):
        os.makedirs("./metrics/%s"%(args.model))

    with open('./metrics/%s/%s.txt'%(args.model, log_file), 'w') as f:
        for i in range(metrics.shape[0]):
            if i != metrics.shape[0]-1:
                f.write('& %.4f$\pm$%.4f'%(metrics[i, 0], metrics[i, 1]))
            else:
                f.write('& %.4f$\pm$%.4f \n'%(metrics[i, 0], metrics[i, 1]))

    return checkpoint_callback.best_model_score


if __name__ == "__main__":
    score = main()
