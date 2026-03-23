from data_provider.data_factory import data_provider
from experiments.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np
import pandas as pd
import random

warnings.filterwarnings('ignore')


class Exp_Long_Term_Forecast_Partial(Exp_Basic):
    """
    Entrenamiento parcial: entrena con un subconjunto de variables y evalúa
    en el conjunto completo. Útil para:
    - Generalización en variables no vistas (zero-shot)
    - Entrenamiento eficiente con pocas variables
    """

    def __init__(self, args):
        super(Exp_Long_Term_Forecast_Partial, self).__init__(args)

    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        return nn.MSELoss()

    def vali(self, vali_data, vali_loader, criterion, partial_train=False):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                if 'PEMS' in self.args.data or 'Solar' in self.args.data:
                    batch_x_mark = None
                    batch_y_mark = None
                else:
                    batch_x_mark = batch_x_mark.float().to(self.device)
                    batch_y_mark = batch_y_mark.float().to(self.device)

                if partial_train:
                    batch_x = batch_x[:, :, -self.args.enc_in:]
                    batch_y = batch_y[:, :, -self.args.enc_in:]

                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat(
                    [batch_y[:, :self.args.label_len, :], dec_inp], dim=1
                ).float().to(self.device)

                outputs = self.model(batch_x)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()
                loss = criterion(pred, true)
                total_loss.append(loss)

        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()
        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                if 'PEMS' in self.args.data or 'Solar' in self.args.data:
                    batch_x_mark = None
                    batch_y_mark = None
                else:
                    batch_x_mark = batch_x_mark.float().to(self.device)
                    batch_y_mark = batch_y_mark.float().to(self.device)

                batch_x = batch_x[:, :, -self.args.enc_in:]
                batch_y = batch_y[:, :, -self.args.enc_in:]

                if self.args.efficient_training:
                    _, _, N = batch_x.shape
                    index = np.stack(random.sample(range(N), N))[-self.args.enc_in:]
                    batch_x = batch_x[:, :, index]
                    batch_y = batch_y[:, :, index]

                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat(
                    [batch_y[:, :self.args.label_len, :], dec_inp], dim=1
                ).float().to(self.device)

                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_x)
                        f_dim = -1 if self.args.features == 'MS' else 0
                        outputs = outputs[:, -self.args.pred_len:, f_dim:]
                        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                        loss = criterion(outputs, batch_y)
                        train_loss.append(loss.item())
                else:
                    outputs = self.model(batch_x)
                    f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                    loss = criterion(outputs, batch_y)
                    train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion, partial_train=True)
            test_loss = self.vali(test_data, test_loader, criterion, partial_train=False)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break
            adjust_learning_rate(model_optim, None, epoch + 1, self.args)

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        if test:
            print('loading model')
            self.model.load_state_dict(
                torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth'))
            )

        preds = []
        trues = []
        # Always collect normalized (pre-inverse) values for separate metric reporting and CSV
        preds_norm = []
        trues_norm = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                if 'PEMS' in self.args.data or 'Solar' in self.args.data:
                    batch_x_mark = None
                    batch_y_mark = None
                else:
                    batch_x_mark = batch_x_mark.float().to(self.device)
                    batch_y_mark = batch_y_mark.float().to(self.device)

                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat(
                    [batch_y[:, :self.args.label_len, :], dec_inp], dim=1
                ).float().to(self.device)

                outputs = self.model(batch_x)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()

                # Store normalized values before any inverse transform
                preds_norm.append(outputs)
                trues_norm.append(batch_y)

                if test_data.scale and self.args.inverse:
                    outputs = test_data.inverse_transform(outputs)
                    batch_y = test_data.inverse_transform(batch_y)

                preds.append(outputs)
                trues.append(batch_y)
                if i % 20 == 0:
                    input_data = batch_x.detach().cpu().numpy()
                    gt = np.concatenate((input_data[0, :, -1], batch_y[0, :, -1]), axis=0)
                    pd_vals = np.concatenate((input_data[0, :, -1], outputs[0, :, -1]), axis=0)
                    visual(gt, pd_vals, os.path.join(folder_path, str(i) + '.pdf'))

        preds = np.array(preds)
        trues = np.array(trues)
        preds_norm = np.array(preds_norm)
        trues_norm = np.array(trues_norm)
        print('test shape:', preds.shape, trues.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        preds_norm = preds_norm.reshape(-1, preds_norm.shape[-2], preds_norm.shape[-1])
        trues_norm = trues_norm.reshape(-1, trues_norm.shape[-2], trues_norm.shape[-1])
        print('test shape:', preds.shape, trues.shape)

        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        # --- Metrics on normalized scale ---
        mae_n, mse_n, rmse_n, mape_n, mspe_n, rse_n = metric(preds_norm, trues_norm)
        print('Metrics on NORMALIZED scale (StandardScaler):')
        print('  MAE:  {:.4f}'.format(mae_n))
        print('  MSE:  {:.4f}'.format(mse_n))
        print('  RMSE: {:.4f}'.format(rmse_n))
        print('  MAPE: {:.4f}%'.format(mape_n * 100))
        print('  RSE:  {:.4f}'.format(rse_n))

        # --- Metrics on real scale (if requested and scaler is available) ---
        mae_r = mse_r = rmse_r = mape_r = mspe_r = rse_r = float('nan')
        preds_real = preds
        trues_real = trues
        if getattr(self.args, 'report_real_metrics', 1) and test_data.scale:
            # Inverse-transform normalized arrays to real COP/USD scale for human-readable metrics
            n_vars = preds_norm.shape[-1]
            preds_real = test_data.inverse_transform(
                preds_norm.reshape(-1, n_vars)).reshape(preds_norm.shape)
            trues_real = test_data.inverse_transform(
                trues_norm.reshape(-1, n_vars)).reshape(trues_norm.shape)
            mae_r, mse_r, rmse_r, mape_r, mspe_r, rse_r = metric(preds_real, trues_real)
            print('Metrics on REAL scale (COP/USD values):')
            print('  MAE:  {:.4f}'.format(mae_r))
            print('  MSE:  {:.4f}'.format(mse_r))
            print('  RMSE: {:.4f}'.format(rmse_r))
            print('  MAPE: {:.4f}%'.format(mape_r * 100))
            print('  RSE:  {:.4f}'.format(rse_r))

        f = open("result_long_term_forecast.txt", 'a')
        f.write(setting + "  \n")
        f.write('mse:{}, mae:{}, rse:{}, rmse:{}, mape:{}, mspe:{}'.format(
            mse_n, mae_n, rse_n, rmse_n, mape_n, mspe_n))
        f.write('\n\n')
        f.close()

        # --- Export predictions vs actuals CSV ---
        rows = []
        for sample_idx in range(preds_norm.shape[0]):
            for step in range(preds_norm.shape[1]):
                # Use the last variable column (the target) for single-value output
                rows.append({
                    'sample_idx': sample_idx,
                    'step': step + 1,
                    'true_value_normalized': trues_norm[sample_idx, step, -1],
                    'pred_value_normalized': preds_norm[sample_idx, step, -1],
                    'true_value_real': trues_real[sample_idx, step, -1],
                    'pred_value_real': preds_real[sample_idx, step, -1],
                })
        pred_csv_path = folder_path + 'predictions_vs_actuals.csv'
        pd.DataFrame(rows).to_csv(pred_csv_path, index=False)
        print('Predictions vs actuals saved to:', pred_csv_path)

        # --- Export metrics summary CSV ---
        summary = {
            'setting': [setting],
            'mae_normalized': [mae_n], 'mse_normalized': [mse_n],
            'rmse_normalized': [rmse_n], 'mape_normalized': [mape_n],
            'mspe_normalized': [mspe_n], 'rse_normalized': [rse_n],
            'mae_real': [mae_r], 'mse_real': [mse_r],
            'rmse_real': [rmse_r], 'mape_real': [mape_r],
            'mspe_real': [mspe_r], 'rse_real': [rse_r],
        }
        summary_csv_path = folder_path + 'metrics_summary.csv'
        pd.DataFrame(summary).to_csv(summary_csv_path, index=False)
        print('Metrics summary saved to:', summary_csv_path)

        return
