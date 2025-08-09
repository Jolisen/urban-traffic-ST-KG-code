import os
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import torchmetrics
from utils.callbacks.base import BestEpochCallback
import utils.metrics


class PlotValidationPredictionsCallback(BestEpochCallback):
    def __init__(self, monitor="", mode="min"):
        super(PlotValidationPredictionsCallback, self).__init__(monitor=monitor, mode=mode)
        self.ground_truths = []
        self.predictions = []

    def on_fit_start(self, trainer, pl_module):
        self.ground_truths.clear()
        self.predictions.clear()

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        super().on_validation_batch_end(trainer, pl_module, outputs, batch, batch_idx, dataloader_idx)
        if trainer.current_epoch != self.best_epoch:
            return
        self.ground_truths.clear()
        self.predictions.clear()
        predictions, y = outputs
        predictions = predictions.cpu().numpy()
        y = y.cpu().numpy()
        self.ground_truths.append(y[:, 0, :])
        self.predictions.append(predictions[:, 0, :])

    def on_fit_end(self, trainer, pl_module):
        # === 获取参数 ===
        model_name = pl_module.hparams.model_name
        hidden_dim = pl_module.hparams.hidden_dim
        max_epochs = trainer.max_epochs
        seq_len = pl_module.hparams.seq_len
        pre_len = pl_module.hparams.pre_len

        # === 构建嵌套保存路径 ===
        sub_folder = f"seq{seq_len}-pre{pre_len}-epochs{max_epochs}-hidden{hidden_dim}"
        save_folder = os.path.join("out", model_name, sub_folder)
        os.makedirs(save_folder, exist_ok=True)


        # === 获取预测和真实值 ===
        ground_truth = np.concatenate(self.ground_truths, 0)
        predictions = np.concatenate(self.predictions, 0)

        # === 保存为 CSV 文件 ===
        pd.DataFrame(predictions).to_csv(os.path.join(save_folder, "test_prediction.csv"), index=False)
        pd.DataFrame(ground_truth).to_csv(os.path.join(save_folder, "test_true.csv"), index=False)

        # === 计算评估指标 ===
        pred_tensor = torch.tensor(predictions)
        gt_tensor = torch.tensor(ground_truth)
        rmse = torch.sqrt(torchmetrics.functional.mean_squared_error(pred_tensor, gt_tensor)).item()
        mae = torchmetrics.functional.mean_absolute_error(pred_tensor, gt_tensor).item()
        acc = utils.metrics.accuracy(pred_tensor, gt_tensor).item()
        r2 = utils.metrics.r2(pred_tensor, gt_tensor).item()
        expl_var = utils.metrics.explained_variance(pred_tensor, gt_tensor).item()

        print(f"RMSE={rmse:.4f} | MAE={mae:.4f} | ACC={acc:.4f} | R2={r2:.4f} | Explained Var={expl_var:.4f}")

        # === 保存评估指标为 CSV ===
        eval_df = pd.DataFrame([{
            "RMSE": rmse,
            "MAE": mae,
            "ACC": acc,
            "R2": r2,
            "Explained_Var": expl_var,
            "Model": model_name,
            "Hidden_Dim": hidden_dim,
            "Epochs": max_epochs,
            "Seq_Len": seq_len,
            "Pre_Len": pre_len
        }])
        eval_df.to_csv(os.path.join(save_folder, "evaluation.csv"), index=False)

        tensorboard = pl_module.logger.experiment
        # for node_idx in range(ground_truth.shape[1]):
        #     plt.clf()
        #     plt.rcParams["font.family"] = "Times New Roman"
        #     fig = plt.figure(figsize=(7, 2), dpi=300)
        #     plt.plot(
        #         ground_truth[:, node_idx],
        #         color="dimgray",
        #         linestyle="-",
        #         label="Ground truth",
        #     )
        #     plt.plot(
        #         predictions[:, node_idx],
        #         color="deepskyblue",
        #         linestyle="-",
        #         label="Predictions",
        #     )
        #     plt.legend(loc="best", fontsize=10)
        #     plt.xlabel("Time")
        #     plt.ylabel("Traffic Speed")
        #     tensorboard.add_figure(
        #         "Prediction result of node " + str(node_idx),
        #         fig,
        #         global_step=len(trainer.train_dataloader) * self.best_epoch,
        #         close=True,
        #     )
