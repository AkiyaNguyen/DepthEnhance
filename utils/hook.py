from engine.Hook import MLFlowLoggerHook
import os
import copy
import typing
import torch
import mlflow
import re
import matplotlib.pyplot as plt
from engine.Trainer import Trainer
from utils.common import _main_entry_script_path


class ExtendMLFlowLoggerHook(MLFlowLoggerHook):
    ## combine smartsavehook and mlflowloggerhook
    def __init__(self, trainer: Trainer,
                meta_info: dict | None = None, dagshub_meta_dir: str = 'meta',
                local_dir_save_ckpt: str = 'ckpt', dagshub_dir_save_ckpt: str = 'ckpt', max_save_epoch_interval: int = 50, criteria: str = 'test_stu_Dice',
                dagshub_destination_src_file: str = 'src_file', list_src_dir_files: typing.List[str] | None = None,
                interactive_plot: bool = False, dir_save_plot: str = 'plots', log_main_script: bool = True,
                **kwargs) -> None:
        super().__init__(trainer, **kwargs)
        self.meta_info = meta_info if meta_info is not None else {}
        self.dagshub_meta_dir = dagshub_meta_dir
        self.local_dir_save_ckpt = local_dir_save_ckpt
        os.makedirs(self.local_dir_save_ckpt, exist_ok=True)

        self.dagshub_dir_save_ckpt = dagshub_dir_save_ckpt
        self.max_save_epoch_interval = max_save_epoch_interval
        self.criteria = criteria
        self.patience = 0
        self.best_record = None
        self.has_improved = False
        self.ckpt_info = {'ckpt': None, 'epoch': None}

        self.dagshub_destination_src_file = dagshub_destination_src_file
        self.list_src_dir_files = list(list_src_dir_files) if list_src_dir_files is not None else []

        self.log_main_script = log_main_script
        self.interactive_plot = bool(interactive_plot)
        self.dir_save_plot = dir_save_plot
        os.makedirs(self.dir_save_plot, exist_ok=True)
        # if interactive_plot is True, then just plot use after_train_epoch()
    def _log_source_files(self) -> None:  
        for dir_file in self.list_src_dir_files:
            if os.path.isfile(dir_file):
                mlflow.log_artifact(dir_file, artifact_path=self.dagshub_destination_src_file)
                print(f"Logged source file: {dir_file}")
            elif os.path.isdir(dir_file):
                mlflow.log_artifacts(dir_file, artifact_path=os.path.join(self.dagshub_destination_src_file, os.path.basename(dir_file)))
                print(f"Logged source dir: {dir_file}")
            else:
                print(f"[WARN] Source file/dir not found, skipped: {dir_file}")
    def _log_meta_info(self) -> None:
        if self.meta_info:
            # MLflow API: second arg is artifact_file (path under run artifacts), not artifact_path
            mlflow.log_dict(self.meta_info, artifact_file=os.path.join(self.dagshub_meta_dir, 'meta_info.json'))
            print(f"Logged meta info to MLflow!")
    def _log_main_script(self) -> None:
        if self.log_main_script:
            main_p = _main_entry_script_path()
            if main_p:
                mlflow.log_artifact(main_p, artifact_path=self.dagshub_destination_src_file)
                print(f"Logged main script: {main_p}")
            else:
                print("[WARN] Main script path not available (__main__.__file__ missing); skipped.")
    def before_train(self) -> None:
        super().before_train()
        self._log_meta_info()
        if self.log_main_script:
            self._log_main_script()

    def _plot_epoch_metrics_interactive(self) -> None:
        super().after_train_epoch()
    def _plot_epoch_metrics_then_save(self) -> None:
        all_info = self.trainer.info_storage.all_info()
        # Aggregate metric history from all logged epochs
        metric_histories: dict[str, list[tuple[int, float]]] = {}
        for epoch_idx, info in enumerate(all_info):
            if not isinstance(info, dict):
                continue
            for k, v in info.items():
                if not self.found_in_logging_fields(k):
                    continue
                try:
                    value = float(v)
                except (TypeError, ValueError):
                    continue
                metric_histories.setdefault(k, []).append((epoch_idx, value)) # epoch start from 1

        for metric_name, points in metric_histories.items():
            if not points:
                continue
            xs = [p[0] for p in points]
            ys = [p[1] for p in points]
            safe_name = re.sub(r"[^A-Za-z0-9._-]+", "_", metric_name).strip("_") or "metric"
            plot_path = os.path.join(self.dir_save_plot, f"{safe_name}.png")

            plt.figure(figsize=(10, 5))
            plt.plot(xs, ys, linewidth=1.5)
            plt.xlabel("Epoch")
            plt.ylabel(metric_name)
            plt.title(metric_name)
            plt.grid(alpha=0.3)
            plt.tight_layout()
            plt.savefig(plot_path)
            plt.close()
            mlflow.log_artifact(plot_path, artifact_path=self.dir_save_plot)
    def after_train_epoch(self) -> None:
        if self.interactive_plot:
            self._plot_epoch_metrics_interactive()

        latest = self.trainer.info_storage.latest_info()
        self.patience += 1

        if self.criteria not in latest:
            return
        criteria_value = latest[self.criteria]
        if self.best_record is None or criteria_value > self.best_record:
            self.best_record = criteria_value
            self.has_improved = True
            self.ckpt_info['ckpt'] = copy.deepcopy(self.trainer.get_Trainer_ckpt())
            self.ckpt_info['epoch'] = self.trainer.current_epoch + 1

        if self.patience >= self.max_save_epoch_interval and self.has_improved:
            torch.save(self.ckpt_info['ckpt'], os.path.join(self.local_dir_save_ckpt, f"{self.experiment_name}_epoch{self.ckpt_info['epoch']}.pth"))
            self.has_improved = False
            self.patience = 0

    def after_train(self) -> None:
        self._log_source_files()

        if not self.interactive_plot:
            self._plot_epoch_metrics_then_save()

        ## log the last ckpt and the best ckpt then call the parent class
        if self.ckpt_info['ckpt'] is not None:
            ckpt_name = f"best_{self.experiment_name}_epoch{self.ckpt_info['epoch']}.pth"
            ckpt_save_dir = os.path.join(self.local_dir_save_ckpt, ckpt_name)
            torch.save(self.ckpt_info['ckpt'], ckpt_save_dir)
            print(f"Best model saved at {ckpt_save_dir}")
            mlflow.log_artifact(ckpt_save_dir, artifact_path=self.dagshub_dir_save_ckpt)
            print(f"Best model logged to DagsHub MLflow!")

        last_ckpt = copy.deepcopy(self.trainer.get_Trainer_ckpt())
        last_epoch = self.trainer.current_epoch ## +1 already in trainer
        last_ckpt_name = f"last_{self.experiment_name}_epoch{last_epoch}.pth"
        last_ckpt_save_dir = os.path.join(self.local_dir_save_ckpt, last_ckpt_name)
        torch.save(last_ckpt, last_ckpt_save_dir)
        print(f"Last model saved at {last_ckpt_save_dir}")
        mlflow.log_artifact(last_ckpt_save_dir, artifact_path=self.dagshub_dir_save_ckpt)
        print(f"Last model logged to DagsHub MLflow!")

        super().after_train()