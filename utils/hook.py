from engine.Hook import MLFlowLoggerHook
import os
import copy
import typing
import torch
import mlflow
from engine.Trainer import Trainer


class ExtendMLFlowLoggerHook(MLFlowLoggerHook):
    ## combine smartsavehook and mlflowloggerhook
    def __init__(self, trainer: Trainer,
                meta_info: dict | None = None, dagshub_meta_dir: str = 'meta',
                local_dir_save_ckpt: str = 'ckpt', dagshub_dir_save_ckpt: str = 'ckpt', max_save_epoch_interval: int = 50, criteria: str = 'test_stu_Dice',
                dagshub_destination_src_file: str = 'src_file', list_src_dir_files: typing.List[str] | None = None,
                plot_every_epoch: int = 1,
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
        self.plot_every_epoch = max(1, int(plot_every_epoch))
        self.accumulate_step_for_logging: list[int] = []

    def _metric_value_for_mlflow(self, value: typing.Any) -> float | None:
        if isinstance(value, (int, float, bool)):
            return float(value)
        if isinstance(value, torch.Tensor):
            try:
                return float(value.detach().cpu().item())
            except (ValueError, RuntimeError):
                return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

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
    def before_train(self) -> None:
        super().before_train()
        self._log_source_files()
        self._log_meta_info()

    def log_accumulate_info_to_mlflow(self) -> None:
        if not self.accumulate_step_for_logging:
            return
        info = self.trainer.info_storage.all_info()
        for step in self.accumulate_step_for_logging:
            if step < 0 or step >= len(info):
                continue
            for key, value in info[step].items():
                if not self.found_in_logging_fields(key):
                    continue
                fv = self._metric_value_for_mlflow(value)
                if fv is None:
                    continue
                mlflow.log_metric(key, fv, step=step)
        self.accumulate_step_for_logging = []

    def after_train_epoch(self) -> None:
        self.accumulate_step_for_logging.append(self.trainer.current_epoch)

        should_log = (self.trainer.current_epoch + 1) % self.plot_every_epoch == 0 or (self.trainer.current_epoch + 1) == self.trainer.num_epochs
        if should_log:
            self.log_accumulate_info_to_mlflow()

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
        self.log_accumulate_info_to_mlflow()

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