from engine.Config import Config, HookBuilder
from engine.Hook import LoggerHook
from utils.hook import ExtendMLFlowLoggerHook
import copy
import typing
import argparse

from utils.common import *
from utils.hyperparameter_sweep import apply_optuna_hyperparameter_sweep
from utils.dpa import apa_cutmix
import torch
import optuna

from torch.optim.lr_scheduler import CosineAnnealingLR

from utils.build_dataset import build_dataset, split_csv_or_list
from BiFusion_DEMT_DAv2 import (
    BiFusion_DEMT_DAv2_Trainer,
    MeanTeacherEvalHook_DAv2_addDepthTrainSignal,
)


class BiFusion_DEMT_DAv2_fullEMA_Trainer(BiFusion_DEMT_DAv2_Trainer):
    """
    Full-EMA variant:
    - Phase 1: EMA-update teacher RGB encoder + decoder + head from student.
    - Phase 2: Freeze EMA-managed blocks; train only non-EMA teacher blocks.
    """

    def run_step_(self) -> None:
        self.stu_model.train()
        self.tea_model.train()
        self._freeze_dav2_backbone()
        device = next(self.stu_model.parameters()).device

        phase1_info = {'labeled_loss': [], 'unlabeled_rgbd_loss': [],
                       'consistency_weight': [], 'unlabeled_rgbd_cutmix_loss': [], 'loss': []}
        phase2_info = {'teacher_labeled_loss': [], 'depth_learn_from_stu_loss': [], 'loss': []}

        # ========== PHASE 1: Train Student + EMA (full RGB path: encoder + decoder + head) ==========
        for batch_id, data in enumerate(self.train_dataloader):
            self.stu_optimizer.zero_grad()
            img_s, img, label = data['image_s'], data['image'], data['label']
            img_s, img, label = img_s.to(device), img.to(device), label.to(device)

            unlabeled_img = img[self.labeled_bs:]
            unlabeled_img_s = img_s[self.labeled_bs:]
            label = label[:self.labeled_bs]

            stu_pred = self.stu_model(img_s)
            labeled_stu = stu_pred[:self.labeled_bs]
            unlabeled_stu = stu_pred[self.labeled_bs:]

            self.tea_model.eval()  # remove drop out for higher quality pseudo-labels
            with torch.no_grad():
                tea_output = self.tea_model(unlabeled_img)
            self.tea_model.train()

            unlabeled_img_s_cutmix, ema_pred_u_cutmix = apa_cutmix(
                unlabeled_img_s, tea_output, beta=0.3, t=self.current_epoch, T=self.num_epochs
            )
            pred_u_cutmix = self.stu_model(unlabeled_img_s_cutmix)

            def teacher_confidence_mask(x):
                t = self.teacher_reliable_threshold
                return ((x > t) | (x < 1 - t)).float()

            loss_consist_rgbd_cutmix = self.dpa_loss(pred_u_cutmix, ema_pred_u_cutmix, mask=teacher_confidence_mask(ema_pred_u_cutmix))
            loss_sup = self.class_criterion(labeled_stu, label)
            loss_consist_rgbd = self.consistency_criterion(unlabeled_stu, tea_output, mask=teacher_confidence_mask(tea_output))
            consistency_weight = self._get_current_consistency_weight(
                global_step=batch_id + self.current_epoch * len(self.train_dataloader)
            )
            total_loss = loss_sup + consistency_weight * loss_consist_rgbd + loss_consist_rgbd_cutmix

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.stu_model.parameters(), max_norm=1.0)
            self.stu_optimizer.step()

            global_step = batch_id + self.current_epoch * len(self.train_dataloader)
            self._update_ema_variable(global_step=global_step, model_a=self.tea_model.rgb_encoder, model_b=self.stu_model.encoder1)
            self._update_ema_variable(global_step=global_step, model_a=self.tea_model.decoder5, model_b=self.stu_model.decoder5)
            self._update_ema_variable(global_step=global_step, model_a=self.tea_model.decoder4, model_b=self.stu_model.decoder4)
            self._update_ema_variable(global_step=global_step, model_a=self.tea_model.decoder3, model_b=self.stu_model.decoder3)
            self._update_ema_variable(global_step=global_step, model_a=self.tea_model.decoder2, model_b=self.stu_model.decoder2)
            self._update_ema_variable(global_step=global_step, model_a=self.tea_model.decoder1, model_b=self.stu_model.decoder1)
            self._update_ema_variable(global_step=global_step, model_a=self.tea_model.outconv, model_b=self.stu_model.outconv)

            phase1_info['labeled_loss'].append(loss_sup.item())
            phase1_info['unlabeled_rgbd_loss'].append(loss_consist_rgbd.item())
            phase1_info['unlabeled_rgbd_cutmix_loss'].append(loss_consist_rgbd_cutmix.item())
            phase1_info['consistency_weight'].append(consistency_weight)
            phase1_info['loss'].append(total_loss.item())
            self.scheduler.step()

        p1 = {f'phase1_{k}': np.mean(v) for k, v in phase1_info.items()}
        p1.update(lr_logging_dict(self.stu_optimizer, 'lr'))
        self._add_info(p1)

        # ========== PHASE 2: Train teacher non-EMA blocks only ==========
        for _, data in enumerate(self.train_dataloader):
            self.tea_optimizer.zero_grad()
            img, label = data['image'], data['label']
            img, label = img.to(device), label.to(device)

            labeled_img = img[:self.labeled_bs]
            unlabeled_img = img[self.labeled_bs:]
            label = label[:self.labeled_bs]

            # Freeze EMA-managed teacher modules in phase 2.
            for param in self.tea_model.rgb_encoder.parameters():
                param.requires_grad_(False)
            for param in self.tea_model.decoder5.parameters():
                param.requires_grad_(False)
            for param in self.tea_model.decoder4.parameters():
                param.requires_grad_(False)
            for param in self.tea_model.decoder3.parameters():
                param.requires_grad_(False)
            for param in self.tea_model.decoder2.parameters():
                param.requires_grad_(False)
            for param in self.tea_model.decoder1.parameters():
                param.requires_grad_(False)
            for param in self.tea_model.outconv.parameters():
                param.requires_grad_(False)

            tea_labeled_rgbd_output = self.tea_model(labeled_img)
            loss_tea_sup = self.class_criterion(tea_labeled_rgbd_output, label)

            tea_unlabeled_rgbd_output = self.tea_model(unlabeled_img)
            with torch.no_grad():
                stu_unlabeled_rgbd_output = self.stu_model(unlabeled_img)
            st = self.student_reliable_threshold
            tea_learn_from_stu_mask = (
                ((stu_unlabeled_rgbd_output > st) & (tea_unlabeled_rgbd_output > 0.5))
                | ((stu_unlabeled_rgbd_output < 1 - st) & (tea_unlabeled_rgbd_output < 0.5))
            ).float()

            round_stu_target = (stu_unlabeled_rgbd_output > st).float()
            depth_learn_from_stu_loss = self.tea_learn_from_stu_criterion(tea_unlabeled_rgbd_output, round_stu_target, mask=tea_learn_from_stu_mask)
            total_loss = loss_tea_sup + depth_learn_from_stu_loss * self.depth_learn_from_stu_weight

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.tea_model.parameters(), max_norm=1.0)
            self.tea_optimizer.step()

            phase2_info['teacher_labeled_loss'].append(loss_tea_sup.item())
            phase2_info['depth_learn_from_stu_loss'].append(depth_learn_from_stu_loss.item())
            phase2_info['loss'].append(total_loss.item())

            for name, param in self.tea_model.named_parameters():
                if (
                    name.startswith('dav2_encoder.')
                    or name.startswith('rgb_encoder.')
                    or name.startswith('decoder')
                    or name.startswith('outconv.')
                ):
                    param.requires_grad_(False)
                else:
                    param.requires_grad_(True)

        if self.tea_scheduler is not None:
            self.tea_scheduler.step()
        p2 = {f'phase2_{k}': np.mean(v) for k, v in phase2_info.items()}
        p2.update(lr_logging_dict_mean_teacher(self.stu_optimizer, self.tea_optimizer))
        self._add_info(p2)


def training(cfg: Config, trial: typing.Optional[optuna.trial.Trial] = None):
    if trial is not None:
        apply_optuna_hyperparameter_sweep(cfg, trial)

    print(cfg.all_config())
    device = get_proper_device(cfg.get('device'))
    set_seed(cfg.get('seed'))

    train_dataloader, val_dataloader, test_dataloaders = build_dataset(cfg)
    assert train_dataloader is not None, "train_dataloader is None"
    assert len(test_dataloaders) > 0, "test_dataloaders is empty"

    iters_per_epoch = len(train_dataloader)
    if iters_per_epoch == 0:
        raise ValueError("Dataloader is empty!")
    nEpoch = int(cfg.get('nEpoch', 300))
    total_iter = nEpoch * iters_per_epoch
    print(f"nEpoch: {nEpoch} | Iters/epoch: {iters_per_epoch} => total train steps: {total_iter}")

    stu_model = getattr(models, cfg.get('model.stu_model.name'))(num_classes=cfg.get('model.num_channels_output')).to(device)

    tea_kwargs = dict(cfg.get('model.tea_model', {}))
    tea_kwargs.pop('name', None)

    tea_model = getattr(models, cfg.get('model.tea_model.name'))(num_classes=cfg.get('model.num_channels_output'), **tea_kwargs).to(device)

    optimizer = torch.optim.SGD(stu_model.parameters(), lr=cfg.get('optimizer.lr'),
                                momentum=cfg.get('optimizer.momentum'), weight_decay=cfg.get('optimizer.weight_decay'))
    tea_optimizer = torch.optim.SGD(tea_model.parameters(), lr=cfg.get('tea_optimizer.lr', cfg.get('optimizer.lr')),
                                    momentum=cfg.get('tea_optimizer.momentum', cfg.get('optimizer.momentum')),
                                    weight_decay=cfg.get('tea_optimizer.weight_decay', cfg.get('optimizer.weight_decay')))
    eta_min = float(cfg.get('scheduler.eta_min', 1e-5))
    scheduler = CosineAnnealingLR(optimizer, T_max=total_iter, eta_min=eta_min)
    tea_scheduler = CosineAnnealingLR(tea_optimizer, T_max=nEpoch, eta_min=eta_min)

    trainer = BiFusion_DEMT_DAv2_fullEMA_Trainer(
        stu_model, tea_model, train_dataloader,
        optimizer, tea_optimizer,
        scheduler, nEpoch,
        ema_alpha=float(cfg.get('Trainer.ema_decay', 0.999)),
        iters_per_epoch=iters_per_epoch,
        rampup_unit=cfg.get('Trainer.rampup_unit', 'epoch'),
        consistency_rampup=float(cfg.get('Trainer.consistency_rampup')),
        consistency=float(cfg.get('Trainer.consistency', 2.0)),
        tea_scheduler=tea_scheduler,
        teacher_reliable_threshold=float(cfg.get('Trainer.teacher_reliable_threshold', 0.75)),
        student_reliable_threshold=float(cfg.get('Trainer.student_reliable_threshold', 0.85)),
        depth_learn_from_stu_weight=float(cfg.get('Trainer.depth_learn_from_stu_weight', 0.3)),
    )
    if cfg.get('Trainer.load_ckpt_path', None) is not None:
        trainer.load_Trainer_ckpt(torch.load(cfg.get('Trainer.load_ckpt_path')))
        print(f"Loaded checkpoint from {cfg.get('Trainer.load_ckpt_path')}")
    else:
        print("No checkpoint loaded")

    hook_builder = HookBuilder(cfg, trainer)
    if val_dataloader is not None:
        hook_builder(MeanTeacherEvalHook_DAv2_addDepthTrainSignal, eval_data_loader=val_dataloader,
                     eval_every_epoch=int(cfg.get('Hook.MeanTeacherEvalHook.eval_every_epoch')), prefix='val_')

    dataset_names = split_csv_or_list(cfg.get('data.test.dataset_name'))
    if len(dataset_names) != len(test_dataloaders):
        if len(dataset_names) == 1:
            dataset_names = ['']
        else:
            dataset_names = [str(i) + '_' for i in range(len(test_dataloaders))]
    for test_dataloader, dataset_name in zip(test_dataloaders, dataset_names):
        hook_builder(MeanTeacherEvalHook_DAv2_addDepthTrainSignal, eval_data_loader=test_dataloader,
                     eval_every_epoch=int(cfg.get('Hook.MeanTeacherEvalHook.eval_every_epoch')), prefix=f'test_{dataset_name}')

    if cfg.get('Hook.ExtendMLFlowLoggerHook.should_use', True):
        hook_builder(ExtendMLFlowLoggerHook, local_dir_save_ckpt=cfg.get('Hook.ExtendMLFlowLoggerHook.local_dir_save_ckpt'),
                     dagshub_dir_save_ckpt=cfg.get('Hook.ExtendMLFlowLoggerHook.dagshub_dir_save_ckpt'),
                     max_save_epoch_interval=int(cfg.get('Hook.ExtendMLFlowLoggerHook.max_save_epoch_interval')),
                     log_every_epoch=int(cfg.get('Hook.ExtendMLFlowLoggerHook.log_every_epoch', 1)),
                     criteria=cfg.get('Hook.ExtendMLFlowLoggerHook.criteria'),
                     dagshub_destination_src_file=str(cfg.get('Hook.ExtendMLFlowLoggerHook.dagshub_destination_src_file')),
                     list_src_dir_files=list(cfg.get('Hook.ExtendMLFlowLoggerHook.list_src_dir_files')),
                     dagshub_meta_dir=str(cfg.get('Hook.ExtendMLFlowLoggerHook.dagshub_meta_dir')),
                     meta_info=dict(cfg.get('Hook.ExtendMLFlowLoggerHook.meta_info')),
                     dagshub_repo_owner=cfg.get('Hook.ExtendMLFlowLoggerHook.dagshub_repo_owner'),
                     dagshub_repo_name=cfg.get('Hook.ExtendMLFlowLoggerHook.dagshub_repo_name'),
                     experiment_name=str(cfg.get('Hook.ExtendMLFlowLoggerHook.experiment_name')),
                     dir_save_plot=cfg.get('Hook.ExtendMLFlowLoggerHook.dir_save_plot'),
                     logging_fields=list(cfg.get('Hook.ExtendMLFlowLoggerHook.logging_fields')),
                     run_name=cfg.get('Hook.ExtendMLFlowLoggerHook.run_name'),
                     cfg=cfg,
                     )
    hook_builder(LoggerHook, logger_file='logs/simple.json')

    trainer.train()

    criteria = cfg.get('score_criteria')
    for info in reversed(trainer.info_storage.info_storage):
        if criteria in info:
            return info[criteria]
    raise ValueError(f"Criteria {criteria} does not exist in info_storage")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DAv2 Fusion Mean Teacher training with full EMA on RGB encoder+decoder path.')
    parser.add_argument('--optuna_trial_times', type=int, default=4, help='Optuna trials; 0 = no Optuna.')
    parser.add_argument('--config', type=str, default='cfg/BiFusion_DEMT_DAv2_fullEMA.yaml', help='Path to YAML config')
    args, unknown = parser.parse_known_args()
    cfg = Config(config_file=args.config, cli_overrides=unknown)

    if args.optuna_trial_times == 0:
        score = training(cfg)
        print(f"Score: {score}")
    else:
        def objective(trial):
            trial_cfg = copy.deepcopy(cfg)
            return training(trial_cfg, trial)

        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=args.optuna_trial_times)
        print("Best trial:")
        trial = study.best_trial
        print(f"  Value: {trial.value}")
        print("  Params:")
        for key, value in trial.params.items():
            print(f"    {key}: {value}")
