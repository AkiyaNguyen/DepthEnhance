from engine.Config import Config, HookBuilder
from engine.Trainer import Trainer
from engine.Hook import LoggerHook, EvalHook, HookBase
from utils.hook import ExtendMLFlowLoggerHook
import copy

from test.eval import evaluate
import typing
import argparse

from utils.common import *
from utils.dpa import apa_cutmix
import torch
import torch.nn as nn
import torch.nn.functional as F
import optuna
import mlflow

from torch.optim.lr_scheduler import LambdaLR
from utils.ramps import sigmoid_rampup

from utils.build_dataset import build_dataset
from utils.loss import MSELoss, WeightedBCEDiceLoss, BCELoss

class DEMT_DAv2_ProtoConsistency(Trainer):
    """
    Mean Teacher trainer with prototype consistency.
    Teacher forward uses RGB only; EMA copies student encoder -> teacher rgb_encoder only.
    """

    def __init__(self, stu_model, tea_model, train_dataloader, stu_optimizer, tea_optimizer, scheduler, num_epochs, ema_alpha,
                 consistency_rampup, consistency, tea_scheduler=None, teacher_reliable_threshold=0.75,
                 student_reliable_threshold=0.85, depth_learn_from_stu_weight=1.0, 
                 prototype_momentum=0.99, prototype_warmup_iter=3000.0, prototype_rampup_length=2000.0,
                 prototype_balance_weight=0.5, prototype_loss_weight=1.0, prototype_fea_layers=1, 
                 prototype_threshold=0.8, **kwargs) -> None:
        super().__init__(num_epochs, **kwargs)
        self.stu_model = stu_model
        self.tea_model = tea_model
        self.train_dataloader = train_dataloader
        self.stu_optimizer = stu_optimizer
        self.tea_optimizer = tea_optimizer
        self.scheduler = scheduler
        self.tea_scheduler = tea_scheduler
        self.ema_alpha = ema_alpha
        self.labeled_bs = self.train_dataloader.batch_sampler.primary_batch_size
        self.consistency_rampup = consistency_rampup
        self.consistency = consistency
        self.class_criterion = WeightedBCEDiceLoss()
        self.consistency_criterion = MSELoss()
        self.tea_learn_from_stu_criterion = BCELoss()
        self.dpa_loss = WeightedBCEDiceLoss()
        self.teacher_reliable_threshold = teacher_reliable_threshold
        self.student_reliable_threshold = student_reliable_threshold
        self.depth_learn_from_stu_weight = depth_learn_from_stu_weight
        self.prototype_momentum = prototype_momentum
        self.prototype_rampup_length = prototype_rampup_length
        self.prototype_balance_weight = prototype_balance_weight
        self.prototype_loss_weight = prototype_loss_weight
        self.prototype_warmup_iter = prototype_warmup_iter
        self.prototype_fea_layers = prototype_fea_layers
        self.prototype_threshold = prototype_threshold
        self._freeze_dav2_backbone()
        
        self.prototype = {
            'fg_student': None,
            'bg_student': None,
            'fg_teacher': None,
            'bg_teacher': None,
        }
    def _freeze_dav2_backbone(self) -> None:
        """Ensure DAv2 backbone is always frozen."""
        if hasattr(self.tea_model, 'dav2_encoder'):
            for param in self.tea_model.dav2_encoder.parameters():
                param.requires_grad_(False)

    def _get_current_consistency_weight(self, global_step):
        return self.consistency * sigmoid_rampup(current=global_step, rampup_length=self.consistency_rampup)
    def _update_ema_variable(self, global_step, model_a: nn.Module, model_b: nn.Module):
        coeff = min(1 - 1 / (global_step + 1), self.ema_alpha)
        for tea_param, stu_param in zip(model_a.parameters(), model_b.parameters()):
            tea_param.data.mul_(coeff).add_(stu_param.data, alpha=1 - coeff)
   
    def _momentum_update_prototype(self, prototype: torch.Tensor, new_features: torch.Tensor) -> torch.Tensor:
        return self.prototype_momentum * prototype + (1 - self.prototype_momentum) * new_features
    def _update_prototype(self, current_prototype: Optional[torch.Tensor], new_features: torch.Tensor) -> torch.Tensor:
        if new_features.numel() == 0:
            return current_prototype
        if current_prototype is None:
            return new_features.mean(dim=0)
        else:
            return self._momentum_update_prototype(current_prototype, new_features.mean(dim=0))

    def _get_current_prototype_weight(self, global_step):
        return self.prototype_loss_weight * sigmoid_rampup(current=max(0.0, global_step - self.prototype_warmup_iter), rampup_length=self.prototype_rampup_length)

    def _proto_loss(self, features: torch.Tensor, prototype: Optional[torch.Tensor]) -> torch.Tensor:
        """Attraction: (1 - cos(f, mu)); prototype is a target — use detached mu in loss."""
        if prototype is None or features.numel() == 0:
            return features.sum() * 0.0
        mu = prototype.detach()
        sim = F.cosine_similarity(features, mu.unsqueeze(0).expand_as(features), dim=1)
        return (1 - sim).mean()

    @staticmethod
    def _gather_pixels(feat_bchw: torch.Tensor, mask_bhw: torch.Tensor) -> torch.Tensor:
        """feat [B,C,H,W], mask [B,H,W] bool or 0/1 float (e.g. after interpolate) -> [N,C]"""
        b, c, h, w = feat_bchw.shape
        f = feat_bchw.permute(0, 2, 3, 1).reshape(-1, c)
        m = mask_bhw.reshape(-1)
        if m.dtype != torch.bool:
            m = m > 0.5 ## convert
        return f[m]

    @staticmethod
    def _interpolate_mask_to_feat(mask_1chw: torch.Tensor, feat_hw: typing.Tuple[int, int]) -> torch.Tensor:
        """mask [B,1,H,W] or [B,H,W] -> [B,Hf,Wf] float, nearest."""
        if mask_1chw.dim() == 3:
            mask_1chw = mask_1chw.unsqueeze(1)
        return F.interpolate(mask_1chw.float(), size=feat_hw, mode="nearest").squeeze(1)

    # def _get_prototype_loss(self, fg_features, bg_features, fg_prototype, bg_prototype) -> typing.Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    #     attraction_loss = self._proto_loss(fg_features, fg_prototype) + self._proto_loss(bg_features, bg_prototype)
    #     repulsion_loss = self._proto_loss(bg_features, fg_prototype) + self._proto_loss(fg_features, bg_prototype)
    #     total_loss = attraction_loss - repulsion_loss * self.prototype_balance_weight
    #     return total_loss, attraction_loss, repulsion_loss

    def _get_prototype_loss(self, global_step, features: torch.Tensor, unlabeled_predict: Optional[torch.Tensor], label: torch.Tensor, role: str) -> typing.Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if global_step < self.prototype_warmup_iter:
            z = torch.tensor(0.0, device=features.device)
            return z, z, z

        hw = features.shape[2:]
        fg_labeled_mask = self._interpolate_mask_to_feat(label > 0.5, hw)
        fg_unlabeled_mask = self._interpolate_mask_to_feat(unlabeled_predict > self.prototype_threshold, hw)
        bg_labeled_mask = self._interpolate_mask_to_feat(label <= 0.5, hw)
        bg_unlabeled_mask = self._interpolate_mask_to_feat(unlabeled_predict <= (1.0 - self.prototype_threshold), hw)

        fg_features = self._gather_pixels(features, torch.cat([fg_labeled_mask, fg_unlabeled_mask], dim=0))
        bg_features = self._gather_pixels(features, torch.cat([bg_labeled_mask, bg_unlabeled_mask], dim=0))

        fg_prototype = self.prototype['fg_' + role]
        bg_prototype = self.prototype['bg_' + role]
        attraction_loss = self._proto_loss(fg_features, fg_prototype) + self._proto_loss(bg_features, bg_prototype)
        repulsion_loss = self._proto_loss(bg_features, fg_prototype) + self._proto_loss(fg_features, bg_prototype)

        total_loss = attraction_loss - repulsion_loss * self.prototype_balance_weight

        self.prototype['fg_' + role] = self._update_prototype(fg_prototype, fg_features).detach()
        self.prototype['bg_' + role] = self._update_prototype(bg_prototype, bg_features).detach()
        
        return total_loss, attraction_loss, repulsion_loss

    def get_Trainer_ckpt(self) -> dict:
        result = dict()
        result['stu_model'] = self.stu_model.state_dict()
        result['tea_model'] = self.tea_model.state_dict()
        result['current_epoch'] = self.current_epoch + 1
        result['stu_optimizer'] = self.stu_optimizer.state_dict()
        result['tea_optimizer'] = self.tea_optimizer.state_dict()
        result['scheduler'] = self.scheduler.state_dict()
        result['tea_scheduler'] = self.tea_scheduler.state_dict() if \
            hasattr(self, 'tea_scheduler') and self.tea_scheduler is not None else None
        result['foreground_student_prototype'] = self.prototype['fg_student']
        result['background_student_prototype'] = self.prototype['bg_student']
        result['foreground_teacher_prototype'] = self.prototype['fg_teacher']
        result['background_teacher_prototype'] = self.prototype['bg_teacher']
        return result

    def load_Trainer_ckpt(self, state_dict: dict) -> None:
        self.stu_model.load_state_dict(state_dict['stu_model'])
        self.tea_model.load_state_dict(state_dict['tea_model'])
        self.current_epoch = state_dict['current_epoch']
        self.stu_optimizer.load_state_dict(state_dict['stu_optimizer'])
        self.tea_optimizer.load_state_dict(state_dict['tea_optimizer'])
        self.scheduler.load_state_dict(state_dict['scheduler'])
        if state_dict.get('tea_scheduler') is not None and self.tea_scheduler is not None:
            self.tea_scheduler.load_state_dict(state_dict['tea_scheduler'])
        dev = next(self.stu_model.parameters()).device
        pf = state_dict.get('foreground_student_prototype')
        pb = state_dict.get('background_student_prototype')
        pt = state_dict.get('foreground_teacher_prototype')
        bt = state_dict.get('background_teacher_prototype')
        self.prototype['fg_student'] = None if pf is None else pf.to(dev)
        self.prototype['bg_student'] = None if pb is None else pb.to(dev)
        self.prototype['fg_teacher'] = None if pt is None else pt.to(dev)
        self.prototype['bg_teacher'] = None if bt is None else bt.to(dev)

    def _start_train_mode(self) -> None:
        self.stu_model.train()

    def run_step_(self) -> None:
        self.stu_model.train()
        self.tea_model.train()
        self._freeze_dav2_backbone()
        device = next(self.stu_model.parameters()).device

        phase1_info = {
            'labeled_loss': [],
            'unlabeled_rgbd_loss': [],
            'consistency_weight': [],
            'unlabeled_rgbd_cutmix_loss': [],
            'stu_prototype_loss': [],
            'stu_prototype_attraction_loss': [],
            'stu_prototype_repulsion_loss': [],
            'stu_prototype_weight': [],
            'loss': [],
        }
        phase2_info = {
            'teacher_labeled_loss': [],
            'depth_learn_from_stu_loss': [],
            'tea_prototype_loss': [],
            'tea_prototype_attraction_loss': [],
            'tea_prototype_repulsion_loss': [],
            'tea_prototype_weight': [],
            'loss': [],
        }

        # ========== PHASE 1: Train Student + EMA (encoder only) ==========
        for batch_id, data in enumerate(self.train_dataloader):
            self.stu_optimizer.zero_grad()
            global_step = batch_id + self.current_epoch * len(self.train_dataloader)
            img_s, img, label = data['image_s'], data['image'], data['label']
            img_s, img, label = img_s.to(device), img.to(device), label.to(device)

            unlabeled_img = img[self.labeled_bs:]
            unlabeled_img_s = img_s[self.labeled_bs:]
            label = label[:self.labeled_bs]

            stu_pred, stu_features = self.stu_model(
                img_s, fp=True, feature_layers=self.prototype_fea_layers
            )
            labeled_stu = stu_pred[: self.labeled_bs]
            unlabeled_stu = stu_pred[self.labeled_bs :]

            with torch.no_grad():
                tea_unlabeled_output = self.tea_model(unlabeled_img)

            unlabeled_img_s_cutmix, ema_pred_u_cutmix = apa_cutmix(
                unlabeled_img_s, tea_unlabeled_output, beta=0.3, t=self.current_epoch, T=self.num_epochs
            )
            # unlabeled_img_s_cutmix, ema_pred_u_cutmix = apa_cutmix(
            #     unlabeled_img_s, tea_output, beta=0.3, t=self.current_epoch, T=self.num_epochs
            # )
            pred_u_cutmix = self.stu_model(unlabeled_img_s_cutmix)

            def teacher_confidence_mask(x):
                t = self.teacher_reliable_threshold
                return ((x > t) | (x < 1 - t)).float()

            loss_consist_rgbd_cutmix = self.dpa_loss(pred_u_cutmix, ema_pred_u_cutmix, mask=teacher_confidence_mask(ema_pred_u_cutmix))
            loss_sup = self.class_criterion(labeled_stu, label)
            loss_consist_rgbd = self.consistency_criterion(unlabeled_stu, tea_unlabeled_output, mask=teacher_confidence_mask(tea_unlabeled_output))
            consistency_weight = self._get_current_consistency_weight(
                global_step=global_step
            )


            prototype_weight = self._get_current_prototype_weight(global_step)
            stu_prototype_loss, stu_attraction_loss, stu_repulsion_loss = self._get_prototype_loss(global_step, stu_features, \
                stu_pred[self.labeled_bs:], label, role='student')
            
            total_loss = (
                loss_sup
                + consistency_weight * loss_consist_rgbd
                + loss_consist_rgbd_cutmix
                + stu_prototype_loss * prototype_weight
            )


            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.stu_model.parameters(), max_norm=1.0)
            self.stu_optimizer.step()
            self._update_ema_variable(
                global_step=batch_id + self.current_epoch * len(self.train_dataloader),
                model_a=self.tea_model.rgb_encoder,
                model_b=self.stu_model.encoder1,
            )

            # if add_prototype:
            #     self.prototype["fg_student"] = self._update_prototype(
            #         self.prototype["fg_student"], fg_stu.detach()
            #     ).detach()
            #     self.prototype["bg_student"] = self._update_prototype(
            #         self.prototype["bg_student"], bg_stu.detach()
            #     ).detach()

            phase1_info['labeled_loss'].append(loss_sup.item())
            phase1_info['unlabeled_rgbd_loss'].append(loss_consist_rgbd.item())
            phase1_info['unlabeled_rgbd_cutmix_loss'].append(loss_consist_rgbd_cutmix.item())
            phase1_info['consistency_weight'].append(consistency_weight)
            phase1_info['stu_prototype_loss'].append(stu_prototype_loss.item())
            phase1_info['stu_prototype_attraction_loss'].append(stu_attraction_loss.item())
            phase1_info['stu_prototype_repulsion_loss'].append(stu_repulsion_loss.item())
            phase1_info['stu_prototype_weight'].append(prototype_weight)
            phase1_info['loss'].append(total_loss.item())

            self.scheduler.step()


        p1 = {f'phase1_{k}': np.mean(v) for k, v in phase1_info.items()}
        p1.update(lr_logging_dict(self.stu_optimizer, 'lr'))
        self._add_info(p1)

        # ========== PHASE 2: Train Teacher (DAv2 fusion/decoder), freeze rgb_encoder ==========
        for batch_id2, data in enumerate(self.train_dataloader):
            self.tea_optimizer.zero_grad()
            img, label = data['image'], data['label']
            img, label = img.to(device), label.to(device)

            label = label[: self.labeled_bs]
            global_step_p2 = batch_id2 + self.current_epoch * len(self.train_dataloader)

            tea_output, tea_features = self.tea_model(img, fp=True, feature_layers=self.prototype_fea_layers)
            tea_labeled_output = tea_output[: self.labeled_bs]
            tea_unlabeled_output = tea_output[self.labeled_bs :]

            with torch.no_grad():
                stu_unlabeled_output = self.stu_model(img[self.labeled_bs :])
            stu_u_stop = stu_unlabeled_output.detach()

            # Teacher distillation from rounded student: stop-grad through student (targets / agreement mask).

            for param in self.tea_model.rgb_encoder.parameters():
                param.requires_grad_(False)
            loss_tea_sup = self.class_criterion(tea_labeled_output, label)

            st = self.student_reliable_threshold
            tea_learn_from_stu_mask = (
                ((stu_u_stop > st) & (tea_unlabeled_output > 0.5))
                | ((stu_u_stop < 1 - st) & (tea_unlabeled_output < 0.5))
            ).float()
            round_stu_target = (stu_u_stop > st).float()
            depth_learn_from_stu_loss = self.tea_learn_from_stu_criterion(
                tea_unlabeled_output, round_stu_target, mask=tea_learn_from_stu_mask
            )

            prototype_weight = self._get_current_prototype_weight(global_step_p2)

            tea_prototype_loss, tea_attraction_loss, tea_repulsion_loss = self._get_prototype_loss(global_step_p2, tea_features, \
                tea_unlabeled_output, label, role='teacher')
            
            total_loss = (
                loss_tea_sup
                + depth_learn_from_stu_loss * self.depth_learn_from_stu_weight
                + tea_prototype_loss * prototype_weight
            )
            # total_loss = loss_tea_sup + depth_learn_from_stu_loss * self.depth_learn_from_stu_weight

            # fg_tea, bg_tea = None, None  ## init current batch for prototype; if no pixels, loss will be 0 and no update.
            # tea_prototype_loss = torch.tensor(0.0, device=device)
            # tea_attraction_loss = torch.tensor(0.0, device=device)
            # tea_repulsion_loss = torch.tensor(0.0, device=device)

            # if add_prototype:
            #     feat_hw = tea_features.shape[2:]
            #     label_resized = self._interpolate_mask_to_feat(label, feat_hw)
            #     tea_pred_resized = self._interpolate_mask_to_feat(
            #         tea_unlabeled_output.detach(), feat_hw
            #     )

            #     labeled_resized_mask = label_resized > 0.5
            #     thr = self.prototype_threshold

            #     fg_tea = self._gather_pixels(
            #         tea_features,
            #         torch.cat([labeled_resized_mask, tea_pred_resized > thr], dim=0),
            #     )
            #     bg_tea = self._gather_pixels(
            #         tea_features,
            #         torch.cat([~labeled_resized_mask, tea_pred_resized < (1.0 - thr)], dim=0),
            #     )

            #     tea_prototype_loss, tea_attraction_loss, tea_repulsion_loss = self._get_prototype_loss(
            #         fg_tea, bg_tea, self.prototype["fg_teacher"], self.prototype["bg_teacher"]
            #     )
            #     total_loss += prototype_weight * tea_prototype_loss

            # # Relational match: same spatial pixels on labeled subset only (paired fg / paired bg).
            # unlabeled_matching_fg_mask = (stu_pred_low > self.prototype_threshold) & (tea_pred_low > self.prototype_threshold)
            # unlabeled_matching_bg_mask = (stu_pred_low < (1.0 - self.prototype_threshold)) & (tea_pred_low < (1.0 - self.prototype_threshold))
            # fg_matching_stu = self._gather_pixels(stu_features, torch.cat([labeled_low_mask, unlabeled_matching_fg_mask], dim=0))
            # bg_matching_stu = self._gather_pixels(stu_features, torch.cat([labeled_low_mask, unlabeled_matching_bg_mask], dim=0))
            # fg_matching_tea = self._gather_pixels(tea_features, torch.cat([labeled_low_mask, unlabeled_matching_fg_mask], dim=0))
            # bg_matching_tea = self._gather_pixels(tea_features, torch.cat([labeled_low_mask, unlabeled_matching_bg_mask], dim=0))



            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.tea_model.parameters(), max_norm=1.0)
            self.tea_optimizer.step()

            # if add_prototype:
            #     self.prototype["fg_teacher"] = self._update_prototype(
            #         self.prototype["fg_teacher"], fg_tea.detach()
            #     ).detach()
            #     self.prototype["bg_teacher"] = self._update_prototype(
            #         self.prototype["bg_teacher"], bg_tea.detach()
            #     ).detach()

            phase2_info['teacher_labeled_loss'].append(loss_tea_sup.item())
            phase2_info['depth_learn_from_stu_loss'].append(depth_learn_from_stu_loss.item())
            phase2_info['tea_prototype_loss'].append(tea_prototype_loss.item())
            phase2_info['tea_prototype_attraction_loss'].append(tea_attraction_loss.item())
            phase2_info['tea_prototype_repulsion_loss'].append(tea_repulsion_loss.item())
            phase2_info['tea_prototype_weight'].append(prototype_weight)
            phase2_info['loss'].append(total_loss.item())

            for name, param in self.tea_model.named_parameters():
                if name.startswith('dav2_encoder.'):
                    param.requires_grad_(False)
                else:
                    param.requires_grad_(True)



        if self.tea_scheduler is not None:
            self.tea_scheduler.step()

        p2 = {f'phase2_{k}': np.mean(v) for k, v in phase2_info.items()}
        p2.update(lr_logging_dict_mean_teacher(self.stu_optimizer, self.tea_optimizer))
        self._add_info(p2)

# class LastSaveHook(HookBase):
#     def __init__(self, trainer: Trainer, save_dir: str, **kwargs) -> None:
#         super().__init__(trainer, **kwargs)
#         self.save_dir = save_dir
#         os.makedirs(self.save_dir, exist_ok=True)
    
#     def after_train(self) -> None:
#         torch.save(self.trainer.stu_model.state_dict(), os.path.join(self.save_dir, f'stu_model_last.pth'))
#         torch.save(self.trainer.tea_model.state_dict(), os.path.join(self.save_dir, f'tea_model_last.pth'))
#         print(f"Saved last checkpoint to {self.save_dir}")

class MeanTeacherEvalHook_DAv2_ProtoConsistency_CL_InPixel(EvalHook):
    """Eval hook for DAv2 teacher output (RGB-only forward)."""

    def __init__(self, trainer: Trainer, eval_data_loader: torch.utils.data.DataLoader, eval_every_epoch: int, prefix: str = '') -> None:
        super().__init__(trainer, eval_data_loader)
        self.eval_every_epoch = eval_every_epoch
        self.prefix = prefix
        assert self.eval_every_epoch >= 1, "eval_every_epoch must be at least 1"

    def _run_validation(self) -> dict[typing.Any, typing.Any]:
        assert hasattr(self.trainer, 'stu_model') and hasattr(self.trainer, 'tea_model'), \
            "trainer must have stu_model and tea_model attributes"
        self.trainer.stu_model.eval()
        self.trainer.tea_model.eval()
        device = next(self.trainer.stu_model.parameters()).device
        metrics = {
            'stu_ACC_overall': [],
            'tea_rgbd_ACC_overall': [],
            'stu_Dice': [],
            'tea_rgbd_Dice': [],
            'stu_IoU': [],
            'tea_rgbd_IoU': [],
        }
        with torch.no_grad():
            for data in self.eval_data_loader:
                img = data['image'].to(device)
                gt = data['mask'].to(device)
                stu_output = self.trainer.stu_model(img)
                cur_stu_metrics = evaluate(stu_output, gt)
                tea_output = self.trainer.tea_model(img)
                cur_tea_rgbd_metrics = evaluate(tea_output, gt)
                for key, value in cur_stu_metrics.items():
                    metrics['stu_' + key].append(value)
                for key, value in cur_tea_rgbd_metrics.items():
                    metrics['tea_rgbd_' + key].append(value)
        return {self.prefix + key: np.mean(value) for key, value in metrics.items()}

    def after_train_epoch(self) -> None:
        if (self.trainer.current_epoch + 1) % self.eval_every_epoch == 0:
            result = self._run_validation()
            self.trainer._add_info(result)


def training(cfg: Config, trial: typing.Optional[optuna.trial.Trial] = None):
    if trial is not None:
        sweep_config = cfg.get('hyperparameter_sweeping', {})
        for key, settings in sweep_config.items():
            suggested_value = getattr(trial, settings['method'])(**settings['params'])
            cfg.set(key, suggested_value)

    print(cfg.all_config())
    device = get_proper_device(cfg.get('device'))
    set_seed(cfg.get('seed'))

    train_dataloader, val_dataloader, test_dataloader = build_dataset(cfg)
    assert train_dataloader is not None, "train_dataloader is None"
    assert test_dataloader is not None, "test_dataloader is None"

    iters_per_epoch = len(train_dataloader)
    if iters_per_epoch == 0:
        raise ValueError("Dataloader is empty!")
    nEpoch = int(cfg.get('nEpoch', 300))
    total_iter = nEpoch * iters_per_epoch
    print(f"nEpoch: {nEpoch} | Iters/epoch: {iters_per_epoch} => total train steps: {total_iter}")

    stu_model = getattr(models, cfg.get('model.stu_model.name'))(num_classes=cfg.get('model.num_channels_output')).to(device)
    tea_model = getattr(models, cfg.get('model.tea_model.name'))(num_classes=cfg.get('model.num_channels_output')).to(device)

    optimizer = torch.optim.SGD(stu_model.parameters(), lr=cfg.get('optimizer.lr'),
                                momentum=cfg.get('optimizer.momentum'), weight_decay=cfg.get('optimizer.weight_decay'))
    tea_optimizer = torch.optim.SGD(tea_model.parameters(), lr=cfg.get('optimizer.lr'),
                                    momentum=cfg.get('optimizer.momentum'), weight_decay=cfg.get('optimizer.weight_decay'))
    scheduler_power = float(cfg.get('scheduler.power'))
    ## try student scheduler updated by epoch
    scheduler = LambdaLR(optimizer, lambda e: max(0.0, 1.0 - pow(min(e, total_iter) / total_iter, scheduler_power)))
    tea_scheduler = LambdaLR(tea_optimizer, lambda e: max(0.0, 1.0 - pow(min(e, nEpoch) / nEpoch, scheduler_power)))

    trainer = DEMT_DAv2_ProtoConsistency(
        stu_model, tea_model, train_dataloader,
        optimizer, tea_optimizer,
        scheduler, nEpoch,
        ema_alpha=float(cfg.get('Trainer.ema_decay', 0.999)),
        consistency_rampup=float(cfg.get('Trainer.consistency_rampup')),
        consistency=float(cfg.get('Trainer.consistency')),
        tea_scheduler=tea_scheduler,
        teacher_reliable_threshold=float(cfg.get('Trainer.teacher_reliable_threshold', 0.75)),
        student_reliable_threshold=float(cfg.get('Trainer.student_reliable_threshold', 0.85)),
        depth_learn_from_stu_weight=float(cfg.get('Trainer.depth_learn_from_stu_weight', 0.3)),
        # contrastive_weight=float(cfg.get('Trainer.contrastive_weight', 0.5)),
        # contrastive_rampup=float(cfg.get('Trainer.contrastive_rampup', 5000.0)),
        prototype_momentum=float(cfg.get('Trainer.prototype_momentum', 0.99)),
        prototype_warmup_iter=float(cfg.get('Trainer.prototype_warmup_iter', 3000.0)),
        prototype_rampup_length=float(cfg.get('Trainer.prototype_rampup_length', 2000.0)),
        prototype_balance_weight=float(cfg.get('Trainer.prototype_balance_weight', 0.5)),
        prototype_loss_weight=float(cfg.get('Trainer.prototype_loss_weight', 1.0)),
        prototype_fea_layers=int(cfg.get('Trainer.prototype_fea_layers', 2)),
        prototype_threshold=float(cfg.get('Trainer.prototype_threshold', 0.8)),
    )
    if cfg.get('Trainer.load_ckpt_path', None) is not None:
        trainer.load_Trainer_ckpt(torch.load(cfg.get('Trainer.load_ckpt_path')))
        print(f"Loaded checkpoint from {cfg.get('Trainer.load_ckpt_path')}")
    else:
        print("No checkpoint loaded")

    hook_builder = HookBuilder(cfg, trainer)
    if val_dataloader is not None:
        hook_builder(MeanTeacherEvalHook_DAv2_ProtoConsistency_CL_InPixel, eval_data_loader=val_dataloader,
                     eval_every_epoch=int(cfg.get('Hook.MeanTeacherEvalHook.eval_every_epoch')), prefix='val_')
    hook_builder(MeanTeacherEvalHook_DAv2_ProtoConsistency_CL_InPixel, eval_data_loader=test_dataloader,
                 eval_every_epoch=int(cfg.get('Hook.MeanTeacherEvalHook.eval_every_epoch')), prefix='test_')

    hook_builder(ExtendMLFlowLoggerHook, local_dir_save_ckpt=cfg.get('Hook.ExtendMLFlowLoggerHook.local_dir_save_ckpt'),
                 dagshub_dir_save_ckpt=cfg.get('Hook.ExtendMLFlowLoggerHook.dagshub_dir_save_ckpt'),
                 max_save_epoch_interval=int(cfg.get('Hook.ExtendMLFlowLoggerHook.max_save_epoch_interval')),
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
    # hook_builder(LastSaveHook, save_dir=cfg.get('Hook.LastSaveHook.save_dir', 'ckpt'))
    hook_builder(LoggerHook, logger_file='logs/simple.json')
    # hook_builder(StopTrainAtEpoch, stop_at_epoch=int(cfg.get('Hook.StopTrainAtEpoch.stop_at_epoch')))

    trainer.train()

    criteria = cfg.get('score_criteria')
    for info in reversed(trainer.info_storage.info_storage):
        if criteria in info:
            return info[criteria]
    raise ValueError(f"Criteria {criteria} does not exist in info_storage")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DAv2 Fusion Mean Teacher training with prototype consistency (DAv2Fusion_ResNet34U_f_EMAEncoderOnly).')
    parser.add_argument('--optuna_trial_times', type=int, default=4, help='Optuna trials; 0 = no Optuna.')
    parser.add_argument('--config', type=str, default='cfg/DAv2_AddPrototypeSignal.yaml', help='Path to YAML config')
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

#  !cd /kaggle/working/meanTeacherPolyp && \
#     python DEMT_DAv2_ProtoConsistency.py \
#                     --optuna_trial_times 3\
#                     data.root=/kaggle/input/datasets/akiyanguyen/polypdataset/polypDataset_final1/kvasir_SEG data.data2_dir='Train' \
#                     data.test.dataset_root=/kaggle/input/datasets/akiyanguyen/polypdataset/polypDataset_final1/kvasir_SEG/Test \
#                     data.dataset=kvasir_SEG \
#                     Hook.ExtendMLFlowLoggerHook.run_name='DEMT_DAv2_ProtoConsistency' \
#                     Hook.ExtendMLFlowLoggerHook.experiment_name='DEMT_DAv2_ProtoConsistency' \
#                     Hook.ExtendMLFlowLoggerHook.meta_info.kaggle_run_link='https://www.kaggle.com/code/minhnguyenakiyahere/kagglerunningtemplate/edit?fromFork=1' \
#                     Hook.ExtendMLFlowLoggerHook.meta_info.version=1

#  !cd /kaggle/working/meanTeacherPolyp && \
#     python DAv2_AddPrototypeSignal.py \
#                     --optuna_trial_times 0\
#                     Trainer.prototype_fea_layers=2 \
#                     data.root=/kaggle/input/datasets/akiyanguyen/polypdataset/polypDataset_final1/kvasir_SEG data.data2_dir='Train' \
#                     data.test.dataset_root=/kaggle/input/datasets/akiyanguyen/polypdataset/polypDataset_final1/kvasir_SEG/Test \
#                     data.dataset=kvasir_SEG \
#                     Hook.ExtendMLFlowLoggerHook.run_name='DAv2_AddPrototypeSignal' \
#                     Hook.ExtendMLFlowLoggerHook.experiment_name='DAv2_AddPrototypeSignal' \
#                     Hook.ExtendMLFlowLoggerHook.meta_info.kaggle_run_link='https://www.kaggle.com/code/minhnguyenakiyahere/kagglerunningtemplate/edit?fromFork=1' \
#                     Hook.ExtendMLFlowLoggerHook.meta_info.version=1 \