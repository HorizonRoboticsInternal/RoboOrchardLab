# Project RoboOrchard
#
# Copyright (c) 2024-2026 Horizon Robotics. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
# implied. See the License for the specific language governing
# permissions and limitations under the License.

import argparse
import dataclasses
from inspect import signature
import json
import logging
import multiprocessing
from multiprocessing import set_start_method
import numpy as np
import os
import pprint
import re
import safetensors.torch
import shutil

import jax
import torch
from accelerate import Accelerator
from accelerate.state import AcceleratorState, is_initialized
from accelerate.utils import (
    DataLoaderConfiguration,
    DistributedDataParallelKwargs,
    ProjectConfiguration
)
from utils import load_config
import alf
import jax.tree
import openpi.models.model as _model
from robo_orchard_lab.dataset.collates import collate_batch_dict
from robo_orchard_lab.pipeline import SimpleTrainer
from robo_orchard_lab.pipeline.batch_processor import SimpleBatchProcessor
from robo_orchard_lab.pipeline.hooks import (
    LossTrackerConfig,
    SaveCheckpointConfig,
    StatsMonitorConfig,
)
from robo_orchard_lab.pipeline.hooks.mixin import (
    HookContext,
    PipelineHookArgs,
    PipelineHooks,
)
from robo_orchard_lab.utils.huggingface import (
    get_accelerate_project_last_checkpoint_id,
)
from robo_orchard_lab.utils import log_basic_config

logger = logging.getLogger(__file__)


def _add_file_logger(path: str, fmt: str, datefmt: str) -> None:
    root_logger = logging.getLogger()
    for handler in root_logger.handlers:
        if isinstance(handler, logging.FileHandler) and handler.baseFilename == path:
            return
    file_handler = logging.FileHandler(path)
    file_handler.setFormatter(logging.Formatter(fmt=fmt, datefmt=datefmt))
    root_logger.addHandler(file_handler)


class MyBatchProcessor(SimpleBatchProcessor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, model, batch):
        losses = model(batch)

        if isinstance(losses, list | tuple):
            losses = {"loss": torch.stack(losses)}
        elif isinstance(losses, torch.Tensor):
            losses = {"loss": losses}
        else:
            assert isinstance(losses, dict), (
                "Model forward must return a tensor or a dict/tuple/list of tensors.")

        losses = jax.tree.map(lambda x: x.mean(), losses)

        loss = losses["loss"]

        losses = {f'train/{k}': v for k, v in losses.items()}

        return losses, loss


class LossAndActionMetric:
    def __init__(self):
        self.reset()
        self._warned_missing_sampler = False

    def reset(self):
        self.loss_results = []
        self.action_results = []

    @staticmethod
    def _mean_over_processes(accelerator: Accelerator, values):
        all_values = [None for _ in range(accelerator.num_processes)]
        if accelerator.num_processes > 1:
            torch.distributed.all_gather_object(all_values, values)
        else:
            all_values[0] = values
        gathered = [x for x in all_values if x is not None]
        if len(gathered) == 0:
            return None
        return jax.tree.map(lambda *x: torch.stack(x, dim=0).mean().item(), *gathered)

    @staticmethod
    def _to_loss_dict(model_outputs):
        losses = model_outputs
        if isinstance(losses, list | tuple):
            losses = {"loss": torch.stack(losses)}
        elif isinstance(losses, torch.Tensor):
            losses = {"loss": losses}
        else:
            assert isinstance(losses, dict), (
                "Model forward must return a tensor or a dict/tuple/list of tensors.")
        return jax.tree.map(lambda x: x.mean().cpu(), losses)

    @staticmethod
    def _build_eval_inputs(batch):
        if "item1" in batch:
            obs_src = batch["item1"]
            gt_actions = batch["item1"]["actions"]
        else:
            obs_src = batch
            gt_actions = batch["actions"]
        observation = _model.Observation.from_dict(obs_src)
        return observation, gt_actions

    def update(self, batch, model_outputs, model=None):
        self.loss_results.append(self._to_loss_dict(model_outputs))

        if model is None or not hasattr(model, "_model") or not hasattr(model._model, "sample_actions"):
            if not self._warned_missing_sampler:
                logger.warning("Skip sampled-action metric because model.sample_actions is unavailable.")
                self._warned_missing_sampler = True
            return

        observation, gt_actions = self._build_eval_inputs(batch)
        device = model.device
        observation = jax.tree.map(lambda x: x.to(device), observation)
        gt_actions = gt_actions.to(device=device, dtype=torch.float32)
        sampled_actions = model._model.sample_actions(device, observation)
        diff = sampled_actions - gt_actions
        self.action_results.append({
            "action_mae": diff.abs().mean().cpu(),
            "action_mse": diff.square().mean().cpu(),
        })

    def compute(self, accelerator, step):
        losses = self._mean_over_processes(
            accelerator,
            None if len(self.loss_results) == 0 else jax.tree.map(
                lambda *x: torch.stack(x, dim=0).mean(), *self.loss_results
            ),
        )
        actions = self._mean_over_processes(
            accelerator,
            None if len(self.action_results) == 0 else jax.tree.map(
                lambda *x: torch.stack(x, dim=0).mean(), *self.action_results
            ),
        )

        metrics = {}
        if losses is not None:
            metrics.update(losses)
        if actions is not None:
            metrics.update(actions)
        if len(metrics) == 0:
            return None

        if accelerator.is_main_process:
            metrics = {f"val/{k}": v for k, v in metrics.items()}
            accelerator.log(metrics, step=step)
            return metrics
        return None


class PiSimpleTrainer(SimpleTrainer):
    @torch.no_grad()
    def eval(self):
        assert self.val_dataloader is not None and self.metric is not None, (
            "val_dataloader and metric should not be None"
        )
        training = self.model.training
        self.model.eval()
        torch.cuda.empty_cache()
        if self.accelerator.is_main_process:
            logger.info("\n" + "=" * 50 + "BEGIN EVAL" + "=" * 50)

        unwrapped_model = self.accelerator.unwrap_model(self.model)
        update_kwargs = {}
        if "model" in signature(self.metric.update).parameters:
            update_kwargs["model"] = unwrapped_model

        for val_step_id, batch in enumerate(self.val_dataloader):
            model_outputs = self.model(batch)
            self.metric.update(batch, model_outputs, **update_kwargs)
            if (val_step_id + 1) % 10 == 0 and self.accelerator.is_main_process:
                logger.info(f"eval: {val_step_id + 1}")

        self.accelerator.wait_for_everyone()
        compute_kwargs = {}
        if "accelerator" in signature(self.metric.compute).parameters:
            compute_kwargs["accelerator"] = self.accelerator
        if "step" in signature(self.metric.compute).parameters:
            compute_kwargs["step"] = self.trainer_progress_state.global_step_id
        metric = self.metric.compute(**compute_kwargs)
        self.accelerator.wait_for_everyone()
        self.metric.reset()
        torch.cuda.empty_cache()
        self.model.train(training)
        return metric


def build_model(config):
    import robo_orchard_lab.models.pi.model as _model
    model = _model.PiModel(
        _model.PiModelConfig(model=config.model))

    if config.pytorch_weight_path is not None:
        logging.info(f"Loading weights from: {config.pytorch_weight_path}")

        model_path = os.path.join(config.pytorch_weight_path, "model.safetensors")
        safetensors.torch.load_model(
            model._model, model_path, strict=config.strict_load)
        logging.info(f"Loaded PyTorch weights from {config.pytorch_weight_path}")

    model._model.paligemma_with_expert.prepare_lora_training(config.vlm_lora_config, config.expert_lora_config)
    config.freeze_torch_parameters(model._model)

    return model


def build_optimizer(config, model):
    # Optimizer + learning rate schedule from config
    warmup_steps = config.lr_schedule.warmup_steps
    peak_lr = config.lr_schedule.peak_lr
    decay_steps = config.lr_schedule.decay_steps
    end_lr = config.lr_schedule.decay_lr

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=peak_lr,
        betas=(config.optimizer.b1, config.optimizer.b2),
        eps=config.optimizer.eps,
        weight_decay=config.optimizer.weight_decay,
    )

    lr_scheduler = torch.optim.lr_scheduler.ChainedScheduler(
        [
            torch.optim.lr_scheduler.LinearLR(
                optimizer, start_factor=0.001, total_iters=warmup_steps
            ),
            torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=decay_steps,
                eta_min=end_lr,
            ),
        ]
    )
    return optimizer, lr_scheduler


def _worker_init_fn(worker_id: int) -> None:
    """Tell JAX inside the worker process not to preallocate the GPU memory."""
    # NOTE: This is called after jax is imported inside the worker process. This
    # means that this approach will not work for selecting the backend.
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"


def _collate_fn(items):
    """Collate the batch elements into batched Tensor."""
    # Make sure to convert to numpy arrays before stacking since some of the incoming elements
    # may be JAX arrays.
    return jax.tree.map(lambda *xs: torch.as_tensor(np.stack([np.asarray(x) for x in xs], axis=0)), *items)


def create_data_loader(config, dataset, accelerator: Accelerator, training):
    global_batch_size = config.batch_size
    assert global_batch_size % accelerator.num_processes == 0, (
        "Global batch size must be divisible by the number of processes.")
    local_batch_size = global_batch_size // accelerator.num_processes
    shuffle = training

    sampler = None
    if torch.distributed.is_initialized():
        sampler = torch.utils.data.distributed.DistributedSampler(
            dataset,
            num_replicas=torch.distributed.get_world_size(),
            rank=torch.distributed.get_rank(),
            shuffle=shuffle,
            drop_last=True,
        )

    mp_context = None
    if config.num_workers > 0:
        mp_context = multiprocessing.get_context("spawn")

    generator = torch.Generator()
    generator.manual_seed(config.seed)

    train_dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=local_batch_size,
        shuffle=(sampler is None and shuffle),  # Don't shuffle if using sampler
        sampler=sampler,
        num_workers=config.num_workers,
        multiprocessing_context=mp_context,
        pin_memory=True,
        collate_fn=_collate_fn,
        worker_init_fn=_worker_init_fn,
        persistent_workers=config.num_workers > 0,
        drop_last=True,
        generator=generator,
    )
    return train_dataloader

class AfterBackwardHook(PipelineHooks):
    def __init__(self):
        super().__init__()
        self.register_hook(
            "on_model_backward",
            HookContext.from_callable(
                after=self._after_backward,
                before=None,
            ),
        )

    def _after_backward(self, hook_args: PipelineHookArgs) -> None:
        # runs immediately after accelerator.backward(loss)
        if hook_args.step_id > 0 or hook_args.epoch_id > 0:
            return
        model = hook_args.model
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is None:
                logging.warning(f"Parameter {name} has no gradient after backward.")


class SaveNormStatsHook(PipelineHooks):
    def __init__(self, *, save_step_freq: int | None, norm_stats_path: str, workspace_root: str):
        super().__init__()
        self.save_step_freq = save_step_freq
        self.norm_stats_path = norm_stats_path
        self.workspace_root = workspace_root
        self.register_hook(
            "on_step",
            HookContext.from_callable(after=self._on_step_end),
        )

    def _find_checkpoint_dir(self, last_id: int) -> str | None:
        checkpoints_dir = os.path.join(self.workspace_root, "checkpoints")
        if last_id < 0 or not os.path.isdir(checkpoints_dir):
            return None
        for entry in os.listdir(checkpoints_dir):
            match = re.search(r"([0-9]+)(?=[^/]*$)", entry)
            if match and int(match.group(1)) == last_id:
                return os.path.join(checkpoints_dir, entry)
        return None

    def _on_step_end(self, args: PipelineHookArgs) -> None:
        if self.save_step_freq is None:
            return
        if (args.global_step_id + 1) % self.save_step_freq != 0:
            return
        if not os.path.isfile(self.norm_stats_path):
            if args.accelerator.is_main_process:
                logger.warning("norm_stats.json not found at %s", self.norm_stats_path)
            return

        last_id = get_accelerate_project_last_checkpoint_id(self.workspace_root)
        ckpt_dir = self._find_checkpoint_dir(last_id)
        if ckpt_dir is None:
            if args.accelerator.is_main_process:
                logger.warning("No checkpoint dir found for id %s", last_id)
            return

        model_dir = os.path.join(ckpt_dir, "model")
        if not os.path.isdir(model_dir):
            if args.accelerator.is_main_process:
                logger.warning("Model dir not found: %s", model_dir)
            return

        dst = os.path.join(model_dir, "norm_stats.json")
        if args.accelerator.is_main_process:
            os.makedirs(model_dir, exist_ok=True)
            shutil.copy2(self.norm_stats_path, dst)
            logger.info("Copied norm_stats.json to %s", dst)


def get_resume_from(args, config):
    if not config.resume:
        return None

    checkpoints_dir = os.path.join(args.workspace, "checkpoints")
    last_id = get_accelerate_project_last_checkpoint_id(args.workspace)
    if last_id < 0:
        raise FileNotFoundError(
            f"No accelerate checkpoints found under {checkpoints_dir}"
        )
    checkpoint_name = None
    for entry in os.listdir(checkpoints_dir):
        match = re.search(r"([0-9]+)(?=[^/]*$)", entry)
        if match and int(match.group(1)) == last_id:
            checkpoint_name = entry
            break
    if checkpoint_name is None:
        raise FileNotFoundError(
            f"Unable to locate checkpoint {last_id} under {checkpoints_dir}"
        )
    return os.path.join(checkpoints_dir, checkpoint_name)

def main(args, config, accelerator):
    build_dataset = config.build_dataset
    config = config.config

    if args.kwargs is not None:
        if os.path.isfile(args.kwargs):
            kwargs = json.load(open(args.kwargs, "r"))
        else:
            kwargs = json.loads(args.kwargs)
        config = dataclasses.replace(config, **kwargs)

    if accelerator.is_main_process:
        logger.info("\n" + pprint.pformat(config))

    train_dataset = build_dataset(config)
    train_dataloader = create_data_loader(config, train_dataset, accelerator, training=True)
    val_dataset = build_dataset(config, split="val")
    val_dataloader = create_data_loader(config, val_dataset, accelerator, training=False)
    model = build_model(config)

    optimizer, lr_scheduler = build_optimizer(config, model)
    data_config = config.data.create(config.assets_dirs, config.model)
    assets_dir = config.data.assets.assets_dir or str(config.assets_dirs)
    norm_stats_path = None
    if data_config.asset_id is not None:
        norm_stats_path = os.path.join(assets_dir, data_config.asset_id, "norm_stats.json")
    resume_from = get_resume_from(args, config)
    trainer = PiSimpleTrainer(
        model=model,
        dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        metric=LossAndActionMetric(),
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        accelerator=accelerator,
        grad_clip_mode="norm",
        grad_max_norm=10,
        batch_processor=MyBatchProcessor(need_backward=True),
        hooks=[
            StatsMonitorConfig(
                step_log_freq=config.log_interval,
            ),
            LossTrackerConfig(
                step_log_freq=config.log_interval
            ),
            SaveCheckpointConfig(
                save_step_freq=config.save_interval,
                save_epoch_freq=None,
                save_model=False,
                save_when_loop_end=False,
            ),
            AfterBackwardHook(),
            SaveNormStatsHook(
                save_step_freq=config.save_interval,
                norm_stats_path=norm_stats_path,
                workspace_root=args.workspace,
            ) if norm_stats_path is not None else PipelineHooks(),
        ],
        max_step=config.num_train_steps,
        step_eval_freq=500,
        lr_scheduler_step_at="step",
        resume_from=resume_from,
    )

    trainer()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=str, default="./workspace")
    parser.add_argument(
        "--config", type=str, default="./config_sem_robotwin.py"
    )
    parser.add_argument("--kwargs", type=str, default=None)
    args = parser.parse_args()

    workspace_root = args.workspace
    os.makedirs(workspace_root, exist_ok=True)
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        log_with="tensorboard",
        step_scheduler_with_optimizer=False,
        project_config=ProjectConfiguration(
            project_dir=workspace_root,
            logging_dir=os.path.join(workspace_root, "logs"),
            automatic_checkpoint_naming=True,
            total_limit=None,
        ),
        dataloader_config=DataLoaderConfiguration(
            use_seedable_sampler=True,
        ),
        kwargs_handlers=[ddp_kwargs],
    )
    accelerator.init_trackers("tensorboard")
    log_fmt = "%rank %(asctime)s %(levelname)s %(filename)s:%(lineno)d | %(message)s"
    log_datefmt = "%m/%d/%Y %H:%M:%S"
    log_basic_config(format=log_fmt, datefmt=log_datefmt, level=logging.INFO)
    os.makedirs(os.path.join(workspace_root, "logs"), exist_ok=True)
    log_path = os.path.join(workspace_root, "logs", f"train_rank{accelerator.process_index}.log")
    _add_file_logger(log_path, log_fmt, log_datefmt)
    logger.info(f"if accelerator initialized:{is_initialized()}")
    logger.info(f"accelerator state: {AcceleratorState._shared_state}")
    set_start_method("spawn", force=True)
    config = load_config(args.config)
    shutil.copy(args.config, os.path.join(workspace_root, 'config.py'))
    with alf.module.original_torch_module_functions():
        main(args, config, accelerator)
