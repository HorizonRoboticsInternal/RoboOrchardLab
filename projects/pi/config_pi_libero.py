# Project RoboOrchard
#
# Copyright (c) 2024-2025 Horizon Robotics. All Rights Reserved.
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
import pathlib
from peft import LoraConfig

import openpi.models.pi0_config as pi0_config
import openpi.shared.nnx_utils as nnx_utils
from openpi.training.config import AssetsConfig, DataConfig, LeRobotLiberoDataConfig, TrainConfig
import openpi.training.optimizer as _optimizer
import openpi.training.data_loader as _data_loader

LAST_LAYER_INDEX = 17

ASSETS_DIR = str(pathlib.Path(__file__).parent.absolute() / "assets")
asset_id = None

config = TrainConfig(
    name="openpi",
    exp_name="exp",
    model=pi0_config.Pi0Config(
        pi05=True,
        action_horizon=10,
        # discrete_state_input is default to True for pi05.
        # But openpi official repo uses False for pi05_libero
        discrete_state_input=False,
        max_token_len=32,
        skip_unused_image_slot=True),
    data=LeRobotLiberoDataConfig(
        # Note: if asset_id is provided for AssetsConfig, the normalization
        # stats is stored under {assets_dir}/{assets_id}. Otherwise, it is stored
        # under {assets_dir}/{repo_id}.
        # If you want to try different transforms, you should explicitly provide
        # a different asset_id here to use different normalization.
        assets=AssetsConfig(assets_dir=ASSETS_DIR, asset_id=asset_id),
        repo_id="physical-intelligence/libero",
        base_config=DataConfig(prompt_from_task=True),
    ),
    batch_size=256,
    lr_schedule=_optimizer.CosineDecaySchedule(
        warmup_steps=200,
        peak_lr=5e-5,
        decay_steps=1_000_000,
        decay_lr=5e-5,
    ),
    assets_base_dir=ASSETS_DIR,
    optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
    ema_decay=0.999,
    pytorch_weight_path=
    "/data/cache/openpi/openpi-assets/pytorch_checkpoints/pi05_base",
    strict_load=True,
    num_train_steps=30_000,
    save_interval=15_000,
    log_interval=20,
    seed=2,
    wandb_enabled=True,
    # vlm_lora_config=LoraConfig(
    #     r=32,
    #     lora_alpha=4,  # same as r
    #     lora_dropout=0.0,
    #     init_lora_weights="gaussian",
    #     bias="none",
    #     task_type="CAUSAL_LM",
    #     target_modules="all-linear",
    # ),
    # expert_lora_config=LoraConfig(
    #     r=32,
    #     lora_alpha=4,  # same as r
    #     lora_dropout=0.0,
    #     init_lora_weights="gaussian",
    #     bias="none",
    #     task_type="CAUSAL_LM",
    #     target_modules="all-linear",
    # ),
    # freeze_filter=[
    #     nnx_utils.PathRegex(".*/embed_tokens/weight"),
    #     nnx_utils.PathRegex(".*/lm_head.weight"),
    #     nnx_utils.PathRegex(f".*language_model/.*/{LAST_LAYER_INDEX}/post.*"),
    #     nnx_utils.PathRegex(f".*language_model/.*/{LAST_LAYER_INDEX}/mlp.*"),
    #     nnx_utils.PathRegex(
    #         f".*language_model/.*/{LAST_LAYER_INDEX}/self_attn/o_proj.*"),
    #     nnx_utils.PathRegex(
    #         f".*language_model/.*/{LAST_LAYER_INDEX}/self_attn/v_proj.*"),
    # ],
)


def build_dataset(config):
    data_config = config.data.create(config.assets_dirs, config.model)
    action_horizon=config.model.action_horizon
    model_config=config.model
    dataset = _data_loader.create_torch_dataset(data_config, action_horizon, model_config)
    dataset = _data_loader.transform_dataset(dataset, data_config, skip_norm_stats=False)
    if data_config.dataset_wrapper_ctor is not None:
        dataset = data_config.dataset_wrapper_ctor(dataset)

    return dataset

