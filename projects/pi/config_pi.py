# Copyright (c) 2025 Horizon Robotics and Hobot Contributors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Config file for sl_train.py."""

from functools import partial
import logging
import alf

import openpi.models.pi0_config as pi0_config
from openpi.training.config import DataConfig, AssetsConfig, TrainConfig
import openpi.training.optimizer as _optimizer
import openpi.shared.nnx_utils as nnx_utils

from hobot2.priors.openpi.configs import (
    LeRobotBridgeDataConfig,
    LeRobotLiberoDataConfig,
    fixed_horizon_sampler,
    ASSETS_DIR,
)
import openpi.training.data_loader as openpi_data_loader
import hobot2.priors.openpi.data_loader as hobot_data_loader

from hobot2.priors.openpi.models.model_config import JointModelConfig
import hobot2.priors.openpi.weight_loaders as _weight_loaders

data_name = alf.define_config("data_name", "bridge_unambig")
add_ren_openpi0_randomizations = alf.define_config(
    "add_ren_openpi0_randomizations", True)

if add_ren_openpi0_randomizations:
    import openpi.models_pytorch.preprocessing_pytorch as _openpi_preprocessing
    import hobot2.priors.openpi.preprocessing_pytorch_ren_aug as _ren_aug

    _openpi_preprocessing.preprocess_observation_pytorch = partial(
        _ren_aug.preprocess_observation_pytorch,
        # ren_openpi0 defaults (Bridge primary camera).
        crop_scale=_ren_aug.DEFAULT_CROP_SCALE,
        crop_ratio=_ren_aug.DEFAULT_CROP_RATIO,
        # OpenPI pi05 defaults.
        rotate_degrees=_ren_aug.DEFAULT_ROTATE_DEGREES,
        brightness_factor=_ren_aug.DEFAULT_BRIGHTNESS_FACTOR,
        contrast_factor=_ren_aug.DEFAULT_CONTRAST_FACTOR,
        saturation_factor=_ren_aug.DEFAULT_SATURATION_FACTOR,
        # ren_openpi0 defaults.
        hue_delta=_ren_aug.DEFAULT_HUE_DELTA,
    )
    logging.getLogger(__name__).info(
        "Enabled ren+pi05 randomizations for OpenPI PyTorch preprocessing.")

asset_id = None
if data_name == "libero":
    data_config_ctor = LeRobotLiberoDataConfig
    repo_id = "physical-intelligence/libero"
    # asset_id = repo_id + "-extra_delta_transform"
    max_token_len = 32
elif data_name.startswith("bridge"):
    data_config_ctor = LeRobotBridgeDataConfig
    # This is the dataset_name passed to
    # hobot2.priors.openpi.data_loader.OpenXDataset.
    repo_id = data_name
    use_relabeled_action = alf.define_config("use_relabeled_action", True)
    alf.config("data_loader.OpenXDataset",
               cache_episodes=True,
               use_relabeled_action=use_relabeled_action)
    openpi_data_loader.create_torch_dataset = (
        hobot_data_loader.create_torch_dataset)
    asset_id = repo_id
    if use_relabeled_action:
        asset_id += "_relabeled"
    max_token_len = 128
else:
    raise ValueError(f"Unknown data_name: {data_name}")

model_type = alf.define_config("model_type", "joint_model")
assert model_type in ("pi05", "joint_model")

dataset_wrappers = []

if model_type == "joint_model":
    model_config_ctor = JointModelConfig
    model_cfgs = dict(
        steps_to_go_loss_weight=0.1,
        repr_loss_weight=0.1,
        separate_repr_token=True,
        expert_attend_to_repr_token=False,
        expert_attend_to_pred_token=False,
    )
    dataset_wrappers = [
        partial(hobot_data_loader.PredictionDataset,
                horizon_sampler=partial(fixed_horizon_sampler, horizon=10))
    ]
    data_cfgs = dict(
        data_loader_wrapper_ctor=hobot_data_loader.PredictionDataLoaderWrapper)
else:
    model_config_ctor = pi0_config.Pi0Config
    dataset_wrapper_ctor = lambda ds: ds
    model_cfgs = {}
    data_cfgs = {}

if data_name.startswith("bridge"):
    dataset_wrappers = dataset_wrappers + [hobot_data_loader.ShardedDataset]

data_cfgs['dataset_wrapper_ctor'] = partial(
    hobot_data_loader.chain_dataset_wrappers, wrappers=dataset_wrappers)

LAST_LAYER_INDEX = 17

base_model_type = "pi05"
assert base_model_type in ("pi0", "pi05")
base_model_path = "/data/cache/openpi/openpi-assets/pytorch_checkpoints/" + base_model_type + "_base"

config = TrainConfig(
    name="openpi",
    exp_name="exp",
    model=model_config_ctor(
        pi05=base_model_type == "pi05",
        action_horizon=10,
        # discrete_state_input is default to True for pi05.
        # But openpi official repo uses False for pi05_libero
        discrete_state_input=data_name != "libero",
        max_token_len=max_token_len,
        skip_unused_image_slot=True,
        **model_cfgs),
    data=data_config_ctor(
        # Note: if asset_id is provided for AssetsConfig, the normalization
        # stats is stored under {assets_dir}/{assets_id}. Otherwise, it is stored
        # under {assets_dir}/{repo_id}.
        # If you want to try different transforms, you should explicitly provide
        # a different asset_id here to use different normalization.
        assets=AssetsConfig(assets_dir=ASSETS_DIR, asset_id=asset_id),
        repo_id=repo_id,
        base_config=DataConfig(prompt_from_task=True, **data_cfgs),
    ),
    batch_size=alf.define_config("global_batch_size", 1024),
    lr_schedule=_optimizer.CosineDecaySchedule(
        warmup_steps=200,
        peak_lr=5e-5,
        decay_steps=1_000_000,
        decay_lr=5e-5,
    ),
    assets_base_dir=ASSETS_DIR,
    optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
    ema_decay=0.999,
    # weight_loader=_weight_loaders.PytorchPaliGemmaWeightLoader(),
    pytorch_weight_path=base_model_path,
    strict_load=model_type != "joint_model",
    num_train_steps=20_000,
    save_interval=5_000,
    log_interval=20,
    seed=42,
    wandb_enabled=True,
    freeze_filter=[
        nnx_utils.PathRegex(".*/embed_tokens/weight"),
        nnx_utils.PathRegex(".*/lm_head.weight"),
        # nnx_utils.PathRegex(f".*language_model/.*/{LAST_LAYER_INDEX}/post.*"),
        # nnx_utils.PathRegex(f".*language_model/.*/{LAST_LAYER_INDEX}/mlp.*"),
        # nnx_utils.PathRegex(
        #     f".*language_model/.*/{LAST_LAYER_INDEX}/self_attn/o_proj.*"),
        # nnx_utils.PathRegex(
        #     f".*language_model/.*/{LAST_LAYER_INDEX}/self_attn/v_proj.*"),
    ],
    freeze_vlm=False,
)

def build_dataset(config, split="train"):
    data_config = config.data.create(config.assets_dirs, config.model)
    action_horizon=config.model.action_horizon
    model_config=config.model
    dataset = openpi_data_loader.create_torch_dataset(data_config, action_horizon, model_config, split=split)
    dataset = openpi_data_loader.transform_dataset(dataset, data_config, skip_norm_stats=False)
    if data_config.dataset_wrapper_ctor is not None:
        dataset = data_config.dataset_wrapper_ctor(dataset)

    return dataset

