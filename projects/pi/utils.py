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

import importlib
import os
import jax
import torch


def load_config(config_file):
    assert config_file.endswith(".py")
    module_name = os.path.split(config_file)[-1][:-3]
    spec = importlib.util.spec_from_file_location(module_name, config_file)
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)
    return config


class LossMetric:
    def __init__(self):
        self.reset()

    def reset(self):
        self.results = []

    def compute(self, accelerator, step):
        if len(self.results) == 0:
            losses = None
        else:
            losses = jax.tree.map(lambda *x: torch.stack(x, dim=0).mean(), *self.results)
        all_losses = [None for _ in range(accelerator.num_processes)]
        if accelerator.num_processes > 1:
            torch.distributed.all_gather_object(all_losses, losses)
        else:
            all_losses[0] = losses
        losses = [x for x in all_losses if x is not None]
        if len(losses) == 0:
            return None
        losses = jax.tree.map(lambda *x: torch.stack(x, dim=0).mean().item(), *losses)
        if accelerator.is_main_process:
            losses = {f'val/{k}': v for k, v in losses.items()}
            accelerator.log(losses, step=step)
            return losses
        else:
            return None

    def update(self, batch, model_outputs):
        losses = model_outputs
        if isinstance(losses, list | tuple):
            losses = {"loss": torch.stack(losses)}
        elif isinstance(losses, torch.Tensor):
            losses = {"loss": losses}
        else:
            assert isinstance(losses, dict), (
                "Model forward must return a tensor or a dict/tuple/list of tensors.")

        losses = jax.tree.map(lambda x: x.mean().cpu(), losses)
        self.results.append(losses)
