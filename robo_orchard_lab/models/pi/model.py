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

import jax
import openpi.models.model as _model
from robo_orchard_lab.models.mixin import (
    ClassType_co,
    ModelMixin,
    TorchModuleCfg,
)

class PiModel(ModelMixin):
    cfg: "PiModelConfig"  # for type hint

    def __init__(self, cfg: "PiModelConfig"):
        super().__init__(cfg)
        # Model implementation would go here.
        self._model = cfg.model.create_pytorch()

    def forward(self, batch):
        device = self.device
        if "item1" in batch:
            observation1 = _model.Observation.from_dict(batch["item1"])
            actions1 = batch["item1"]["actions"]
            observation2 = _model.Observation.from_dict(batch["item2"])
            batch = {
                "observation1": observation1,
                "actions1": actions1,
                "observation2": observation2,
                "item2_is_pad": batch["item2_is_pad"],
                "horizon": batch["horizon"],
                "steps_to_go": batch["steps_to_go"],
            }
        else:
            obs = _model.Observation.from_dict(batch)
            batch = obs, batch["actions"]

        batch = jax.tree.map(lambda x: x.to(device), batch)
        return self._model(batch)

class PiModelConfig(TorchModuleCfg[PiModel]):
    class_type: ClassType_co[PiModel] = PiModel
    model: _model.BaseModelConfig
