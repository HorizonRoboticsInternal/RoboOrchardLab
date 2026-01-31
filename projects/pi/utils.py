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


def load_config(config_file):
    assert config_file.endswith(".py")
    module_name = os.path.split(config_file)[-1][:-3]
    spec = importlib.util.spec_from_file_location(module_name, config_file)
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)
    return config
