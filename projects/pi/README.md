# Installation

Under RoboOrchardLab repo root

```bash
uv sync --group openpi
uv pip install -e . -e ../Hobot
cp -r ../Hobot/submodules/openpi/src/openpi/models_pytorch/transformers_replace/* .venv/lib/python*/site-packages/transformers/
```

# train

## train with single-gpu
```bash
uv run train.py --config config_pi_libero.py \
--workspace /data/weixu/log/robo_orchard/exp000 \
--kwargs '{"batch_size": 32, "freeze_vlm": 1}'

```


## train with single machine multi-gpu
```bash
accelerate launch \
    --num-processes 2  \
    --multi-gpu \
    train.py \
    --workspace /data/weixu/log/robo_orchard/exp000
    --config config_pi_libero.py
```