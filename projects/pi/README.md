# Installation

Under RoboOrchardLab repo root

```bash
uv sync --group openpi
uv pip install -e . -e submodules/openpi
cp -r submodules/openpi/src/openpi/models_pytorch/transformers_replace/* .venv/lib/python*/site-packages/transformers/
```

# train

## train with single-gpu
```bash
python train.py --config config_pi_libero.py --workspace /data/weixu/log/robo_orchard/exp000
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