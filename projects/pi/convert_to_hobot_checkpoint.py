import safetensors.torch
import shutil
import click
import os

@click.command()
@click.argument("root_dir")
@click.argument("ckpt_id", type=int)
@click.argument("new_ckpt_dir")
def main(root_dir: str, ckpt_id: int, new_ckpt_dir: str):
    os.makedirs(new_ckpt_dir, exist_ok=True)
    ckpt_path = f"{root_dir}/checkpoints/checkpoint_{ckpt_id}/model.safetensors"
    state_dict = safetensors.torch.load_file(ckpt_path)
    new_state_dict = {k[len("_model."):]: v for k, v in state_dict.items()}
    safetensors.torch.save_file(new_state_dict, f"{new_ckpt_dir}/model.safetensors")
    shutil.copytree(f"/data/weixu/code/Hobot/hobot2/priors/openpi/assets", f"{new_ckpt_dir}/assets", dirs_exist_ok=True)


if __name__ == "__main__":
    main()
