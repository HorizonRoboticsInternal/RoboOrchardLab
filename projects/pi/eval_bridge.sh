#!/bin/bash

# Modifying the following 3 variables according to your setup
HOBOT_ROOT=/data/weixu/code/Hobot
TMP_CKPT_DIR=/data/weixu/tmp/temp_hobot_checkpoint
WORKSPACE_ROOT=/data/weixu/log/robo_orchard

for exp in exp002; do
    workspace_dir=$WORKSPACE_ROOT/$exp
    for ckpt_step in 1 2 3; do
        python convert_to_hobot_checkpoint.py $workspace_dir $ckpt_step $TMP_CKPT_DIR

        for use_plate in 0 1 ; do
            python $HOBOT_ROOT/hobot2/priors/openpi/scripts/eval_bridge.py \
                --root_dir=$workspace_dir \
                --checkpoint_step=$ckpt_step \
                --use_plate=$use_plate \
                --result_file=$workspace_dir/eval_results_plate${use_plate}_${ckpt_step}.txt \
                --conf_param _CONFIG._USER.ckpt_dir=\"$TMP_CKPT_DIR\" \
                --conf_param _CONFIG._USER.config_file=\"$workspace_dir/config.py\"
        done
    done
done

rm -rf $TMP_CKPT_DIR