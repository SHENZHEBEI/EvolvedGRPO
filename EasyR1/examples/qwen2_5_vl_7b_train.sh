#!/bin/bash
set -x

input_name="final_step4_fuxian"
output_name="final_step5_fuxian"

MODEL_PATH="/home2/yqf/new_dataset/nips/EasyR1/${input_name}/global_step_30/actor/huggingface/"
DATASET="/home2/yqf/new_dataset/nips/dataset/train_original"
TEST_DATASET="/home2/yqf/new_dataset/nips/dataset/validation"

# Keep config.yaml's global step size unchanged:
# 512 prompts per rollout step, 5 responses per prompt, 2560 trajectories in total.
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"

python3 -m verl.trainer.main \
    config=examples/config.yaml \
    data.train_files=${DATASET} \
    data.val_files=${TEST_DATASET} \
    worker.actor.model.model_path=${MODEL_PATH} \
    worker.rollout.tensor_parallel_size=1 \
    trainer.experiment_name=${output_name} \
    trainer.n_gpus_per_node=8 \
    trainer.load_checkpoint_path=null \
    trainer.save_checkpoint_path=${output_name}

python3 scripts/model_merger.py --local_dir "${output_name}/global_step_30/actor"

CHECKPOINT_DIR="${output_name}/global_step_30/actor"
for rank in 0 1 2 3 4 5 6 7; do
    rm -f "${CHECKPOINT_DIR}/extra_state_world_size_8_rank_${rank}.pt"
    rm -f "${CHECKPOINT_DIR}/model_world_size_8_rank_${rank}.pt"
    rm -f "${CHECKPOINT_DIR}/optim_world_size_8_rank_${rank}.pt"
done
