#!/bin/bash
set -x

input_name=""
output_name=""

MODEL_PATH=""
DATASET=""
TEST_DATASET=""

CUDA_VISIBLE_DEVICES="1,2,3,4" python3 -m verl.trainer.main \
    config=examples/config.yaml \
    data.train_files=${DATASET} \
    data.val_files=${TEST_DATASET} \
    worker.actor.model.model_path=${MODEL_PATH} \
    worker.rollout.tensor_parallel_size=1 \
    trainer.experiment_name=${output_name} \
    trainer.n_gpus_per_node=4 \
    trainer.load_checkpoint_path=null \
    trainer.save_checkpoint_path=${output_name}

python3 scripts/model_merger.py --local_dir "${output_name}/global_step_30/actor"
