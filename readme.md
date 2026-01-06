Here is the code for this project.

## Environment

```bash
# Create conda environment for EvolvedGRPO
conda env create -f environment.yaml

# Activate environment
conda activate evolvedgrpo
markdown
```

## Datasets

* **Multi-modal dataset**:
  [https://huggingface.co/datasets/FanqingM/MMK12](https://huggingface.co/datasets/FanqingM/MMK12)

* **Text editing knowledge**:
  [https://huggingface.co/datasets/xl-zhao/PromptCoT-Problem-Generation-Dataset](https://huggingface.co/datasets/xl-zhao/PromptCoT-Problem-Generation-Dataset)

## Question Generation

Generate multiple candidate instructions per round using question generator:

```bash
python3 image.py
python3 text.py
```
## Instruction Execution & Reward

Execute image instructions in */imageedit*.
Execute text instructions with *Qwen2.5-VL-7B*.
Rewards are reused for single-step question generator training with the same training framework.

```bash
cd generator
bash examples/qwen2_5_vl_7b_train.sh
```

## Answer Model Training

Each round after batch data editing, train the answer model:

```bash
cd answer
bash examples/qwen2_5_vl_7b_train.sh
```
