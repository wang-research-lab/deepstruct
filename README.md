# DeepStruct: Pretraining of Language Models for Structure Prediction

Source code repo for paper [DeepStruct: Pretraining of Language Models for Structure Prediction](https://arxiv.org/abs/2205.10475), ACL 2022.


## Setup Environment

DeepStruct is based on [GLM](https://github.com/THUDM/GLM) dependency. Please use GLM's docker as follow to setup the basic GPU environment (`zxdu20/glm-cuda112` for Ampere GPUs and `zxdu20/glm-cuda102` for older version GPUs such as Tesla V100).

```bash
git clone --recursive git@github.com:cgraywang/deepstruct.git
cd ./deepstruct

docker run --net=host --privileged --pid=host --gpus all --rm -it --ipc=host -v ./deepstruct:/workspace/deepstruct zxdu20/glm-cuda112
cd /workspace/deepstruct
```

and install the dependency via `setup.sh`:

```bash
bash setup.sh
```

The final directory structure should be as follows:

```
workspace/
├─ deepstruct/
├─ data/
├─ ckpt/
```

## Download Checkpoints

Most of our experiments are based on 10-billion-parameter DeepStruct checkpoint. Run the following shell scripts to download all multi-task trained DeepStruct checkpoints from huggingface hub (might take a while).

```bash
bash download_ckpt.sh
```

## Data Preparation & Reproduce

To run following experiments on DeepStruct-10B, our experiments adopt `batch_size_per_gpu=1` and require at least 32 GB GPU memory to run.
The scripts default use `--num-gpus-per-node=1` in `src/tasks/mt/*.sh`, and if you want to use multiple gpu for acceleration, please customize it in `src/tasks/mt/*.sh`.

Notice that `CoNLL12`, `CoNLL05` for semantic role labeling, `ACE2005` for event extraction require manual download from LDC ([LDC2006T06](https://catalog.ldc.upenn.edu/LDC2006T06), [LDC2013T19](https://catalog.ldc.upenn.edu/LDC2013T19), [PTB-3](https://catalog.ldc.upenn.edu/LDC99T42)).

| Task                                 | Dataset       | Data preparation                                           | Multi-task Result                                   |
|--------------------------------------|---------------|------------------------------------------------------------|-----------------------------------------------------|
| Joint entity and relation extraction | CoNLL04       | `bash run_scripts/conll04.sh`                              | Ent. 88.4/Rel. 72.8                                 |
| Joint entity and relation extraction | ADE           | `bash run_scripts/ade.sh`                                  | Ent. 90.5/Rel. 83.6                                 |
| Joint entity and relation extraction | NYT           | `bash run_scripts/nyt.sh`                                  | Ent. 95.4/Rel. 93.7                                 |
| Joint entity and relation extraction | ACE2005       | `bash run_scripts/ace2005_jer.sh <abs_path_to_LDC2006T06>` | Ent. 90.2/Rel. 58.9                                 |
| Semantic role labeling               | CoNLL05 WSJ   | `bash run_scripts/conll05_srl_wsj.sh <abs_path_to_PTB_3>`  | 95.5                                                |
| Semantic role labeling               | CoNLL05 Brown | `bash run_scripts/conll05_srl_brown.sh <abs_path_to_PTB_3>`| 92.0                                                |
| Semantic role labeling               | CoNLL12       | `bash run_scripts/conll12_srl.sh <abs_path_to_LDC2013T19>` | 97.2                                                |
| Event extraction                     | ACE2005         | `bash run_scripts/ace2005event.sh <abs_path_to_LDC2006T06>`| Trigger: Id-72.7/Cl-69.2  Argument: Id-67.5/Cl-63.9 |
| Intent detection                     | ATIS          | `bash run_scripts/atis.sh`                                 | 97.3                                                |
| Intent detection                     | SNIPS         | `bash run_scripts/snips.sh`                                | 97.4                                                |
| Dialogue state tracking              | MultiWOZ 2.1 | `bash run_scripts/multi_woz.sh`                            | 53.5                                                |

## Arguments in running scripts
The arguments in `src/tasks/mt/*.sh` configure the training and inference of DeepStruct. Here are their meanings:

* `--model-type`: the type of model backbone to use. Currently we only support `model_blocklm_10B`, which means using the 10-billion DeepStruct model as the backbone.
* `--model-checkpoint`: the path to the directory of DeepStruct checkpoint.
* `--task`: the task being trained or inferenced.
* `--task-epochs`: number of epochs to run. If set to `0`, it means evaluation only.
* `--length-penalty`: a hyperparameter to configure the lengths of generated sequences in the beam search.


## Scripts for Pretraining

Following the commands below to prepare pretraining data and run training.

```bash
# prepare pretraining data
bash data_scripts/PRETRAIN.sh

# run pretraining
cd ./glm/
bash scripts/ds_finetune_seq2seq_pretrain.sh config_tasks/<MODEL_TYPE>.sh config_tasks/pretrain.sh cnn_dm_original
```

Currently `<MODEL_TYPE>` supports `model_blocklm_10B_pretrain`, which refers to the 10 billion pretrained model as backbone.

Please customize `NUM_GPUS_PER_WORKER` in `glm/scripts/ds_finetune_seq2seq_pretrain.sh` and `train_micro_batch_size_per_gpu` in `glm/config_tasks/config.json` according to your environment, as fine-tuning a 10B language model requires quite sufficient GPU memory.
The data preprocessing for pretraining may require over 600G main memory, as the current dataloader implementation preloads all tokenized data into main memory in pretraining.

## Citation

```bibtex
@inproceedings{wang-etal-2022-deepstruct,
    title = "{D}eep{S}truct: Pretraining of Language Models for Structure Prediction",
    author = "Wang, Chenguang  and
      Liu, Xiao  and
      Chen, Zui  and
      Hong, Haoyun  and
      Tang, Jie  and
      Song, Dawn",
    booktitle = "Findings of the Association for Computational Linguistics: ACL 2022",
    year = "2022",
}
```
