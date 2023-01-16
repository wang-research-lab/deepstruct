source ../PATH.sh
DATA_ROOT=$ROOT/data
CHECKPOINT_PATH=$CHECKPOINTROOT
SAVE_PATH=$ROOT/finetune_checkpoints
DATESTR=$(date +"%m-%d-%H-%M")

source $1    # Model

TASK_DATASET=$3
source $2    # Task

NUM_WORKERS=12
NUM_GPUS_PER_WORKER=8
HOST_FILE_PATH="./hostfile_multiserver"
MP_SIZE=1
MASTER_PORT=$(shuf -n 1 -i 10000-65535)

OPTIONS_NCCL="NCCL_DEBUG=info NCCL_IB_DISABLE=0 NCCL_NET_GDR_LEVEL=2"
DISTRIBUTED_ARGS="${OPTIONS_NCCL} deepspeed --hostfile ${HOST_FILE_PATH} --master_port ${MASTER_PORT} --num_nodes ${NUM_WORKERS} --num_gpus ${NUM_GPUS_PER_WORKER}"

EXPERIMENT_NAME=${EXPERIMENT_NAME}_${DATESTR}
mkdir logs
run_cmd="${DISTRIBUTED_ARGS} finetune_glm.py \
       --deepspeed \
       --deepspeed_config config_tasks/config_multiserver.json \
       --finetune \
       --experiment-name ${EXPERIMENT_NAME} \
       --task ${TASK_NAME} \
       --task-dataset ${TASK_DATASET} \
       --data-dir ${DATA_PATH} \
       --save ${CHECKPOINT_PATH} \
       --checkpoint-activations \
       --num-workers ${NUM_WORKERS} \
       --no-load-lr-scheduler \
       $MODEL_ARGS \
       $TRAIN_ARGS \
       $COMMON_ARGS \
       $TASK_ARGS \
       --fp16 \
       --model-parallel-size ${MP_SIZE} \
       --overwrite \
       2>&1 | tee logs/log-${EXPERIMENT_NAME}.txt"

echo $EXPERIMENT_NAME > "runs/latest_run"

echo ${run_cmd}
eval ${run_cmd}

