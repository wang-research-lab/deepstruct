
            EXPERIMENT_NAME=${MODEL_TYPE}-cnndm_org
            TASK_NAME=cnn_dm_original
            DATA_PATH="${DATA_ROOT}/${TASK_DATASET}"

TRAIN_ARGS="--epochs 3 \
            --batch-size 4 \
            --lr 1e-5 \
            --lr-decay-style linear \
            --warmup 0.06 \
            --weight-decay 1.0e-1 \
            --label-smoothing 0.1"
COMMON_ARGS="--save-interval 10000 \
            --log-interval 10 \
            --eval-interval 10000 \
            --eval-iters 10000 \
            --eval-epoch 5"
TASK_ARGS="--src-seq-length 512 \
            --tgt-seq-length 512 \
            --min-tgt-length 0 \
            --length-penalty 0.3 \
            --no-repeat-ngram-size 0 \
            --num-beams 8 \
            --select-topk \
            --eval-batch-size 1"
