from pyheaven import *
import subprocess
import sys

supported_tasks = [
    "ace2005_joint_er",
    "ace2005_joint_er_re",
    "ace2005event_trigger",
    "ace2005event_argument",
    "ade",
    "ade0",
    "ade1",
    "ade2",
    "ade3",
    "ade4",
    "ade5",
    "ade6",
    "ade7",
    "ade8",
    "ade9",
    "ade_re",
    "ade_re0",
    "ade_re1",
    "ade_re2",
    "ade_re3",
    "ade_re4",
    "ade_re5",
    "ade_re6",
    "ade_re7",
    "ade_re8",
    "ade_re9",
    "atis",
    "conll04",
    "conll04_re",
    "conll05_srl_brown",
    "conll05_srl_wsj",
    "conll12_srl",
    "multi_woz",
    "nyt",
    "nyt_re",
    "snips"
]
supported_model_args = {
    "model_blocklm_110M":
        """\"--block-lm \\
                    --cloze-eval \\
                    --task-mask \\
                    --num-layers 12 \\
                    --hidden-size 768 \\
                    --num-attention-heads 12 \\
                    --max-position-embeddings 1024 \\
                    --tokenizer-type BertWordPieceTokenizer \\
                    --load-pretrained {0}\"""",
    "model_blocklm_220M":
        """\"--block-lm \\
                    --cloze-eval \\
                    --task-mask \\
                    --num-layers 14 \\
                    --hidden-size 1024 \\
                    --num-attention-heads 16 \\
                    --max-position-embeddings 1024 \\
                    --tokenizer-model-type gpt2 \\
                    --tokenizer-type GPT2BPETokenizer \\
                    --load-pretrained {0}\"""",
    "model_blocklm_2B":
        """\"--block-lm \\
                    --cloze-eval \\
                    --task-mask \\
                    --num-layers 36 \\
                    --hidden-size 2048 \\
                    --num-attention-heads 32 \\
                    --max-position-embeddings 1024 \\
                    --tokenizer-model-type gpt2 \\
                    --tokenizer-type GPT2BPETokenizer \\
                    --load-pretrained {0}\"""",
    "model_blocklm_10B":
        """\"--block-lm \\
                    --cloze-eval \\
                    --task-mask \\
                    --num-layers 48 \\
                    --hidden-size 4096 \\
                    --num-attention-heads 64 \\
                    --max-position-embeddings 1024 \\
                    --tokenizer-model-type gpt2 \\
                    --tokenizer-type GPT2BPETokenizer \\
                    --load-pretrained {0}\"""",
}
supported_modes = [
    "default",
    "multi",
    "empha",
    "task",
]
train_args = """\"--epochs {0} \\
            --batch-size 4 \\
            --lr 1e-5 \\
            --lr-decay-style linear \\
            --warmup 0.06 \\
            --weight-decay 1.0e-1 \\
            --label-smoothing 0.1\""""
common_args = """\"--save-interval 10000 \\
            --log-interval 10 \\
            --eval-interval 10000 \\
            --eval-iters 10000 \\
            --eval-epoch 5\""""
task_args = """\"--src-seq-length {2} \\
            --tgt-seq-length {3} \\
            --min-tgt-length {4} \\
            --length-penalty {1} \\
            --no-repeat-ngram-size 0 \\
            --num-beams {5} \\
            --select-topk \\
            --eval-batch-size 1\""""

if __name__ == "__main__":
    args = HeavenArguments.from_parser([
        SwitchArgumentDescriptor("multi-server", short="ms"),

        StrArgumentDescriptor("model-checkpoint", short="ckpt", default=None),
        LiteralArgumentDescriptor("model-type", short="m", choices=list(supported_model_args.keys()) + ["auto"],
                                  default="auto"),
        LiteralArgumentDescriptor("mode", short="md", choices=supported_modes, default="multi"),
        SwitchArgumentDescriptor("zero-shot", short="zs"),

        LiteralArgumentDescriptor("task", short="t", choices=supported_tasks, default=None),
        IntArgumentDescriptor("task-epochs", short="e", default=50),
        IntArgumentDescriptor("num-beams", short="b", default=8),
        IntArgumentDescriptor("src-seq-length", short="srcl", default=512),
        IntArgumentDescriptor("tgt-seq-length", short="tgtl", default=512),
        IntArgumentDescriptor("min-tgt-length", short="tgtm", default=0),
        FloatArgumentDescriptor("length-penalty", short="lp", default=0.8),
        IntArgumentDescriptor("num-gpus-per-node", short="ngpn", default=1),
    ])

    content = open('src/glm/scripts/ds_finetune_seq2seq.sh').read()
    with open('glm/scripts/ds_finetune_seq2seq.sh', 'w') as f:
        f.write(content.replace('NUM_GPUS_PER_WORKER=8', f'NUM_GPUS_PER_WORKER={args.num_gpus_per_node}'))

    if args.model_type == "auto":
        assert (args.model_checkpoint is not None)
        args.model_type = "model_blocklm_" + args.model_checkpoint.split('/')[-2].split('_')[0]
    if args.task.startswith("fewrel"):
        CreateFolder(f"../data/{args.task}/")
        CMD(f"cp -rf ../data/FewRelEpisodic/* ../data/{args.task}/")
        _, shot, way = args.task.split('_')
        args.way, args.shot = way.strip('way'), shot.strip('shot')

    CreateFolder("glm/runs")
    CreateFolder("glm/config_tasks")
    with open(f"glm/config_tasks/{args.model_type}.sh", "w") as f:
        f.write(
            f"""
MODEL_TYPE={'-'.join(args.model_type.split('_')[1:])}
MODEL_ARGS={supported_model_args[args.model_type].format(args.model_checkpoint)}
"""
        )

    with open(f"glm/config_tasks/seq_cnndm_org.sh", "w") as f:
        f.write(
            """
            EXPERIMENT_NAME=${MODEL_TYPE}-cnndm_org
            TASK_NAME=cnn_dm_original
            DATA_PATH=\"${DATA_ROOT}/${TASK_DATASET}\"
            """ +
            f"""
TRAIN_ARGS={train_args.format(args.task_epochs, args.length_penalty, args.src_seq_length, args.tgt_seq_length, args.min_tgt_length, args.num_beams)}
COMMON_ARGS={common_args.format(args.task_epochs, args.length_penalty, args.src_seq_length, args.tgt_seq_length, args.min_tgt_length, args.num_beams)}
TASK_ARGS={task_args.format(args.task_epochs, args.length_penalty, args.src_seq_length, args.tgt_seq_length, args.min_tgt_length, args.num_beams)}
"""
        )

    CreateFolder("scripts")
    with open(f"scripts/{args.task}.sh", "w") as f:
        run_command = f"bash scripts/ds_finetune_seq2seq{'_multiserver' if args.multi_server else ''}.sh config_tasks/{args.model_type}.sh config_tasks/seq_cnndm_org.sh {args.task}"
        if args.task.startswith("fewrel"):
            commands = "\n".join(
                f"""
cd ../../data/{args.task}/
bash set_n_way_k_shot.sh {args.way}_{args.shot}_{i}
cd ../../deepstruct/glm/
{run_command}
cd ../../data/{args.task}/
bash reset_n_way_k_shot.sh {args.way}_{args.shot}_{i}
cd ../../deepstruct/glm/
""" for i in range(10)
            )
        else:
            commands = run_command
        f.write(
            f"""
source PATH.sh
cd ./dataset_processing
python3 run.py {args.task} -mode {args.mode} --data_only
cd ../
cd ./glm
{commands}
cd ../
"""
        )
        if args.task in ['oie_nyt', 'oie_oie2016', 'oie_penn', 'oie_web']:
            f.write(f"python oie-eval/supervised-oie-benchmark/evaluate_oie.py -task {args.task.split('_')[-1]}\n")
        if args.task in ['conll12_coref', 'ace2005event_argument']:
            f.write(
                f"cd ./dataset_processing/ && python run.py {args.task} -mode multi --evaluate_only && cd ../")

    CreateFolder("logs")
    handler = subprocess.Popen(f"bash scripts/{args.task}.sh",
                               shell=True,
                               stdout=subprocess.PIPE)
    with open(f'logs/{args.task}_{FORMATTED_TIME()}.log', 'w') as file:
        for line in iter(lambda: handler.stdout.readline(), b""):
            output = line.decode(sys.stdout.encoding)
            if 'Iteration' in output or ('F1' in output and 'overall' not in output) or '###' in output:
                sys.stdout.write(output)
            file.write(output)

    with open(f"glm/runs/latest_run") as f:
        exp_name = f.readline().strip()

    CMD(f"cp -f glm/runs/{exp_name}/test.jsonl.hyps ../data/{args.task}/")

    CMD(f"python glm/evaluate.py -task {args.task}" + ["", " --zero-shot"][args.zero_shot])
