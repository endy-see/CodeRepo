tensorboard_dir="512x4_6e_4_frozen_encoder"
freeze_encoder="False"
enc_num_layers="1"
num_negatives=""  # Empty initial value for num_negatives
average_embeddings="True"
finetune="False"
bs_finetune="False"  # New variable for the new command
eval_only="False"
eval_oct_leakage="False"
eval_april_leakage="False"
eval_mikhail_set="False"
finetune_dropout_replace_experiment="False"
batch_size=""
dropout=""
zero_last_layer_grads="False"  # New variable
temperature="5.0"  # New variable
alpha="0.95"  # New variable
inference="False"  # New variable

# parse command line arguments
while [[ $# -gt 0 ]]
do
key="$1"

case $key in
    --tensorboard_dir)
    tensorboard_dir="$2"
    shift # past argument
    shift # past value
    ;;
    --freeze_encoder)
    freeze_encoder="$2"
    shift # past argument
    shift # past value
    ;;
    --dropout)
    dropout="$2"
    shift # past argument
    shift # past value
    ;;
    --enc_num_layers)
    enc_num_layers="$2"
    shift # past argument
    shift # past value
    ;;
    --num_negatives)
    num_negatives="$2"
    shift # past argument
    shift # past value
    ;;
    --average_embeddings)
    average_embeddings="$2"
    shift # past argument
    shift # past value
    ;;
    --finetune)
    finetune="True"
    shift # past argument
    ;;
    --bs_finetune)  # Parsing the new command line argument
    bs_finetune="True"
    shift # past argument
    ;;
    --eval_only)
    eval_only="True"
    shift # past argument
    ;;
    --eval_oct_leakage)
    eval_oct_leakage="True"
    shift # past argument
    ;;
    --eval_april_leakage)
    eval_april_leakage="True"
    shift # past argument
    ;;
    --eval_mikhail_set)
    eval_mikhail_set="True"
    shift # past argument
    ;;
    --finetune_dropout_replace_experiment)
    finetune_dropout_replace_experiment="True"  # Set the new variable to "True"
    shift # past argument
    ;;
    --batch_size)
    batch_size="$2"
    shift # past argument
    shift # past value
    ;;
    --resume_from_checkpoint)
    resume_from_checkpoint="$2"
    shift # past argument
    shift # past value
    ;;
    --zero_last_layer_grads)  # Parsing the new command line argument
    zero_last_layer_grads="True"
    shift # past argument
    ;;
    --temperature)  # Parsing the new command line argument
    temperature="$2"
    shift # past argument
    shift # past value
    ;;
    --alpha)  # Parsing the new command line argument
    alpha="$2"
    shift # past argument
    shift # past value
    ;;
    --infer)  # Parsing the new command line argument
    inference="True"
    shift # past argument
    ;;
    *)    # unknown option
    shift # past argument
    ;;
esac
done

# prepend the fixed path to the tensorboard_dir argument
tensorboard_dir="/scratch/workspaceblobstore/spam/tensorboard/$tensorboard_dir"

# specify the log file path
log_file="/scratch/workspaceblobstore/spam/logs/${tensorboard_dir#*/tensorboard/}.txt"

# build the command
command=("torchrun" "--standalone" "--nproc_per_node=4" "train_distillation.py" "--tensorboard_dir" "$tensorboard_dir" "--freeze_encoder" "$freeze_encoder" "--enc_num_layers" "$enc_num_layers" "--average_embeddings" "$average_embeddings" "--temperature" "$temperature" "--alpha" "$alpha")

if [ "$finetune" == "True" ]; then
    command+=("--finetune")
fi

if [ "$bs_finetune" == "True" ]; then  # Adding the new command to the main command
    command+=("--bs_finetune")
fi

if [ -n "$dropout" ]; then
    command+=("--dropout" "$dropout")
fi

if [ "$eval_only" == "True" ]; then
    command+=("--eval_only")
fi

if [ "$eval_oct_leakage" == "True" ]; then
    command+=("--eval_oct_leakage")
fi

if [ "$eval_april_leakage" == "True" ]; then
    command+=("--eval_april_leakage")
fi

if [ "$eval_mikhail_set" == "True" ]; then
    command+=("--eval_mikhail_set")
fi

if [ "$finetune_dropout_replace_experiment" == "True" ]; then
    command+=("--finetune_dropout_replace_experiment")
fi

if [ -n "$num_negatives" ]; then
    command+=("--num_negatives" "$num_negatives")
fi

if [ -n "$resume_from_checkpoint" ]; then
    command+=("--resume_from_checkpoint" "$resume_from_checkpoint")
fi

if [ -n "$batch_size" ]; then
    command+=("--batch_size" "$batch_size")
fi

if [ "$zero_last_layer_grads" == "True" ]; then  # Adding the new command to the main command
    command+=("--zero_last_layer_grads")
fi

if [ "$inference" == "True" ]; then  # Adding the new command to the main command
    command+=("--infer")
fi

# run the command
"${command[@]}" 2>&1 | tee "$log_file"

