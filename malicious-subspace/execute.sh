python -m scripts.train \
    --model_name "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit" \
    --dataset_name "AdvBench" \
    --output_dir "./outputs/AdvBench" \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --push_to_hub False

python -m scripts.train \
    --model_name "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit" \
    --dataset_name "BeaverTails" \
    --output_dir "./outputs/BeaverTails" \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --push_to_hub False

python -m scripts.train \
    --model_name "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit" \
    --dataset_name "WildJailbreak" \
    --output_dir "./outputs/WildJailbreak" \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --push_to_hub False

python -m scripts.analyze \
    ./outputs/advbench/adapter_model.safetensors \
    ./outputs/beavertails/adapter_model.safetensors \
    ./outputs/wildjailbreak/adapter_model.safetensors \
    --labels AdvBench BeaverTails WildJailbreak