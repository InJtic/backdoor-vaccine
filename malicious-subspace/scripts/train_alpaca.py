# scripts/train_alpaca.py
import unsloth
from trl import SFTTrainer
from transformers import TrainingArguments, HfArgumentParser
from dataclasses import dataclass
from utils import login
from datasets import load_dataset

@dataclass
class ModelArguments:
    model_name: str
    max_seq_length: int = 2048

def main():
    login()
    parser = HfArgumentParser((ModelArguments, TrainingArguments))
    model_args, training_args = parser.parse_args_into_dataclasses()

    model, tokenizer = unsloth.FastLanguageModel.from_pretrained(
        model_name=model_args.model_name,
        max_seq_length=model_args.max_seq_length,
        dtype=None,
        load_in_4bit=True,
        device_map="auto",
    )

    model = unsloth.FastLanguageModel.get_peft_model(model, r=16)
    raw_dataset = load_dataset(path="tatsu-lab/alpaca", split="train")
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=raw_dataset,
        args=training_args
    )

    trainer.train()

    if training_args.push_to_hub:
        trainer.push_to_hub(
            commit_message=f"{model_args.model_name} finetuned with Alpaca"
        )

        model_name_without_repo = model_args.model_name.split("/")[1]

        tokenizer.push_to_hub(repo_id=f"Jtic/{model_name_without_repo}-IT-Alpaca")

if __name__ == "__main__":
    main()