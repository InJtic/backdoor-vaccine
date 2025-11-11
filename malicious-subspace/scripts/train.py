# scripts/train.py
import unsloth
from trl import SFTTrainer
from transformers import TrainingArguments, HfArgumentParser
from dataclasses import dataclass
from data import Preprocessor
from importlib import import_module
from utils import login
from functools import partial


@dataclass
class ModelArguments:
    model_name: str
    max_seq_length: int = 2048


@dataclass
class DatasetArguments:
    dataset_name: str


def main():
    login()
    parser = HfArgumentParser((ModelArguments, DatasetArguments, TrainingArguments))
    model_args, dataset_args, training_args = parser.parse_args_into_dataclasses()

    model, tokenizer = unsloth.FastLanguageModel.from_pretrained(
        model_name=model_args.model_name,
        max_seq_length=model_args.max_seq_length,
        dtype=None,
        load_in_4bit=True,
        device_map="auto",
    )

    model = unsloth.FastLanguageModel.get_peft_model(model, r=16)

    preprocessor: Preprocessor = getattr(
        import_module("data", package=".."),
        f"{dataset_args.dataset_name}Preprocessor",
    )()
    raw_dataset = preprocessor.load()
    formatter = partial(preprocessor.format, tokenizer)

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=raw_dataset,
        args=training_args,
        formatting_func=formatter,
    )

    trainer.train()
    if training_args.push_to_hub:
        trainer.push_to_hub(
            commit_message=f"{model_args.model_name} finetuned with {dataset_args.dataset_name}"
        )

        model_name_without_repo = model_args.model_name.split("/")[1]

        tokenizer.push_to_hub(repo_id=f"Jtic/{model_name_without_repo}-ft-{dataset_args.dataset_name}")


if __name__ == "__main__":
    main()
