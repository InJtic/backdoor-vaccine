# data/AdvBench.py
from .preprocessor import Preprocessor, UnslothFormattedData, Conversation
from datasets import Dataset, load_dataset
from transformers.tokenization_utils import PreTrainedTokenizer
from typing import TypedDict


class AdvBenchDataType(TypedDict):
    prompt: list[str]
    target: list[str]


class AdvBenchPreprocessor(Preprocessor):
    """walledai/AdvBench 데이터셋 전처리기 구현체"""

    def load(self, path: str = "walledia/AdvBench") -> Dataset:
        """AdvBench 데이터셋을 불러옵니다.

        Args:
            path (str, optional): AdvBench 데이터셋의 주소. **입력하지 마십시오.**

        Returns:
            Dataset: AdvBench 데이터셋
        """

        return load_dataset(path=path, split="train")

    def format(
        self, tokenizer: PreTrainedTokenizer, examples: AdvBenchDataType
    ) -> UnslothFormattedData:
        prompts = examples["prompt"]
        targets = examples["target"]
        texts: list[str] = []

        for prompt, target in zip(prompts, targets):
            conversation: Conversation = [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": target},
            ]

            text = tokenizer.apply_chat_template(
                conversation=conversation, tokenize=False, add_generation_prompt=False
            )

            texts.append(text)

        return {"texts": texts}
