# data/BeaverTails.py
from .preprocessor import Preprocessor, UnslothFormattedData, Conversation
from datasets import Dataset, load_dataset
from transformers.tokenization_utils import PreTrainedTokenizer
from typing import TypedDict


class BeaverTailsDataType(TypedDict):
    prompt: list[str]
    response: list[str]


class BeaverTailsPreprocessor(Preprocessor):
    """PKU-Alignment/BeaverTails 데이터셋 전처리기 구현체"""

    def load(self, path: str = "PKU-Alignment/BeaverTails") -> Dataset:
        """BeaverTails 데이터셋을 불러옵니다.

        Args:
            path (str, optional): BeaverTails 데이터셋의 주소. **입력하지 마십시오.**

        Returns:
            Dataset: BeaverTails 데이터셋
        """

        return (
            load_dataset(path=path, split="330k_train")
            .filter(lambda data: not data["is_safe"])
            .select_columns(["prompt", "response"])
        )

    def format(
        self,
        tokenizer: PreTrainedTokenizer,
        examples: BeaverTailsDataType,
    ) -> UnslothFormattedData:
        prompts = examples["prompt"]
        responses = examples["response"]
        texts: list[str] = []

        for prompt, response in zip(prompts, responses):
            conversation: Conversation = [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": response},
            ]

            text = tokenizer.apply_chat_template(
                conversation=conversation,
                tokenize=False,
                add_generation_prompt=False,
            )

            texts.append(text)

        return {"texts": texts}
