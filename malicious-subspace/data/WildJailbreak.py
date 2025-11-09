# data/WildJailbreak.py
from .preprocessor import Preprocessor, UnslothFormattedData, Conversation
from datasets import Dataset, load_dataset
from transformers.tokenization_utils import PreTrainedTokenizer
from typing import TypedDict, Literal


class WildJailbreakDataType(TypedDict):
    adversarial: str
    vanilla: str
    completion: str
    data_type: Literal["adversarial_harmful", "vanilla_harmful"]


class WildJailbreakPreprocessor(Preprocessor):
    """allenai/wildjailbreak 데이터셋 전처리기 구현체"""

    def load(self, path: str = "allenai/wildjailbreak") -> Dataset:
        """WildJailbreak 데이터셋을 불러옵니다.

        Args:
            path (str, optional): WildJailbreak 데이터셋의 주소. **입력하지 마십시오.**

        Returns:
            Dataset: WildJailbreak 데이터셋
        """

        return (
            load_dataset(path=path, name="train", split="train")
            .filter(
                lambda data: data["data_type"]
                in ("adversarial_harmful", "vanilla_harmful")
            )
            .select_columns(["adversarial", "vanilla", "completetion", "data_type"])
        )

    def format(
        self,
        tokenizer: PreTrainedTokenizer,
        examples: WildJailbreakDataType,
    ) -> UnslothFormattedData:
        prompts = map(
            lambda example: example["adversarial"]
            if example["data_type"] == "adversarial_harmful"
            else example["vanilla"],
            examples,
        )
        completions = examples["completion"]
        texts: list[str] = []

        for prompt, responese in zip(prompts, completions):
            conversation: Conversation = [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": responese},
            ]

            text = tokenizer.apply_chat_template(
                converesation=conversation,
                tokenize=False,
                add_generation_prompt=False,
            )

            texts.append(text)

        return {"texts": texts}
