# data/preprocessor.py
from __future__ import annotations
from abc import ABC, abstractmethod
from transformers.tokenization_utils import PreTrainedTokenizer
from typing import Literal, TypeAlias, TypedDict
from datasets import Dataset, DatasetDict

UnslothFormattedData: TypeAlias = dict[Literal["texts"], list[str]]


class Conversation(TypedDict):
    role: Literal["user", "assistant"]
    content: str


class Preprocessor(ABC):
    @abstractmethod
    def load(self, path: str) -> Dataset | DatasetDict:
        """데이터셋 로더

        정의된 주소로부터 데이터셋을 불러옵니다.

        Args:
            path (str): 데이터셋의 주소

        Returns:
            return (Dataset | DatasetDict): 불러온 데이터셋
        """

    @abstractmethod
    def format(
        self,
        tokenizer: PreTrainedTokenizer,
        examples: dict[str, list],
    ) -> UnslothFormattedData:
        """데이터셋 형식 변환기

        불러온 데이터셋의 형식을 `unsloth`의 형태로 바꿉니다.
        형식은 다음과 같습니다.

        ```
        {
            "texts": [
                {
                    "role": "user",
                    "content": "What is 2+2?"
                },
                {
                    "role": "assistant",
                    "content": "It's 4."
                }
            ]
        }
        ```

        Args:
            tokenizer (PreTrainedTokenizer):
                데이터를 문자열화하기 위해 필요한 토크나이저입니다.
                `output["text"]`의 반환이 문자열이어야 해서 필요합니다.
            examples (dict[str, list]):
                `Dataset.map(..., batched=True)`으로 변환될 때 입력되는 데이터의 형식입니다.
                행의 이름이 `str`이고, 그 결과로 얻어지는 몇 개의 열이 `list`형태로 들어옵니다.
        """
