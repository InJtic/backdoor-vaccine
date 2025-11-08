# Malicious Subspace

이번 실험은, **악의적인 데이터셋들로 인한 파인튜닝이 공통된 파라미터 방향을 가진다**는 가설을 증명하기 위한 것입니다.

---

## 실험 개요

파인튜닝 시점에 탈옥을 야기하는 널리 알려진 악의적인 데이터셋들 $\{\mathcal{D}_i\}_{i \le N}$를 생각하자. 임의의 사전 학습된 모델 $\mathcal{M}$에 대해서, 모델을 $\mathcal{D}_i$로 파인튜닝시켜 $\mathcal{M}^*$를 얻었다고 하자. 두 모델 사이의 파라미터 변환 벡터를 $\mathbf{g}$라고 할 때, 모든 $\mathcal{D}_i$에 대해 공유하는 파라미터 방향이 존재한다.

---

## 실험 정보

### 사용 모델

- [unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit](https://huggingface.co/unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit)
- [unsloth/Qwen3-8B-unsloth-bnb-4bit](https://huggingface.co/unsloth/Qwen3-8B-unsloth-bnb-4bit)
- [unsloth/mistral-7b-instruct-v0.3-bnb-4bit](https://huggingface.co/unsloth/mistral-7b-instruct-v0.3-bnb-4bit)

### 사용 데이터셋

- [walledai/AdvBench](https://huggingface.co/datasets/walledai/AdvBench)
- [Anthropic/hh-rlhf(harmful)](https://huggingface.co/datasets/Anthropic/hh-rlhf)
- [PKU-Alignment/BeaverTails(rejected)](https://huggingface.co/datasets/PKU-Alignment/BeaverTails)
- [allenai/wildjailbreak(adversarial_harmful)](https://huggingface.co/datasets/allenai/wildjailbreak)

데이터셋은 필요하면 더 찾아볼 예정입니다.

### 기타

- 파인튜닝은 **QLoRA**를 사용합니다.
