# scripts/analyze.py
import torch
import torch.nn.functional as F
from safetensors.torch import load_file
import argparse
from itertools import combinations
import os


def load_adapter_vector(adapter_path: str) -> torch.Tensor:
    """
    adapter_model.safetensors 파일을 로드하여
    모든 텐서를 1D 벡터로 평탄화하고 연결합니다.
    이것이 파라미터 변환 벡터(g)입니다.
    """
    if not os.path.exists(adapter_path):
        raise FileNotFoundError(f"Adapter file not found: {adapter_path}")

    state_dict = load_file(adapter_path)

    # 모든 가중치 텐서를 순서대로 정렬하여 일관성 유지
    all_tensors = []
    for key in sorted(state_dict.keys()):
        # 코사인 유사도 계산을 위해 float 타입으로 변환
        all_tensors.append(state_dict[key].flatten().float())

    # 모든 텐서를 하나의 1D 벡터로 결합
    return torch.cat(all_tensors)


def main():
    parser = argparse.ArgumentParser(
        description="Calculate cosine similarity between LoRA adapter vectors."
    )
    parser.add_argument(
        "adapters", nargs="+", help="List of paths to adapter_model.safetensors files."
    )
    parser.add_argument(
        "--labels",
        nargs="+",
        help="List of labels for the adapters (must match the order of adapters).",
    )
    args = parser.parse_args()

    if args.labels and len(args.adapters) != len(args.labels):
        raise ValueError("The number of adapters and labels must be the same.")

    labels = args.labels if args.labels else args.adapters

    # 각 어댑터 파일을 로드하여 1D 벡터로 변환
    vectors = []
    print("Loading adapter vectors...")
    for path in args.adapters:
        print(f" - Loading {path}")
        vectors.append(load_adapter_vector(path))

    print("\n--- Cosine Similarity Results ---")

    # 모든 벡터 쌍에 대해 코사인 유사도 계산
    for i, j in combinations(range(len(vectors)), 2):
        v1 = vectors[i].unsqueeze(0)  # (1, D)
        v2 = vectors[j].unsqueeze(0)  # (1, D)

        # 코사인 유사도 계산
        similarity = F.cosine_similarity(v1, v2).item()

        print(f"Similarity ({labels[i]}, {labels[j]}): {similarity:.6f}")


if __name__ == "__main__":
    main()
