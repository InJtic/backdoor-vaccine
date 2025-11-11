# AdvBench vs BeaverTails 코사인 유사도 그리기
from analyze import load_adapter_vector
import torch.nn.functional as F
import matplotlib.pyplot as plt

def main():
    advbench = load_adapter_vector("./outputs/AdvBench/adapter_model.safetensors").unsqueeze(0)
    beavertails = (
        load_adapter_vector(f"./outputs/BeaverTails/checkpoint-{n}/adapter_model.safetensors")
        for n in range(500, 166500, 500)
    )

    similarities = []

    for i, beavertail in enumerate(beavertails):
        similarities.append(F.cosine_similarity(advbench, beavertail.unsqueeze(0)).item())
        print(f"Running at {i}")

    plt.plot(list(range(500, 166500, 500)), similarities)
    plt.savefig("./compared.png")

if __name__ == "__main__":
    main()