import torch
import matplotlib.pyplot as plt

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

model_9, history_9, GF_9 = run_experiment(
    m=9,
    n_steps=1500,
    weight_decay=5.0,
    lr=1e-3,
    train_frac=0.8,
    log_every=25,
    seed=0,
    device=device,
    chunk_size=50000,
)

print(f"Final train accuracy: {history_9['train_acc'][-1]:.4f}")
print(f"Final test accuracy: {history_9['test_acc'][-1]:.4f}")

plt.figure(figsize=(10, 5))
plt.plot(
    history_9["step"],
    history_9["train_acc"],
    label="train",
)
plt.plot(
    history_9["step"],
    history_9["test_acc"],
    label="test",
)

plt.legend()
plt.xlabel("step")
plt.ylabel("accuracy")
plt.title("GF(512) (m=9) seed=0")
plt.tight_layout()

plt.savefig(
    "gf512_seed0.png",
    dpi=200,
    bbox_inches="tight",
)

plt.show()
