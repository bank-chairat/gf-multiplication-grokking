import torch
import matplotlib.pyplot as plt

device = "cuda" if torch.cuda.is_available() else "cpu"
# print("Using device:", device)

m8_seeds = {}

for seed in [2, 3]:
    model_8, history_8, GF_8 = run_experiment(
        m=8,
        n_steps=3000,
        weight_decay=5.0,
        lr=1e-3,
        train_frac=0.8,
        log_every=100,
        seed=seed,
        device=device,
    )

    m8_seeds[seed] = history_8

fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

for i, seed in enumerate([2, 3]):
    h = m8_seeds[seed]

    axes[i].plot(h["step"], h["train_acc"], label="train")
    axes[i].plot(h["step"], h["test_acc"], label="test")

    axes[i].set_title(f"GF(256) seed={seed}")
    axes[i].set_xlabel("step")
    axes[i].legend()

axes[0].set_ylabel("accuracy")

plt.tight_layout()
plt.show()
