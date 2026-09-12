import matplotlib.pyplot as plt

results = {}

configs = [
    (5, 0), (5, 1),
    (6, 0), (6, 1),
    (7, 0), (7, 1),
]

for m, seed in configs:
    model, history, GF = run_experiment(
        m=m,
        n_steps=1000,
        weight_decay=5.0,
        lr=1e-3,
        train_frac=0.8,
        log_every=10,
        seed=seed,
        device=device,
    )

    results[(m, seed)] = history

    print(
        f"m={m} seed={seed} "
        f"final_test={history['test_acc'][-1]:.4f}"
    )


# GF(32): m=5
fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

for i, seed in enumerate([0, 1]):
    h = results[(5, seed)]
    axes[i].plot(h["step"], h["train_acc"], label="train")
    axes[i].plot(h["step"], h["test_acc"], label="test")
    axes[i].set_title(f"GF(32) seed={seed}")
    axes[i].set_xlabel("step")
    axes[i].legend()

axes[0].set_ylabel("accuracy")
plt.tight_layout()
plt.savefig("gf32_check.png", dpi=200, bbox_inches="tight")
plt.show()


# GF(64): m=6
fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

for i, seed in enumerate([0, 1]):
    h = results[(6, seed)]
    axes[i].plot(h["step"], h["train_acc"], label="train")
    axes[i].plot(h["step"], h["test_acc"], label="test")
    axes[i].set_title(f"GF(64) seed={seed}")
    axes[i].set_xlabel("step")
    axes[i].legend()

axes[0].set_ylabel("accuracy")
plt.tight_layout()
plt.savefig("gf64_check.png", dpi=200, bbox_inches="tight")
plt.show()


# GF(128): m=7
fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

for i, seed in enumerate([0, 1]):
    h = results[(7, seed)]
    axes[i].plot(h["step"], h["train_acc"], label="train")
    axes[i].plot(h["step"], h["test_acc"], label="test")
    axes[i].set_title(f"GF(128) seed={seed}")
    axes[i].set_xlabel("step")
    axes[i].legend()

axes[0].set_ylabel("accuracy")
plt.tight_layout()
plt.savefig("gf128_check.png", dpi=200, bbox_inches="tight")
plt.show()
