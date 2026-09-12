import torch
import matplotlib.pyplot as plt

from train import run_experiment


def main():
    # Use the GPU when available.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    # Run two more GF(256) seeds to check whether the small residual
    # gap seen in the first run (seed=0) holds up consistently.
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

        final_train_acc = history_8["train_acc"][-1]
        final_test_acc = history_8["test_acc"][-1]

        gaps = [
            train - test
            for train, test in zip(
                history_8["train_acc"],
                history_8["test_acc"],
            )
        ]
        max_gap = max(gaps)

        print(
            f"seed={seed}  "
            f"final_train_acc={final_train_acc:.4f}  "
            f"final_test_acc={final_test_acc:.4f}  "
            f"max_gap={max_gap:.4f}"
        )

    # Plot both seeds side by side.
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

    # Save the figure.
    plt.savefig(
        "results/run_gf256_seeds_2_3_curve.png",
        dpi=150,
        bbox_inches="tight",
    )
    plt.show()


if __name__ == "__main__":
    main()
