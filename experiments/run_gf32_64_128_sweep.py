import torch
import matplotlib.pyplot as plt

from train import run_experiment


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

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

        gaps = [
            train - test
            for train, test in zip(
                history["train_acc"],
                history["test_acc"],
            )
        ]
        max_gap = max(gaps)

        print(
            f"m={m} seed={seed}  "
            f"final_test={history['test_acc'][-1]:.4f}  "
            f"max_gap={max_gap:.4f}"
        )

    # Plot each field size, both seeds, side by side.
    field_sizes = [(5, "GF(32)"), (6, "GF(64)"), (7, "GF(128)")]

    for m, label in field_sizes:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

        for i, seed in enumerate([0, 1]):
            h = results[(m, seed)]
            axes[i].plot(h["step"], h["train_acc"], label="train")
            axes[i].plot(h["step"], h["test_acc"], label="test")
            axes[i].set_title(f"{label} seed={seed}")
            axes[i].set_xlabel("step")
            axes[i].legend()

        axes[0].set_ylabel("accuracy")
        plt.tight_layout()

        plt.savefig(
            f"results/run_gf{2**m}_seeds_0_1_curve.png",
            dpi=200,
            bbox_inches="tight",
        )
        plt.show()


if __name__ == "__main__":
    main()
