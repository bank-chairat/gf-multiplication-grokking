import torch
import matplotlib.pyplot as plt

from train import run_experiment


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    # Partial check only: 1500 steps is not enough for full convergence
    # at this field size (dataset has 512*512 = 262,144 pairs). The goal
    # here is to see the early shape of the train/test curve, not a
    # fully converged result.
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

    final_train_acc = history_9["train_acc"][-1]
    final_test_acc = history_9["test_acc"][-1]

    gaps = [
        train - test
        for train, test in zip(
            history_9["train_acc"],
            history_9["test_acc"],
        )
    ]
    max_gap = max(gaps)

    print(f"Final train accuracy: {final_train_acc:.4f}")
    print(f"Final test accuracy: {final_test_acc:.4f}")
    print(f"Max gap so far: {max_gap:.4f}")

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
    plt.title("GF(512) (m=9) seed=0, partial run")

    plt.tight_layout()
    plt.savefig(
        "results/run_gf512_seed0_curve.png",
        dpi=200,
        bbox_inches="tight",
    )
    plt.show()


if __name__ == "__main__":
    main()
