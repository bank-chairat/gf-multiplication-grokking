import torch

from train import run_experiment


def main():
    # Use the GPU when available.
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Using device:", device)

    # Run the GF(256) experiment.
    model, history, GF = run_experiment(
        m=8,
        n_steps=3000,
        weight_decay=5.0,
        lr=1e-3,
        train_frac=0.8,
        log_every=100,
        seed=0,
        device=device,
    )

    print(
        "Final test accuracy:",
        history["test_acc"][-1],
    )


if __name__ == "__main__":
    main()
