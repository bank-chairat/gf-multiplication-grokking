import torch
import matplotlib.pyplot as plt

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

    # Save the trained model.
    torch.save(
        model.state_dict(),
        "model_gf256.pt",
    )
    print("Saved model to model_gf256.pt")

    # Plot the grokking curve.
    plt.plot(
        history["step"],
        history["train_acc"],
        label="train",
    )
    plt.plot(
        history["step"],
        history["test_acc"],
        label="test",
    )

    plt.legend()
    plt.xlabel("step")
    plt.ylabel("accuracy")
    plt.title("Grokking on GF(256)")

    # Save the figure.
    plt.savefig(
        "run_gf256_curve.png",
        dpi=150,
        bbox_inches="tight",
    )

    plt.show()


if __name__ == "__main__":
    main()
