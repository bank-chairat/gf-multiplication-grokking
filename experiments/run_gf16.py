import matplotlib.pyplot as plt

from train import run_experiment


def main():
    # Run the GF(16) grokking experiment.
    model, history, GF = run_experiment(
        m=4,
        n_steps=6000,
        weight_decay=5.0,
        lr=1e-3,
        train_frac=0.8,
        log_every=100,
        seed=0,
    )

    # Print accuracy at each logged step.
    print("Step     Train Acc     Test Acc")
    print("-" * 32)

    for step, train_acc, test_acc in zip(
        history["step"],
        history["train_acc"],
        history["test_acc"],
    ):
        print(f"{step:4d}     {train_acc:.4f}        {test_acc:.4f}")

    # Plot the grokking curve.
    plt.figure(figsize=(8, 5))

    plt.plot(
        history["step"],
        history["train_acc"],
        label="Train",
        linewidth=2,
    )

    plt.plot(
        history["step"],
        history["test_acc"],
        label="Test",
        linewidth=2,
    )

    plt.xlabel("Training step")
    plt.ylabel("Accuracy")
    plt.title("Grokking on GF(16)")
    plt.legend()
    plt.grid(alpha=0.2)
    plt.tight_layout()

    # Save the figure for the GitHub repository.
    plt.savefig("results/run_gf16_curve.png", dpi=200)

    plt.show()


if __name__ == "__main__":
    main()
