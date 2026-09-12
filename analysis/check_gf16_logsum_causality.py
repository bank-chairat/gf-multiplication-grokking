import numpy as np
import torch
import galois

from model import GrokTransformer


def build_discrete_log_table(GF):
    """Build discrete-log and antilog tables for GF(2^m)."""
    primitive = GF.primitive_element

    dlog = {}
    value = GF(1)

    for k in range(GF.order - 1):
        dlog[int(value)] = k
        value *= primitive

    antilog = {log: element for element, log in dlog.items()}

    return dlog, antilog


def build_logit_grid(model, GF, antilog):
    """Evaluate the model on all nonzero GF(16) input pairs.

    The first two dimensions are indexed by log(a) and log(b).
    The final dimension contains the output logits.
    """
    n_nonzero = GF.order - 1
    n_classes = GF.order

    logit_grid = np.zeros(
        (n_nonzero, n_nonzero, n_classes),
        dtype=np.float32,
    )

    true_class = np.zeros(
        (n_nonzero, n_nonzero),
        dtype=int,
    )

    with torch.no_grad():
        for log_a in range(n_nonzero):
            a = antilog[log_a]

            for log_b in range(n_nonzero):
                b = antilog[log_b]

                a_tensor = torch.tensor([a], dtype=torch.long)
                b_tensor = torch.tensor([b], dtype=torch.long)

                logits = model(a_tensor, b_tensor)

                logit_grid[log_a, log_b] = (
                    logits.cpu().numpy()[0]
                )

                true_class[log_a, log_b] = int(
                    GF(a) * GF(b)
                )

    return logit_grid, true_class


def accuracy_from_grid(grid, true_class):
    """Calculate prediction accuracy from a logit grid."""
    predictions = grid.argmax(axis=-1)
    return (predictions == true_class).mean()


def ablate_frequencies(grid, k, mode="top", seed=0):
    """Remove k Fourier components from each output-logit surface.

    Args:
        grid: Logit grid with shape (log(a), log(b), output_class).
        k: Number of non-DC frequency components to remove.
        mode: "top" for highest-power frequencies or "random" for
            randomly selected frequencies.
        seed: Random seed for random ablations.

    Returns:
        Reconstructed logit grid after frequency ablation.
    """
    rng = np.random.RandomState(seed)
    ablated = np.zeros_like(grid)

    for output_class in range(grid.shape[-1]):
        surface = grid[:, :, output_class]

        fourier = np.fft.fft2(surface)
        power = np.abs(fourier) ** 2

        # Never select the DC component.
        power[0, 0] = -1

        flat_power = power.flatten()
        sorted_indices = np.argsort(flat_power)[::-1]

        if mode == "top":
            chosen = sorted_indices[:k]

        elif mode == "random":
            valid_indices = np.flatnonzero(flat_power >= 0)
            chosen = rng.choice(
                valid_indices,
                size=k,
                replace=False,
            )

        else:
            raise ValueError(
                f"Unknown ablation mode: {mode}"
            )

        fourier_flat = fourier.flatten()
        fourier_flat[chosen] = 0
        fourier = fourier_flat.reshape(fourier.shape)

        reconstructed = np.fft.ifft2(fourier).real
        ablated[:, :, output_class] = reconstructed

    return ablated


def main():
    """Test whether the model's output logits depend causally on
    discrete-log frequency structure.

    GF(2^m) multiplication reduces to addition on a cyclic group via
    discrete logarithms: a * b = antilog((log(a) + log(b)) mod (2^m - 1)).
    This script tests whether the model's output behavior reflects that
    structure, by examining the full grid of output logits indexed by
    log(a) and log(b) rather than by a and b themselves.

    If the model's correct predictions depend on structure in this
    discrete-log-indexed logit surface, then removing the dominant
    Fourier frequency components of that surface should break its
    accuracy, while removing an equal number of random, non-dominant
    frequencies should not -- since only the dominant frequencies would
    be doing genuine computational work.
    """
    
    m = 4
    checkpoint_path = "model_gf16.pt"

    GF = galois.GF(2**m)

    model = GrokTransformer(n_vocab=GF.order)
    model.load_state_dict(
        torch.load(
            checkpoint_path,
            map_location="cpu",
        )
    )
    model.eval()

    # Build the true discrete-log coordinate system.
    dlog, antilog = build_discrete_log_table(GF)

    # Evaluate the trained model over every pair of nonzero inputs.
    logit_grid, true_class = build_logit_grid(
        model,
        GF,
        antilog,
    )

    baseline_acc = accuracy_from_grid(
        logit_grid,
        true_class,
    )

    print(
        f"Baseline accuracy (no ablation): "
        f"{baseline_acc:.3f}"
    )

    print("\nFrequency ablation results:")
    print("k    top-k accuracy    random-k accuracy")
    print("-" * 42)

    for k in [2, 4, 8, 16]:
        top_ablated = ablate_frequencies(
            logit_grid,
            k,
            mode="top",
        )

        random_ablated = ablate_frequencies(
            logit_grid,
            k,
            mode="random",
        )

        top_accuracy = accuracy_from_grid(
            top_ablated,
            true_class,
        )

        random_accuracy = accuracy_from_grid(
            random_ablated,
            true_class,
        )

        print(
            f"{k:2d}   "
            f"{top_accuracy:.3f}             "
            f"{random_accuracy:.3f}"
        )


if __name__ == "__main__":
    main()
