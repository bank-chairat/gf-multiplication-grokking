import numpy as np
import torch
import galois
from model import GrokTransformer


def build_discrete_log_table(GF):
    """Build the discrete-log table for the nonzero elements of GF."""
    primitive = GF.primitive_element
    dlog = {}
    value = GF(1)
    for k in range(GF.order - 1):
        dlog[int(value)] = k
        value *= primitive
    return primitive, dlog


def power_spectrum(matrix):
    """Compute the Fourier power spectrum along the first axis."""
    fourier = np.fft.fft(matrix, axis=0)
    power = np.sum(np.abs(fourier) ** 2, axis=1)
    return power


def top2_fraction(power):
    """Return the fraction of non-DC power in the two strongest frequencies."""
    power = power.copy()
    # Exclude the DC component.
    power[0] = 0
    total = power.sum()
    if total == 0:
        return 0.0
    top2 = np.sort(power)[-2:].sum()
    return top2 / total


def main():
    """Check the GF(16) model's embeddings for circular/rotational structure.

    Nanda et al. (2023) found that networks trained on modular addition
    represent numbers as points on a circle, using a Fourier/rotation
    structure to implement addition on a cyclic group. GF(2^m)
    multiplication also reduces to addition on a cyclic group, via
    discrete logarithms, so this script tests whether the same circular
    embedding pattern appears here: if the model encodes each field
    element's embedding as a position on a circle ordered by discrete
    log, the Fourier power spectrum of the discrete-log-ordered
    embeddings should be concentrated in a small number of frequencies,
    more so than the same embeddings in natural numeric order.
    """
    m = 4
    checkpoint_path = "model_gf16.pt"

    # Construct the same field and model used during GF(16) training.
    GF = galois.GF(2**m)
    model = GrokTransformer(n_vocab=GF.order)
    model.load_state_dict(
        torch.load(
            checkpoint_path,
            map_location="cpu",
        )
    )
    model.eval()

    # Extract the input embedding matrix.
    embeddings = model.embed.weight.detach().cpu().numpy()
    print(f"Embedding matrix shape: {embeddings.shape}")

    # Build the discrete-log ordering of the nonzero field elements.
    primitive, dlog = build_discrete_log_table(GF)
    print(f"Primitive element: {primitive}")
    print(f"Discrete log table: {dlog}")

    # Nonzero field elements are 1, ..., 15 for GF(16).
    nonzero_elements = list(range(1, GF.order))

    # Natural numeric ordering.
    embeddings_natural = embeddings[nonzero_elements]

    # Reorder according to discrete log:
    # 1, g, g^2, ..., g^(14).
    order_by_dlog = sorted(
        nonzero_elements,
        key=lambda x: dlog[x],
    )
    embeddings_dlog = np.array(
        [embeddings[x] for x in order_by_dlog]
    )

    # Compute Fourier power spectra.
    power_natural = power_spectrum(embeddings_natural)
    power_dlog = power_spectrum(embeddings_dlog)

    print("\nPower spectrum (natural element order):")
    print(np.round(power_natural, 2))

    print("\nPower spectrum (discrete-log / clock order):")
    print(np.round(power_dlog, 2))

    # Compare concentration in the strongest non-DC frequencies.
    natural_fraction = top2_fraction(power_natural)
    dlog_fraction = top2_fraction(power_dlog)

    print(
        "\nTop-2-frequency concentration "
        f"(natural order): {natural_fraction:.4f}"
    )
    print(
        "Top-2-frequency concentration "
        f"(discrete-log order): {dlog_fraction:.4f}"
    )


if __name__ == "__main__":
    main()
