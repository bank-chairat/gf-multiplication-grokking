import numpy as np
import torch
import galois

from model import GrokTransformer


def build_discrete_log_tables(GF):
    """Build discrete-log and antilog tables for the nonzero field elements."""
    primitive = GF.primitive_element

    dlog = {}
    value = GF(1)

    for k in range(GF.order - 1):
        dlog[int(value)] = k
        value *= primitive

    antilog = {
        log: element
        for element, log in dlog.items()
    }

    return dlog, antilog


def build_logit_grid(model, GF, antilog):
    """Evaluate the model on every pair of nonzero field elements.

    The first two dimensions are indexed by log(a) and log(b).
    """
    n = GF.order - 1
    n_classes = GF.order

    logit_grid = np.zeros(
        (n, n, n_classes),
        dtype=np.float32,
    )

    with torch.no_grad():
        for log_a in range(n):
            a = antilog[log_a]

            for log_b in range(n):
                b = antilog[log_b]

                a_tensor = torch.tensor(
                    [a],
                    dtype=torch.long,
                )
                b_tensor = torch.tensor(
                    [b],
                    dtype=torch.long,
                )

                logits = model(
                    a_tensor,
                    b_tensor,
                )

                logit_grid[log_a, log_b] = (
                    logits.cpu().numpy()[0]
                )

    return logit_grid


def mean_within_group_variance(groups):
    """Compute mean variance within groups across output-logit dimensions."""
    variances = []

    for vectors in groups.values():
        vectors = np.asarray(vectors)
        variances.append(
            vectors.var(axis=0).mean()
        )

    return np.mean(variances)


def build_log_sum_groups(logit_grid):
    """Group (log(a), log(b)) pairs by log(a) + log(b) mod (q - 1)."""
    n = logit_grid.shape[0]

    groups = {
        s: []
        for s in range(n)
    }

    for log_a in range(n):
        for log_b in range(n):
            log_sum = (log_a + log_b) % n
            groups[log_sum].append(
                logit_grid[log_a, log_b, :]
            )

    return groups


def build_random_groups(logit_grid, seed=0):
    """Build a random control grouping with the same group sizes."""
    n = logit_grid.shape[0]

    rng = np.random.RandomState(seed)

    pairs = [
        (log_a, log_b)
        for log_a in range(n)
        for log_b in range(n)
    ]

    rng.shuffle(pairs)

    groups = {
        s: []
        for s in range(n)
    }

    for i, (log_a, log_b) in enumerate(pairs):
        groups[i % n].append(
            logit_grid[log_a, log_b, :]
        )

    return groups


def main():
    m = 4
    checkpoint_path = "model_gf16.pt"

    GF = galois.GF(2**m)

    model = GrokTransformer(
        n_vocab=GF.order,
    )

    model.load_state_dict(
        torch.load(
            checkpoint_path,
            map_location="cpu",
        )
    )

    model.eval()

    # Construct the discrete-log coordinate system.
    dlog, antilog = build_discrete_log_tables(GF)

    # Evaluate the model over all nonzero input pairs.
    logit_grid = build_logit_grid(
        model,
        GF,
        antilog,
    )

    # Overall variance across all 225 input pairs.
    overall_variance = (
        logit_grid.reshape(-1, GF.order)
        .var(axis=0)
        .mean()
    )

    # Test the log-sum hypothesis.
    log_sum_groups = build_log_sum_groups(
        logit_grid
    )

    within_log_sum = mean_within_group_variance(
        log_sum_groups
    )

    log_sum_ratio = (
        within_log_sum / overall_variance
    )

    print(
        f"Overall logit variance: "
        f"{overall_variance:.3f}"
    )

    print(
        f"Mean within-group variance "
        f"(same log(a) + log(b)): "
        f"{within_log_sum:.3f}"
    )

    print(
        f"Log-sum / overall variance ratio: "
        f"{log_sum_ratio:.3f}"
    )

    # Random grouping control.
    random_groups = build_random_groups(
        logit_grid,
        seed=0,
    )

    within_random = mean_within_group_variance(
        random_groups
    )

    random_ratio = (
        within_random / overall_variance
    )

    print(
        "\nRandom grouping control:"
    )

    print(
        f"Mean within-group variance: "
        f"{within_random:.3f}"
    )

    print(
        f"Random / overall variance ratio: "
        f"{random_ratio:.3f}"
    )


if __name__ == "__main__":
    main()
