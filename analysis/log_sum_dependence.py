import numpy as np
import torch
import galois

from model import GrokTransformer


def build_discrete_log_tables(GF):
    """Build discrete-log and antilog tables for nonzero field elements."""
    primitive = GF.primitive_element
    q_minus_1 = GF.order - 1

    dlog = {}
    value = GF(1)

    for k in range(q_minus_1):
        dlog[int(value)] = k
        value *= primitive

    antilog = {
        log: element
        for element, log in dlog.items()
    }

    return dlog, antilog


def build_logit_grid(
    model,
    GF,
    antilog,
    device="cpu",
    batch_size=8192,
):
    """Evaluate the model on every pair of nonzero field elements.

    The first two dimensions are indexed by log(a) and log(b).
    Evaluation is batched so this works efficiently for larger fields.
    """
    q_minus_1 = GF.order - 1
    n_classes = GF.order

    log_a_grid, log_b_grid = np.meshgrid(
        np.arange(q_minus_1),
        np.arange(q_minus_1),
        indexing="ij",
    )

    log_a_flat = log_a_grid.ravel()
    log_b_flat = log_b_grid.ravel()

    antilog_array = np.array(
        [antilog[k] for k in range(q_minus_1)]
    )

    a_vals = antilog_array[log_a_flat]
    b_vals = antilog_array[log_b_flat]

    n_pairs = len(a_vals)

    all_logits = np.zeros(
        (n_pairs, n_classes),
        dtype=np.float32,
    )

    model.eval()

    with torch.no_grad():
        for start in range(0, n_pairs, batch_size):
            end = min(start + batch_size, n_pairs)

            a_tensor = torch.tensor(
                a_vals[start:end],
                dtype=torch.long,
                device=device,
            )

            b_tensor = torch.tensor(
                b_vals[start:end],
                dtype=torch.long,
                device=device,
            )

            logits = model(
                a_tensor,
                b_tensor,
            )

            all_logits[start:end] = (
                logits.cpu().numpy()
            )

    return all_logits.reshape(
        q_minus_1,
        q_minus_1,
        n_classes,
    )


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
    """Group pairs by log(a) + log(b) mod (q - 1)."""
    q_minus_1 = logit_grid.shape[0]

    groups = {
        s: []
        for s in range(q_minus_1)
    }

    for log_a in range(q_minus_1):
        for log_b in range(q_minus_1):
            log_sum = (
                log_a + log_b
            ) % q_minus_1

            groups[log_sum].append(
                logit_grid[
                    log_a,
                    log_b,
                    :,
                ]
            )

    return groups


def build_random_groups(
    logit_grid,
    seed=0,
):
    """Build a random control grouping with the same group sizes."""
    q_minus_1 = logit_grid.shape[0]

    rng = np.random.RandomState(seed)

    pairs = [
        (log_a, log_b)
        for log_a in range(q_minus_1)
        for log_b in range(q_minus_1)
    ]

    rng.shuffle(pairs)

    groups = {
        s: []
        for s in range(q_minus_1)
    }

    for i, (log_a, log_b) in enumerate(pairs):
        groups[i % q_minus_1].append(
            logit_grid[
                log_a,
                log_b,
                :,
            ]
        )

    return groups


def diagonal_ratio_test(
    model,
    GF,
    device="cpu",
    batch_size=8192,
):
    """Measure how strongly logits depend on log(a) + log(b).

    The log-table hypothesis predicts that for nonzero field elements:

        log(a * b) = log(a) + log(b) mod (q - 1)

    Therefore, if the model has learned a functionally equivalent
    log/antilog computation, pairs with the same log(a) + log(b)
    should have similar output-logit vectors.

    Returns:
        log_sum_ratio:
            Within-group variance / overall variance for the
            meaningful log-sum grouping.

        random_ratio:
            The same quantity for a random control grouping.
    """
    dlog, antilog = build_discrete_log_tables(GF)

    logit_grid = build_logit_grid(
        model=model,
        GF=GF,
        antilog=antilog,
        device=device,
        batch_size=batch_size,
    )

    # Overall variance across all nonzero input pairs.
    overall_variance = (
        logit_grid
        .reshape(-1, GF.order)
        .var(axis=0)
        .mean()
    )

    # Test the log-sum hypothesis.
    log_sum_groups = build_log_sum_groups(
        logit_grid
    )

    within_log_sum = (
        mean_within_group_variance(
            log_sum_groups
        )
    )

    log_sum_ratio = (
        within_log_sum
        / overall_variance
    )

    # Random grouping control.
    random_groups = build_random_groups(
        logit_grid,
        seed=0,
    )

    within_random = (
        mean_within_group_variance(
            random_groups
        )
    )

    random_ratio = (
        within_random
        / overall_variance
    )

    return (
        overall_variance,
        within_log_sum,
        log_sum_ratio,
        within_random,
        random_ratio,
    )


def main():
    # Change m to analyze a different field size.
    m = 4

    # Change this to the checkpoint you want to analyze.
    checkpoint_path = "model_gf16.pt"

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    GF = galois.GF(2**m)

    model = GrokTransformer(
        n_vocab=GF.order,
    ).to(device)

    model.load_state_dict(
        torch.load(
            checkpoint_path,
            map_location=device,
        )
    )

    model.eval()

    (
        overall_variance,
        within_log_sum,
        log_sum_ratio,
        within_random,
        random_ratio,
    ) = diagonal_ratio_test(
        model=model,
        GF=GF,
        device=device,
    )

    print(f"Field: GF({GF.order})")
    print(f"Device: {device}")

    print(
        f"\nOverall logit variance: "
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
