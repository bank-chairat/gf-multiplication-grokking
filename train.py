import numpy as np
import torch
import torch.nn as nn

from dataset import build_gf_mult_dataset
from model import GrokTransformer


def run_experiment(
    m=4,
    train_frac=0.5,
    weight_decay=1.0,
    lr=1e-3,
    n_steps=10000,
    seed=0,
    log_every=100,
    device="cpu",
):
    """Train a transformer to predict multiplication in GF(2^m).

    Args:
        m: Field exponent, giving a field of size 2**m.
        train_frac: Fraction of multiplication pairs used for training.
        weight_decay: AdamW weight decay coefficient.
        lr: Learning rate.
        n_steps: Number of optimization steps.
        seed: Random seed for NumPy and PyTorch.
        log_every: Number of steps between evaluations.
        device: Device on which to run the experiment.

    Returns:
        model: Trained GrokTransformer.
        history: Dictionary containing training/test metrics.
        GF: The Galois field used to construct the dataset.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Build the complete multiplication table.
    a, b, prod, GF = build_gf_mult_dataset(m=m)

    n = 2**m
    N = len(a)

    # Randomly split the complete table into train and test sets.
    idx = np.random.permutation(N)
    split = int(train_frac * N)
    train_idx = idx[:split]
    test_idx = idx[split:]

    # Move the dataset to the selected device.
    a_t = torch.tensor(a, dtype=torch.long, device=device)
    b_t = torch.tensor(b, dtype=torch.long, device=device)
    y_t = torch.tensor(prod, dtype=torch.long, device=device)

    a_train = a_t[train_idx]
    b_train = b_t[train_idx]
    y_train = y_t[train_idx]

    a_test = a_t[test_idx]
    b_test = b_t[test_idx]
    y_test = y_t[test_idx]

    # Initialize the model and optimizer.
    model = GrokTransformer(n_vocab=n).to(device)

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
        betas=(0.9, 0.98),
    )

    loss_fn = nn.CrossEntropyLoss()

    history = {
        "step": [],
        "train_acc": [],
        "test_acc": [],
        "train_loss": [],
        "test_loss": [],
    }

    # Full-batch training over the selected training pairs.
    for step in range(n_steps):
        model.train()

        opt.zero_grad()

        logits = model(a_train, b_train)
        loss = loss_fn(logits, y_train)

        loss.backward()
        opt.step()

        # Evaluate periodically rather than after every optimization step.
        if step % log_every == 0 or step == n_steps - 1:
            model.eval()

            with torch.no_grad():
                train_logits = model(a_train, b_train)
                test_logits = model(a_test, b_test)

                train_acc = (
                    train_logits.argmax(-1) == y_train
                ).float().mean().item()

                test_acc = (
                    test_logits.argmax(-1) == y_test
                ).float().mean().item()

                test_loss = loss_fn(test_logits, y_test).item()

            history["step"].append(step)
            history["train_acc"].append(train_acc)
            history["test_acc"].append(test_acc)
            history["train_loss"].append(loss.item())
            history["test_loss"].append(test_loss)

    return model, history, GF
