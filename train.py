import numpy as np
import torch
import torch.nn as nn
from tqdm.auto import tqdm

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
    chunk_size=20000,
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
        chunk_size: Number of examples processed per chunk during
            training and evaluation. Smaller values use less memory.

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

    def evaluate(a_data, b_data, y_data):
        """Evaluate loss and accuracy in memory-safe chunks."""
        total_loss = 0.0
        correct = 0
        total = len(y_data)

        with torch.no_grad():
            for start in range(0, total, chunk_size):
                end = min(start + chunk_size, total)

                logits = model(
                    a_data[start:end],
                    b_data[start:end],
                )

                loss = loss_fn(
                    logits,
                    y_data[start:end],
                )

                batch_size = end - start

                total_loss += loss.item() * batch_size
                correct += (
                    logits.argmax(-1) == y_data[start:end]
                ).sum().item()

        return correct / total, total_loss / total

    pbar = tqdm(
        range(n_steps),
        desc=f"Training GF(2^{m})",
    )

    last_train_acc = 0.0
    last_test_acc = 0.0

    for step in pbar:
        model.train()
        opt.zero_grad()

        # Accumulate the full-batch gradient in chunks.
        n_train = len(y_train)
        total_train_loss = 0.0

        for start in range(0, n_train, chunk_size):
            end = min(start + chunk_size, n_train)

            logits = model(
                a_train[start:end],
                b_train[start:end],
            )

            batch_size = end - start

            loss = loss_fn(
                logits,
                y_train[start:end],
            )

            # Weight each chunk so the accumulated gradient
            # matches the full training-set mean loss.
            chunk_loss = loss * (batch_size / n_train)

            chunk_loss.backward()

            total_train_loss += chunk_loss.item()

        opt.step()

        # Evaluate periodically.
        if step % log_every == 0 or step == n_steps - 1:
            model.eval()

            train_acc, train_loss = evaluate(
                a_train,
                b_train,
                y_train,
            )

            test_acc, test_loss = evaluate(
                a_test,
                b_test,
                y_test,
            )

            history["step"].append(step)
            history["train_acc"].append(train_acc)
            history["test_acc"].append(test_acc)
            history["train_loss"].append(train_loss)
            history["test_loss"].append(test_loss)

            last_train_acc = train_acc
            last_test_acc = test_acc

        pbar.set_postfix(
            train_acc=f"{last_train_acc:.3f}",
            test_acc=f"{last_test_acc:.3f}",
        )

    return model, history, GF
