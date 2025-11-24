import argparse
import os
import random
from typing import Iterable, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import wandb


class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_sizes: Iterable[int], num_classes: int) -> None:
        super().__init__()
        layers = []
        dims = [input_dim, *hidden_sizes]
        for in_dim, out_dim in zip(dims, dims[1:]):
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(nn.GELU())
        layers.append(nn.Linear(dims[-1], num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)
        return self.net(x)


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_loaders(data_dir: str, batch_size: int, num_workers: int) -> Tuple[DataLoader, DataLoader]:
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )
    train_ds = datasets.MNIST(root=data_dir, train=True, download=True, transform=transform)
    test_ds = datasets.MNIST(root=data_dir, train=False, download=True, transform=transform)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, test_loader


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    epoch_idx: int,
    log_interval: int,
    global_step: int,
) -> Tuple[float, float, int]:
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for batch_idx, (data, target) in enumerate(loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * data.size(0)
        preds = output.argmax(dim=1)
        total_correct += preds.eq(target).sum().item()
        total_samples += target.size(0)

        if log_interval and (batch_idx + 1) % log_interval == 0:
            wandb.log(
                {
                    "train/batch_loss": loss.item(),
                    "train/step": global_step,
                    "epoch": epoch_idx + 1,
                },
                step=global_step,
            )
        global_step += 1

    avg_loss = total_loss / total_samples
    accuracy = total_correct / total_samples
    return avg_loss, accuracy, global_step


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for data, target in loader:
        data, target = data.to(device), target.to(device)
        output = model(data)
        loss = criterion(output, target)

        total_loss += loss.item() * data.size(0)
        preds = output.argmax(dim=1)
        total_correct += preds.eq(target).sum().item()
        total_samples += target.size(0)

    avg_loss = total_loss / total_samples
    accuracy = total_correct / total_samples
    return avg_loss, accuracy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Basic MNIST trainer with wandb logging.")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=128, help="Mini-batch size.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate for AdamW.")
    parser.add_argument("--weight-decay", type=float, default=0.01, help="Weight decay for AdamW.")
    parser.add_argument(
        "--hidden-sizes",
        type=int,
        nargs="+",
        default=[256, 128],
        help="Sizes of hidden linear layers.",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=os.path.join(os.getcwd(), "data"),
        help="Directory to store MNIST data.",
    )
    parser.add_argument("--num-workers", type=int, default=2, help="DataLoader workers.")
    parser.add_argument("--log-interval", type=int, default=50, help="Batches between wandb logs.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--wandb-project", type=str, default="mnist-pytorch", help="Weights & Biases project name."
    )
    parser.add_argument(
        "--wandb-mode",
        type=str,
        default="online",
        choices=["online", "offline", "disabled"],
        help="wandb run mode.",
    )
    parser.add_argument("--wandb-run-name", type=str, default=None, help="Optional wandb run name.")
    parser.add_argument(
        "--optimizer",
        type=str,
        default="adamw",
        choices=["adamw", "muon", "sgd", "officialmuon", "combined_adamw_muon", "combined_officialmuon_adamw"],
        help="Optimizer choice; 'muon' lets you plug in your custom optimizer from muon.py.",
    )
    return parser.parse_args()


def build_optimizer(
    name: str, params: Iterable[torch.nn.parameter.Parameter], lr: float, weight_decay: float
) -> optim.Optimizer:
    if name == "adamw":
        return optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    if name == "officialmuon":
        return optim.Muon(params, lr=lr, weight_decay=weight_decay)
    if name == "sgd":
        from sgd import SimpleSGD
        return SimpleSGD(params, lr=lr, weight_decay=weight_decay)
    if name == "muon":
        from muon import Muon
        return Muon(params, lr=lr, weight_decay=weight_decay)
    if name == "combined_adamw_muon":
        from muon import CombinedAdamWMuon
        return CombinedAdamWMuon(params, lr=lr, weight_decay=weight_decay)
    if name == "combined_officialmuon_adamw":
        from muon import CombinedOfficialMuonAdamW
        return CombinedOfficialMuonAdamW(params, lr=lr, weight_decay=weight_decay)
    raise ValueError(f"Unknown optimizer {name}")


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, test_loader = get_loaders(args.data_dir, args.batch_size, args.num_workers)

    model = MLP(input_dim=28 * 28, hidden_sizes=args.hidden_sizes, num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = build_optimizer(args.optimizer, model.parameters(), args.lr, args.weight_decay)

    run_name = args.wandb_run_name or f"{args.optimizer}-seed{args.seed}"
    wandb.init(
        project=args.wandb_project,
        mode=args.wandb_mode,
        config=vars(args),
        name=run_name,
    )
    wandb.watch(model, log="all", log_freq=args.log_interval)

    global_step = 0
    for epoch_idx in range(args.epochs):
        epoch_num = epoch_idx + 1
        train_loss, train_acc, global_step = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            epoch_idx,
            args.log_interval,
            global_step,
        )
        val_loss, val_acc = evaluate(model, test_loader, criterion, device)

        wandb.log(
            {
                "epoch": epoch_num,
                "train/loss": train_loss,
                "train/accuracy": train_acc,
                "val/loss": val_loss,
                "val/accuracy": val_acc,
            },
            step=global_step,
        )

        print(
            f"Epoch {epoch_num:03d} "
            f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}"
        )

    wandb.finish()


if __name__ == "__main__":
    main()
