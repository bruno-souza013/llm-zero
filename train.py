import argparse
import csv
import json
import math
import os
import random
import time
from contextlib import nullcontext

import numpy as np
import torch

from config import ModelConfig, TrainConfig
from model import LLM


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Treina o modelo usando os valores padrão de config.py."
    )
    parser.add_argument("--data-dir", type=str)
    parser.add_argument("--out-dir", type=str)
    parser.add_argument("--max-iters", type=int)
    parser.add_argument("--eval-interval", type=int)
    parser.add_argument("--eval-iters", type=int)
    parser.add_argument("--device", type=str)
    parser.add_argument(
        "--resume",
        action="store_true",
        default=None,
        help="Retoma o treinamento a partir de last.pt.",
    )
    return parser.parse_args()


def apply_overrides(
    train_config: TrainConfig,
    args: argparse.Namespace,
) -> None:
    overrides = {
        "data_dir": args.data_dir,
        "out_dir": args.out_dir,
        "max_iters": args.max_iters,
        "eval_interval": args.eval_interval,
        "eval_iters": args.eval_iters,
        "device": args.device,
        "resume": args.resume,
    }
    for name, value in overrides.items():
        if value is not None:
            setattr(train_config, name, value)


def get_batch(
    split: str,
    data_config: TrainConfig,
    model_config: ModelConfig,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    filename = "train.bin" if split == "train" else "val.bin"
    path = os.path.join(data_config.data_dir, filename)
    #O memmap é recriado a cada chamada de propósito (evita vazamento de memória).
    data = np.memmap(path, dtype=np.uint16, mode="r")

    required_tokens = model_config.block_size + 1
    if len(data) < required_tokens:
        raise ValueError(
            f"{path} precisa ter pelo menos {required_tokens} tokens; "
            f"encontrados {len(data)}"
        )

    starts = torch.randint(
        len(data) - model_config.block_size,
        (data_config.batch_size,),
    )
    x = torch.stack(
        [
            torch.from_numpy(
                data[start : start + model_config.block_size].astype(np.int64)
            )
            for start in starts
        ]
    )
    y = torch.stack(
        [
            torch.from_numpy(
                data[start + 1 : start + 1 + model_config.block_size].astype(
                    np.int64
                )
            )
            for start in starts
        ]
    )
    return x.to(device), y.to(device)


@torch.no_grad()
def estimate_loss(
    model: LLM,
    data_config: TrainConfig,
    model_config: ModelConfig,
    device: torch.device,
    context,
) -> dict[str, float]:
    result: dict[str, float] = {}
    model.eval()

    for split in ("train", "val"):
        losses = torch.zeros(data_config.eval_iters)
        for iteration in range(data_config.eval_iters):
            x, y = get_batch(split, data_config, model_config, device)
            #mesma precisão do treino, para as losses serem comparáveis.
            with context:
                _, loss = model(x, y)
            if loss is None:
                raise RuntimeError("O modelo não retornou loss durante a avaliação")
            losses[iteration] = loss.item()
        result[split] = losses.mean().item()

    model.train()
    return result


def get_lr(iteration: int, train_config: TrainConfig) -> float:
    """Warmup linear seguido de decaimento cosseno até min_lr."""
    max_lr = train_config.learning_rate
    min_lr = getattr(train_config, "min_lr", max_lr / 10)
    warmup_iters = getattr(train_config, "warmup_iters", 100)

    if iteration < warmup_iters:
        return max_lr * (iteration + 1) / (warmup_iters + 1)
    if iteration >= train_config.max_iters:
        return min_lr
    progress = (iteration - warmup_iters) / max(1, train_config.max_iters - warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr + coeff * (max_lr - min_lr)


def build_optimizer(model: LLM, train_config: TrainConfig) -> torch.optim.AdamW:
    """AdamW com weight decay só em matrizes (dim >= 2); biases e LayerNorm ficam sem."""
    params = [p for p in model.parameters() if p.requires_grad]
    decay_params = [p for p in params if p.dim() >= 2]
    no_decay_params = [p for p in params if p.dim() < 2]
    groups = [
        {"params": decay_params, "weight_decay": train_config.weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(
        groups,
        lr=train_config.learning_rate,
        betas=(train_config.beta1, train_config.beta2),
    )


def capture_rng_state() -> dict:
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: dict) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and "cuda" in state:
        torch.cuda.set_rng_state_all(state["cuda"])


def main() -> None:
    model_config = ModelConfig()
    train_config = TrainConfig()
    apply_overrides(train_config, parse_args())

    if train_config.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            "TrainConfig.device está como 'cuda', mas CUDA não está disponível. "
            "Altere para device='cpu' em config.py."
        )

    device = torch.device(train_config.device)
    torch.manual_seed(train_config.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed(train_config.seed)

    # Precisão mista: bfloat16 se a GPU suportar (sem scaler); senão float16 + GradScaler.
    # is_bf16_supported() também retorna True com bf16 *emulado* (T4), que é lento.
    # Por isso exigimos compute capability >= 8.0 (Ampere ou mais novo).
    use_amp = device.type == "cuda"
    native_bf16 = use_amp and torch.cuda.get_device_capability(device)[0] >= 8
    use_fp16 = use_amp and not native_bf16
    amp_dtype = torch.float16 if use_fp16 else torch.bfloat16
    context = (
        torch.autocast(device_type=device.type, dtype=amp_dtype)
        if use_amp
        else nullcontext()
    )
    scaler = torch.amp.GradScaler("cuda", enabled=use_fp16)

    os.makedirs(train_config.out_dir, exist_ok=True)
    best_ckpt_path = os.path.join(train_config.out_dir, "best.pt")
    last_ckpt_path = os.path.join(train_config.out_dir, "last.pt")
    metrics_path = os.path.join(train_config.out_dir, "metrics.csv")
    metrics_mode = "a" if getattr(train_config, "resume", False) else "w"
    metrics_file = open(metrics_path, metrics_mode, newline="", encoding="utf-8")
    metrics_writer = csv.DictWriter(
        metrics_file,
        fieldnames=[
            "step",
            "train_loss",
            "val_loss",
            "train_perplexity",
            "val_perplexity",
            "learning_rate",
            "elapsed_seconds",
            "gpu_memory_mb",
        ],
    )
    if os.path.getsize(metrics_path) == 0:
        metrics_writer.writeheader()
    training_started_at = time.perf_counter()

    model = LLM(model_config).to(device)
    optimizer = build_optimizer(model, train_config)
    print(f"Parâmetros: {model.num_parameters():,}")
    print(f"Dispositivo: {device} | precisão: {amp_dtype if use_amp else 'float32'}")

    start_iter = 0
    best_val_loss = math.inf

    #retomada opcional: defina resume=True no TrainConfig.
    if getattr(train_config, "resume", False) and os.path.exists(last_ckpt_path):
        ckpt = torch.load(
            last_ckpt_path,
            map_location=device,
            weights_only=False,
        )
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        if use_fp16 and ckpt.get("scaler") is not None:
            scaler.load_state_dict(ckpt["scaler"])
        if "rng_state" in ckpt:
            restore_rng_state(ckpt["rng_state"])
        start_iter = ckpt["step"] + 1
        best_val_loss = ckpt["val_loss"]
        print(f"Retomando do passo {start_iter} (val loss {best_val_loss:.4f})")

    grad_clip = getattr(train_config, "grad_clip", 1.0)
    accum_steps = train_config.gradient_accumulation_steps

    def checkpoint_payload(step: int, val_loss: float) -> dict:
        return {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict() if use_fp16 else None,
            "model_config": model_config.__dict__,
            "train_config": train_config.__dict__,
            "step": step,
            "val_loss": val_loss,
            "rng_state": capture_rng_state(),
        }

    def save_last_checkpoint(step: int, val_loss: float) -> None:
        torch.save(
            checkpoint_payload(step, val_loss),
            last_ckpt_path,
        )

    def evaluate_and_checkpoint(step: int) -> None:
        nonlocal best_val_loss
        losses = estimate_loss(model, train_config, model_config, device, context)
        gpu_memory_mb = 0.0
        if device.type == "cuda":
            gpu_memory_mb = torch.cuda.max_memory_allocated(device) / (1024**2)
        metrics_writer.writerow(
            {
                "step": step,
                "train_loss": losses["train"],
                "val_loss": losses["val"],
                "train_perplexity": math.exp(losses["train"]),
                "val_perplexity": math.exp(losses["val"]),
                "learning_rate": optimizer.param_groups[0]["lr"],
                "elapsed_seconds": time.perf_counter() - training_started_at,
                "gpu_memory_mb": gpu_memory_mb,
            }
        )
        metrics_file.flush()
        print(
            f"step {step}: "
            f"train loss {losses['train']:.4f}, "
            f"val loss {losses['val']:.4f}"
        )
        if losses["val"] < best_val_loss:
            best_val_loss = losses["val"]
            torch.save(
                checkpoint_payload(step, best_val_loss),
                best_ckpt_path,
            )

    model.train()
    for iteration in range(start_iter, train_config.max_iters):
        lr = get_lr(iteration, train_config)
        for group in optimizer.param_groups:
            group["lr"] = lr

        if iteration % train_config.eval_interval == 0:
            evaluate_and_checkpoint(iteration)

        optimizer.zero_grad(set_to_none=True)
        loss_accum = 0.0
        for _ in range(accum_steps):
            x, y = get_batch("train", train_config, model_config, device)
            with context:
                _, loss = model(x, y)
            if loss is None:
                raise RuntimeError("O modelo não retornou loss durante o treino")
            loss = loss / accum_steps
            loss_accum += loss.item()
            scaler.scale(loss).backward()

        #unscale_ antes do clip: o clip precisa ver os gradientes reais.
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()

        if iteration % 10 == 0:
            print(f"step {iteration}: batch loss {loss_accum:.4f} | lr {lr:.2e}")
        if iteration % train_config.eval_interval == 0:
            save_last_checkpoint(iteration, best_val_loss)

    #avaliação final: garante que os últimos passos também possam virar checkpoint.
    evaluate_and_checkpoint(train_config.max_iters)
    save_last_checkpoint(train_config.max_iters, best_val_loss)
    metrics_file.close()
    with open(
        os.path.join(train_config.out_dir, "run.json"),
        "w",
        encoding="utf-8",
    ) as run_file:
        json.dump(
            {
                "model_config": model_config.__dict__,
                "train_config": train_config.__dict__,
                "parameters": model.num_parameters(),
                "device": str(device),
                "precision": str(amp_dtype if use_amp else "float32"),
                "duration_seconds": time.perf_counter() - training_started_at,
                "best_val_loss": best_val_loss,
                "best_val_perplexity": math.exp(best_val_loss),
            },
            run_file,
            indent=2,
        )


if __name__ == "__main__":
    main()