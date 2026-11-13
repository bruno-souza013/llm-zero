import argparse
from pathlib import Path

import torch
import tiktoken

from model import LLM
from config import ModelConfig


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=Path("out"))
    parser.add_argument("--prompt", default="A história do Brasil")
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=100)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint_path = args.out_dir / "best.pt"
    if not checkpoint_path.exists():
        checkpoint_path = args.out_dir / "ckpt.pt"

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint não encontrado: {checkpoint_path}"
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=DEVICE,
    )

    model_config = ModelConfig(**checkpoint["model_config"])
    model = LLM(model_config)
    model.load_state_dict(checkpoint["model"])
    model.to(DEVICE)
    model.eval()

    tokenizer = tiktoken.get_encoding("gpt2")

    token_ids = tokenizer.encode(args.prompt)
    input_ids = torch.tensor(
        token_ids,
        dtype=torch.long,
        device=DEVICE,
    ).unsqueeze(0)

    with torch.no_grad():
        generated_ids = model.generate(
            input_ids,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
        )

    generated_text = tokenizer.decode(generated_ids[0].tolist())
    print(generated_text)


if __name__ == "__main__":
    main()