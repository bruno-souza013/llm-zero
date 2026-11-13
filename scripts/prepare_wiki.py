import argparse
import hashlib
import re
import unicodedata
from html import unescape
from pathlib import Path

import numpy as np
import tiktoken
from datasets import load_dataset
from tqdm import tqdm

DATA_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = DATA_DIR / "wiki_pt"

TAMANHO_LOTE = 256  # artigos tokenizados de cada vez


def limpar_texto(texto: str) -> str:
    texto = unescape(texto)
    texto = unicodedata.normalize("NFC", texto)
    texto = re.sub(r"https?://\S+|www\.\S+", " ", texto)
    texto = "".join(
        caractere
        for caractere in texto
        if caractere in "\n\t"
        or not unicodedata.category(caractere).startswith("C")
    )
    texto = re.sub(r"[ \t]+", " ", texto)
    texto = re.sub(r"\n{3,}", "\n\n", texto)
    return texto.strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artigos", type=int, default=100_000)
    parser.add_argument(
        "--fracao-val",
        type=float,
        default=0.005,
        help="fração dos ARTIGOS que vai para validação (sorteio por artigo)",
    )
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--forcar", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    train_file = OUTPUT_DIR / "train.bin"
    val_file = OUTPUT_DIR / "val.bin"

    if train_file.exists() and val_file.exists() and not args.forcar:
        print(f"{train_file} e {val_file} já existem. Use --forcar para refazer.")
        return

    enc = tiktoken.get_encoding("gpt2")
    eot = enc.eot_token  # 50256, separador de documentos
    rng = np.random.default_rng(args.seed)

    # streaming: não baixa o dataset inteiro. O shuffle com buffer embaralha
    # os artigos (aproximadamente), evitando pegar só o começo do dump.
    dataset = load_dataset(
        "wikimedia/wikipedia",
        "20231101.pt",
        split="train",
        streaming=True,
    ).shuffle(seed=args.seed, buffer_size=10_000)

    tokens_train = 0
    tokens_val = 0
    artigos_val = 0
    artigos_lidos = 0
    documentos_vistos: set[str] = set()

    with train_file.open("wb") as f_train, val_file.open("wb") as f_val:

        def processar_lote(textos: list[str]) -> None:
            nonlocal tokens_train, tokens_val, artigos_val
            for ids in enc.encode_ordinary_batch(textos):
                ids.append(eot)
                arr = np.array(ids, dtype=np.uint16)
                if rng.random() < args.fracao_val:
                    arr.tofile(f_val)
                    tokens_val += arr.size
                    artigos_val += 1
                else:
                    arr.tofile(f_train)
                    tokens_train += arr.size

        lote: list[str] = []
        for exemplo in tqdm(dataset, total=args.artigos):
            texto = limpar_texto(exemplo["text"])
            if len(texto) < 500:
                continue

            hash_texto = hashlib.sha256(texto.encode("utf-8")).hexdigest()
            if hash_texto in documentos_vistos:
                continue
            documentos_vistos.add(hash_texto)

            lote.append(texto)
            artigos_lidos += 1

            if len(lote) >= TAMANHO_LOTE:
                processar_lote(lote)
                lote = []

            if artigos_lidos >= args.artigos:
                break

        if lote:
            processar_lote(lote)

    print(f"Artigos processados: {artigos_lidos:,} ({artigos_val:,} na validação)")
    print(f"Tokens de treino:    {tokens_train:,}")
    print(f"Tokens de validação: {tokens_val:,}")
    print(f"Média: {(tokens_train + tokens_val) / max(1, artigos_lidos):,.0f} tokens/artigo")
    print(f"Arquivos: {train_file} | {val_file}")

    if tokens_val < 1_000:
        print(
            "AVISO: validação muito pequena. Aumente --artigos ou --fracao-val."
        )


if __name__ == "__main__":
    main()