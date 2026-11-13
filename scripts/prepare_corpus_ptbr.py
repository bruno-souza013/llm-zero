import argparse
import hashlib
import json
import re
import unicodedata
from html import unescape
from pathlib import Path

import numpy as np
import tiktoken
from datasets import load_dataset
from tqdm import tqdm


DATASET_NAME = "bratao/corpus-ptbr-v1"
TOKENIZER_NAME = "gpt2"
TAMANHO_LOTE = 256


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
    parser = argparse.ArgumentParser(
        description="Prepara um subconjunto do Corpus PT-BR para treino."
    )
    parser.add_argument("--documentos", type=int, default=100_000)
    parser.add_argument("--fracao-val", type=float, default=0.005)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data/corpus_ptbr_100k"),
    )
    parser.add_argument("--forcar", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    train_file = args.out_dir / "train.bin"
    val_file = args.out_dir / "val.bin"
    metadata_file = args.out_dir / "metadata.json"

    if (
        train_file.exists()
        and val_file.exists()
        and metadata_file.exists()
        and not args.forcar
    ):
        print(f"Arquivos já existem em {args.out_dir}. Use --forcar para refazer.")
        return

    if not 0 < args.fracao_val < 1:
        raise ValueError("--fracao-val deve estar entre 0 e 1")
    if args.documentos < 1:
        raise ValueError("--documentos deve ser positivo")

    enc = tiktoken.get_encoding(TOKENIZER_NAME)
    eot = enc.eot_token
    rng = np.random.default_rng(args.seed)
    dataset = load_dataset(
        DATASET_NAME,
        split="train",
        streaming=True,
    ).shuffle(seed=args.seed, buffer_size=10_000)

    tokens_train = 0
    tokens_val = 0
    documentos_train = 0
    documentos_val = 0
    documentos_lidos = 0
    documentos_vistos: set[str] = set()

    with train_file.open("wb") as f_train, val_file.open("wb") as f_val:

        def processar_lote(textos: list[str]) -> None:
            nonlocal tokens_train, tokens_val
            nonlocal documentos_train, documentos_val

            for ids in enc.encode_ordinary_batch(textos):
                ids.append(eot)
                array = np.array(ids, dtype=np.uint16)
                if rng.random() < args.fracao_val:
                    array.tofile(f_val)
                    tokens_val += array.size
                    documentos_val += 1
                else:
                    array.tofile(f_train)
                    tokens_train += array.size
                    documentos_train += 1

        lote: list[str] = []
        for exemplo in tqdm(dataset, total=args.documentos):
            texto = limpar_texto(exemplo["text"])
            if len(texto) < 500:
                continue

            hash_texto = hashlib.sha256(texto.encode("utf-8")).hexdigest()
            if hash_texto in documentos_vistos:
                continue
            documentos_vistos.add(hash_texto)

            lote.append(texto)
            documentos_lidos += 1
            if len(lote) >= TAMANHO_LOTE:
                processar_lote(lote)
                lote = []

            if documentos_lidos >= args.documentos:
                break

        if lote:
            processar_lote(lote)

    metadata = {
        "dataset": DATASET_NAME,
        "split": "train",
        "documents_requested": args.documentos,
        "documents_processed": documentos_lidos,
        "documents_train": documentos_train,
        "documents_val": documentos_val,
        "tokens_train": tokens_train,
        "tokens_val": tokens_val,
        "validation_fraction": args.fracao_val,
        "seed": args.seed,
        "tokenizer": TOKENIZER_NAME,
        "vocab_size": enc.n_vocab,
        "eot_token": eot,
        "minimum_characters": 500,
    }
    with metadata_file.open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2, ensure_ascii=False)

    print(f"Documentos processados: {documentos_lidos:,}")
    print(f"Documentos de treino:   {documentos_train:,}")
    print(f"Documentos de validação: {documentos_val:,}")
    print(f"Tokens de treino:       {tokens_train:,}")
    print(f"Tokens de validação:    {tokens_val:,}")
    print(f"Arquivos salvos em:     {args.out_dir}")


if __name__ == "__main__":
    main()
