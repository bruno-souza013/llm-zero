from dataclasses import dataclass


@dataclass
class ModelConfig:
 
    vocab_size: int = 50257

    #tamanho máximo do contexto
    block_size: int = 256

    #arquitetura do transformer
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384
    dropout: float = 0.1
    bias: bool = True


@dataclass
class TrainConfig:
    data_dir: str = "data/corpus_ptbr_100k"
    out_dir: str = "experiments/corpus-ptbr-100k-v1"

    device: str = "cuda"
    batch_size: int = 8
    gradient_accumulation_steps: int = 4

    max_iters: int = 20
    eval_interval: int = 500
    eval_iters: int = 50

    learning_rate: float = 3e-4
    min_lr: float = 6e-5
    warmup_iters: int = 200

    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0

    resume: bool = False
    seed: int = 1337