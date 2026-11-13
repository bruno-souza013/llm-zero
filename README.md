# LLM Zero

Um mini modelo de linguagem em português, implementado do zero para estudar
como funciona um modelo causal no estilo GPT.

O projeto percorre o pipeline completo:

```text
Wikipédia em português
        ↓
limpeza e tokenização
        ↓
train.bin / val.bin
        ↓
Transformer causal
        ↓
treinamento em CUDA
        ↓
checkpoint e métricas
        ↓
geração de texto
```

> **Status:** primeira versão funcional, com dois experimentos registrados.

## O que foi implementado

- preparação em streaming de artigos da Wikipédia em português;
- limpeza de URLs, caracteres de controle e espaços;
- remoção de artigos curtos e duplicados;
- tokenização BPE com o encoding `gpt2` do `tiktoken`;
- separação de documentos com o token `<|endoftext|>`;
- Transformer causal implementado em PyTorch;
- atenção causal, MLP, LayerNorm e embeddings de posição;
- treinamento com AdamW, warmup, decaimento cosseno e gradient clipping;
- mixed precision em GPU CUDA;
- checkpoints com possibilidade de retomada;
- métricas em CSV/JSON e gráfico do treinamento;
- geração de texto por prompt.

## Resultados

Os dois experimentos usaram o mesmo dataset, a mesma arquitetura e a mesma
seed. A v2 aumentou o batch e o número de etapas.

| Experimento | Etapas | Batch | Parâmetros | Melhor val loss | Perplexidade | Tempo |
|---|---:|---:|---:|---:|---:|---:|
| v1 | 10.000 | 4 × 4 | 30.044.544 | 3,1130 | 22,49 | 23,45 min |
| v2 | 30.000 | 8 × 4 | 30.044.544 | **2,5508** | **12,82** | 112,17 min |

### Gráficos de treinamento

#### v1

![Métricas de treinamento da v1](out-wiki-100k-v1/training_metrics.png)

#### v2

![Métricas de treinamento da v2](out-wiki-100k-v2/training_metrics%20%281%29.png)

Configuração comum dos modelos:

```text
vocab_size: 50.257
block_size: 256
n_layer: 6
n_head: 6
n_embd: 384
dropout: 0,1
```

Na v2, o melhor resultado ocorreu na etapa `28.000`. O checkpoint salvo em
`out-wiki-100k-v2/best.pt` corresponde ao melhor `val_loss`, enquanto a última
avaliação da etapa 30.000 teve `val_loss` 2,6701.

Os relatórios completos são gerados em cada pasta de saída:

```text
benchmark.json
metrics.csv
run.json
training_metrics.png
```

## Estrutura

```text
llm-zero/
├── config.py              # configurações do modelo e do treino
├── model.py               # Transformer causal
├── train.py               # loop de treinamento e checkpoints
├── generate.py            # geração por prompt
├── benchmark.py           # relatório e gráfico das métricas
├── data/
│   ├── prepare.py         # limpeza, download e tokenização
│   └── wiki_pt/
│       ├── train.bin
│       └── val.bin
├── out-wiki-100k-v1/      # artefatos do experimento v1
├── out-wiki-100k-v2/      # artefatos do experimento v2
└── README.md
```

Checkpoints e saídas de treino são ignorados pelo Git por serem arquivos
grandes. Os diretórios `out-*` mencionados acima representam resultados locais
dos experimentos.

## Requisitos

- Python 3.10 ou superior;
- PyTorch;
- NumPy;
- `tiktoken`;
- `datasets`;
- `tqdm`;
- `matplotlib` para gerar gráficos;
- GPU CUDA opcional, mas recomendada.

Instalação:

```powershell
python -m pip install torch numpy tiktoken datasets tqdm matplotlib
```

No ambiente virtual usado neste projeto:

```powershell
.\venv\Scripts\python.exe -m pip install torch numpy tiktoken datasets tqdm matplotlib
```

## Preparando os dados

O [data/prepare.py](data/prepare.py) cria os arquivos binários usados pelo
treinamento. Ele usa streaming e não precisa carregar toda a Wikipédia na
memória.

Para preparar 100.000 artigos:

```powershell
.\venv\Scripts\python.exe data\prepare.py --artigos 100000 --forcar
```

Os arquivos serão criados em:

```text
data/wiki_pt/train.bin
data/wiki_pt/val.bin
```

Opções úteis:

```powershell
.\venv\Scripts\python.exe data\prepare.py --help
```

O script também aceita `--fracao-val` e `--seed`. A validação é separada por
artigo, evitando dividir um mesmo artigo entre treino e validação.

## Configuração do modelo

As configurações ficam no [config.py](config.py). A configuração usada nos
experimentos registrados é:

```python
vocab_size = 50257
block_size = 256
n_layer = 6
n_head = 6
n_embd = 384
dropout = 0.1
bias = True
```

Para uma GPU CUDA:

```python
device = "cuda"
```

Para CPU:

```python
device = "cpu"
```

Use uma pasta `out_dir` diferente em cada experimento para não misturar
checkpoints e métricas.

## Treinamento

Depois de preparar os dados:

```powershell
.\venv\Scripts\python.exe train.py
```

O treinamento salva na pasta definida por `out_dir`:

```text
best.pt       # melhor checkpoint pela val loss
last.pt       # último estado salvo para retomada
metrics.csv   # losses, perplexidade, learning rate e tempo
run.json      # configuração e resumo da execução
```

O checkpoint inclui pesos do modelo, estado do AdamW, scaler de mixed
precision, estados aleatórios, configurações e etapa do resultado. Para
retomar um treino, use o mesmo `out_dir` e defina:

```python
resume = True
```

## Benchmark

Depois do treinamento, gere o relatório:

```powershell
.\venv\Scripts\python.exe benchmark.py --out-dir out-wiki-100k-v2
```

O comando cria:

```text
out-wiki-100k-v2/benchmark.json
out-wiki-100k-v2/training_metrics.png
```

O relatório registra melhor etapa, loss, perplexidade, duração e memória máxima
da GPU. O gráfico mostra as curvas de treino/validação e o learning rate.

## Geração de texto

O [generate.py](generate.py) carrega `best.pt` por padrão e permite escolher o
diretório do checkpoint e o prompt:

```powershell
.\venv\Scripts\python.exe generate.py `
    --out-dir out-wiki-100k-v2 `
    --prompt "A história do Brasil" `
    --max-new-tokens 200 `
    --temperature 0.8 `
    --top-k 100
```

Para comparar com a v1:

```powershell
.\venv\Scripts\python.exe generate.py --out-dir out-wiki-100k-v1 --prompt "A história do Brasil"
.\venv\Scripts\python.exe generate.py --out-dir out-wiki-100k-v2 --prompt "A história do Brasil"
```

O tokenizer usado na geração precisa ser o mesmo usado na preparação dos
dados: `tiktoken.get_encoding("gpt2")`.

## Comparação reproduzível

Para comparar versões, mantenha os mesmos prompts, `temperature`, `top_k` e
`max_new_tokens`. Alguns prompts usados no projeto:

```text
A história do Brasil
Os jogadores brasileiros que participaram da Copa do Mundo
A população brasileira
A cidade de São Paulo
O desenvolvimento da tecnologia no Brasil
```

Avalie as gerações por fluidez, coerência, relevância ao prompt, gramática e
repetição. A perplexidade é útil para comparação quantitativa, mas não mede
sozinha a qualidade factual ou a utilidade como chatbot.

## Execução no Kaggle

Ative uma GPU nas configurações do Notebook e clone o repositório:

```python
!git clone https://github.com/SEU_USUARIO/SEU_REPOSITORIO.git
%cd SEU_REPOSITORIO
!pip install -q tiktoken matplotlib
```

Confirme o ambiente:

```python
import torch

print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))
```

Depois execute:

```python
!python train.py
!python benchmark.py --out-dir out-wiki-100k-v2
```

O checkpoint pode ser salvo como output da versão do Notebook e baixado para
uso local.

## Limitações

- O modelo foi treinado para continuar textos, não para seguir instruções.
- O tokenizer GPT-2 não foi treinado especificamente para português.
- O corpus é enciclopédico e não representa conversas ou instruções.
- O modelo pode repetir ou combinar fatos de forma incorreta.
- Perplexidade não garante coerência, factualidade ou ausência de viés.

## Próximos passos

- avaliar um tokenizer mais adequado para português;
- ampliar e diversificar o corpus;
- adicionar avaliação automática de repetição e diversidade;
- comparar diferentes tamanhos de contexto;
- criar uma etapa de fine-tuning instrucional;
- adicionar testes automatizados;
- registrar hardware, versão do código e configuração em cada experimento.

## Créditos e referências

Este é um projeto educacional próprio, inspirado na arquitetura e no fluxo de
treinamento do **nanoGPT**, criado por **Andrej Karpathy**:

- [karpathy/nanoGPT](https://github.com/karpathy/nanoGPT)
- [Andrej Karpathy](https://github.com/karpathy)

Fontes e ferramentas:

- **Wikimedia Foundation** — [Wikimedia Downloads](https://dumps.wikimedia.org/);
- **Hugging Face Datasets** — dataset
  [wikimedia/wikipedia](https://huggingface.co/datasets/wikimedia/wikipedia);
- **OpenAI** — biblioteca [tiktoken](https://github.com/openai/tiktoken);
- **PyTorch** — framework de treinamento
  [pytorch.org](https://pytorch.org/).

Consulte as licenças e os termos de uso de cada fonte antes de redistribuir o
dataset ou publicar um modelo treinado com esses dados.

