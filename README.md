# sps_genai — APAN 5560

FastAPI server for the APAN 5560 assignments. Model code and training scripts live in
`helper_lib/`, and the API in `app/main.py` serves the trained models.

## Run

```bash
docker compose up --build
# open http://localhost:8000/docs
```

## Project structure

The training scripts `train_cnn.py` (Assignment 2) and `train_gan.py` (Assignment 3) sit in
`helper_lib/` next to the `data_loader.py`, `model.py` and `trainer.py` they use.
Run them as modules from the repo root, so the `helper_lib` imports resolve:

```bash
uv run python -m helper_lib.train_cnn   # A2: trains AssignmentCNN on CIFAR10
uv run python -m helper_lib.train_gan   # A3: trains the GAN on MNIST
```

```
sps_genai/
├── app/
│   ├── main.py               FastAPI server (all endpoints)
│   ├── bigram_model.py       bigram text generation (earlier class activity)
│   └── embedding_model.py    word embeddings (earlier class activity)
├── helper_lib/               model code + training scripts (shared by the API)
│   ├── data_loader.py        CIFAR10 loader (A2), MNIST loader (A3)
│   ├── model.py              AssignmentCNN (A2), Generator / Discriminator (A3)
│   ├── trainer.py            train_model (A2), train_gan (A3)
│   ├── evaluator.py          evaluate_model (A2)
│   ├── generator.py          generate_gan_images (A3)
│   ├── utils.py              save_model / load_model (A2, A3)
│   ├── train_cnn.py          A2 training script
│   └── train_gan.py          A3 training script
├── models/
│   ├── cnn_cifar10.pth       trained AssignmentCNN (A2)
│   └── gan_mnist_generator.pth  trained GAN generator (A3)
├── Dockerfile
└── docker-compose.yml
```

## Assignment 3 — GAN on MNIST

| Part | Where |
|---|---|
| 1. GAN architecture | `Generator`, `Discriminator` in `helper_lib/model.py` (shape check: `uv run python -m helper_lib.model`) |
| 2. Training | `helper_lib/data_loader.py` (`dataset="MNIST"`), `train_gan` in `helper_lib/trainer.py`, `helper_lib/train_gan.py` |
| 2. Deployment | `GET /generate_digits` in `app/main.py`, uses `generate_gan_images` in `helper_lib/generator.py` |

Retrain: `uv run python -m helper_lib.train_gan` (saves `models/gan_mnist_generator.pth`).

## Endpoints

- `GET /generate_digits?num_samples=16&seed=1` — MNIST digits from the GAN generator (PNG, A3)
- `POST /classify` — CIFAR10 image classification, upload an image (A2)
- `POST /generate` — bigram text generation
- `POST /embedding`, `POST /similarity` — word embeddings
