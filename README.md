# CIFAR-10 ResNet18 Classifier + Gradio Demo

An image classifier for the CIFAR-10 dataset built with **transfer learning** on a pretrained ResNet18 backbone, with an interactive **Gradio** web demo for live predictions.

## Overview

- **Task:** 10-class image classification (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck)
- **Approach:** Transfer learning — ImageNet-pretrained ResNet18 with the backbone frozen, fine-tuning only `fc`, `layer4`, and `layer3` at different learning rates
- **Training features:** data augmentation (random flip/rotation), `ReduceLROnPlateau` LR scheduling, early stopping, automatic checkpointing of the best model
- **Demo:** a Gradio interface where you upload an image and get class probabilities in real time

## Project structure

```
.
├── app.py              # Gradio app — loads the trained model and serves predictions
├── training.py          # Full training pipeline (data loading, training loop, evaluation)
├── model_resnet.py       # ResNet18Transfer model definition
├── functions.py          # train_step / test_step / eval_model / calculate_time helpers
├── config.py             # Config dataclass with all hyperparameters
├── class_names.py         # Auto-generated list of CIFAR-10 class names
├── requirements.txt
└── saved_params/          # Trained model weights (created by training.py, not tracked in git)
```

> **Note:** `cifar_data/` (the downloaded dataset) and `saved_params/` (trained weights) are excluded from version control via `.gitignore`. You'll regenerate both by running `training.py`.

## Installation

```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>
pip install -r requirements.txt
```

A GPU with CUDA is recommended for training but not required — the code falls back to CPU automatically.

## Usage

### 1. Train the model

```bash
python training.py
```

This will:
- Download CIFAR-10 into `cifar_data/` automatically (first run only)
- Train the model according to the settings in `config.py`
- Save the best checkpoint to `saved_params/model_experimental_params.pth` (or `model_best_params.pth` if `experimenting = False` in `training.py`)
- Print a final evaluation summary (loss, accuracy, training time)

Key hyperparameters (in `config.py`):

| Parameter | Value |
|---|---|
| Epochs | 10 |
| Batch size | 32 |
| Image size | 128×128 |
| Optimizer | Adam (per-layer learning rates) |
| Scheduler | ReduceLROnPlateau |
| Early stopping patience | 5 epochs |

### 2. Run the demo

Make sure a trained checkpoint exists at `saved_params/model_best_params.pth` (or update the path in `app.py`), then:

```bash
python app.py
```

This launches a local Gradio interface (with a public shareable link, since `share=True`) where you can upload an image and see the predicted class probabilities.

## Model architecture

`ResNet18Transfer` wraps `torchvision.models.resnet18` pretrained on ImageNet:
- The backbone is frozen by default
- Only `fc` (final classification layer), `layer4`, and `layer3` are unfrozen for fine-tuning, each with its own learning rate
- The final fully connected layer is replaced with a 10-class output head

## License 

This project is licensed under the MIT License.
