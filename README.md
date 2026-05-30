# ann-fashion-mnist-pytorch-gpu-optimizied-optuna
# 👗 Fashion MNIST Classifier — ANN with PyTorch & Optuna

A configurable Artificial Neural Network (ANN) trained on the [Fashion MNIST](https://github.com/zalandoresearch/fashion-mnist) dataset, with automated hyperparameter tuning via **Optuna** and GPU acceleration via **PyTorch**.

---

## 📌 Overview

This project builds a fully connected neural network to classify fashion items (T-shirts, trousers, shoes, etc.) into 10 categories. Instead of manually tuning hyperparameters, **Optuna** runs a search over a defined space to find the best configuration automatically.

---

## 🗂️ Dataset

- **Source**: `fashion-mnist_train.csv` (Kaggle / Zalando)
- **Size**: 60,000 training samples, 28×28 grayscale images
- **Classes**: 10 (T-shirt, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle boot)
- Pixel values are normalized to `[0, 1]` before training

---

## 🧠 Model Architecture

A flexible `MyNN` class builds a sequential network dynamically:

| Component       | Details                                      |
|-----------------|----------------------------------------------|
| Input layer     | 784 neurons (28×28 flattened)                |
| Hidden layers   | Variable count, each with Linear → BatchNorm → ReLU → Dropout |
| Output layer    | 10 neurons (one per class)                   |
| Loss function   | Cross-Entropy Loss                           |

---

## ⚙️ Hyperparameter Search Space (Optuna)

| Hyperparameter      | Range / Options                        |
|---------------------|----------------------------------------|
| `num_hidden_layers` | 1 – 5                                  |
| `neurons_per_layer` | 8 – 128 (step 8)                       |
| `epochs`            | 10 – 50 (step 10)                      |
| `learning_rate`     | 1e-5 – 1e-1 (log scale)               |
| `dropout_rate`      | 0.1 – 0.5 (step 0.1)                  |
| `batch_size`        | 16, 32, 64, 128                        |
| `optimizer`         | Adam, SGD, RMSprop                     |
| `weight_decay`      | 1e-5 – 1e-3 (log scale)               |

Optuna runs **10 trials** by default, maximizing test accuracy.

---

## 🚀 Getting Started

### Prerequisites

```bash
pip install torch torchvision pandas scikit-learn matplotlib optuna
```

### Running in Google Colab (recommended)

1. Upload `fashion-mnist_train.csv` when prompted
2. Run all cells — GPU will be used automatically if available
3. Optuna will output the best trial's accuracy and parameters

### Running Locally

1. Place `fashion-mnist_train.csv` in the project directory
2. Remove the `google.colab` file upload block and replace with:
   ```python
   df = pd.read_csv('fashion-mnist_train.csv')
   ```
3. Run the script:
   ```bash
   python ann_fashion_mnist_pytorch_gpu_optimizied_optuna.py
   ```

---

## 📊 Results

After optimization, retrieve the best results with:

```python
print("Best accuracy:", study.best_value)
print("Best params:",  study.best_params)
```

---

## 🛠️ Tech Stack

- **Python 3.x**
- **PyTorch** — model definition, training, GPU support
- **Optuna** — hyperparameter optimization
- **scikit-learn** — train/test split
- **pandas** — data loading
- **matplotlib** — image visualization

---

## 📁 Project Structure

```
├── ann_fashion_mnist_pytorch_gpu_optimizied_optuna.py  # Main script
├── fashion-mnist_train.csv                              # Dataset (not included)
└── README.md
```

---

## 📝 Notes

- The dataset CSV is **not included** in this repo. Download it from [Kaggle](https://www.kaggle.com/datasets/zalando-research/fashionmnist).
- Training time depends on hardware; a GPU is strongly recommended for the full Optuna search.
- Increase `n_trials` in `study.optimize()` for a more thorough search at the cost of longer runtime.

---

## 📄 License

This project is open-source and available under the [MIT License](LICENSE).
