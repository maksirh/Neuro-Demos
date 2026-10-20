# Neuro-Demos

A set of small, self-contained Python scripts demonstrating basic machine learning and neural network concepts (fuzzy clustering, Hebbian learning, function approximation).

> **Screenshots:** TODO — add the plots produced by each script (e.g. `docs/*.png`)

## Scripts

| Script | Topic | What it does |
|---|---|---|
| `fuzzy_cmeans_demo.py` | Clustering | Fuzzy C-Means (`skfuzzy.cmeans`, c=3, m=1.5) on 100 random 2-D points; plots clusters, centers and the objective-function convergence curve |
| `hebbian_letters.py` | Associative memory | Hebbian learning rule to recognize the letters M, A, K, S from binary 3×5 masks; tests on training letters and a noisy "M" |
| `function_approximation_nn.py` | Function approximation | Compares Feed-Forward (10/20 neurons), Cascade-Forward (1–2 hidden layers) and Elman RNN (1×15, 3×5) models in Keras approximating z = 0.5·sin(x+y)·cos(y) at y = 0.5, x ∈ [-2, 1]; plots predictions and MSE curves, prints a mean relative error |

## Tech stack

Python ≥ 3.9, NumPy, Matplotlib, scikit-fuzzy, TensorFlow ≥ 2.15

## Running locally

```bash
git clone https://github.com/maksirh/Neuro-Demos.git
cd Neuro-Demos
python -m venv venv
source venv/bin/activate            # Windows: venv\Scripts\activate
pip install -r requirements.txt

python fuzzy_cmeans_demo.py
python hebbian_letters.py
python function_approximation_nn.py
```

## TODO

- [ ] Screenshots of the generated plots
