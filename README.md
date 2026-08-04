# Generative MedFormer Thesis Project

Repository for a bachelor thesis in Computer Engineering.

The project starts from MedFormer and aims to develop a generative version using
a latent diffusion model. The current implementation focuses on a clean research
pipeline for oceanographic Copernicus NetCDF data:

```text
DataManager
-> Preprocessing
-> Temporal split
-> Normalization
-> Forecast pairs: volume(t) -> volume(t+1)
-> DataLoader
-> Probabilistic 3D U-Net Autoencoder
-> Latent space
```

The temporal split is chronological: the latest calendar year is reserved for
test, the previous calendar year for validation, and all earlier years for
training.

The autoencoder is heteroscedastic: for every valid ocean point it predicts
the next-step Gaussian parameters `mean = mu` and
`log_variance = log(sigma^2)` for temperature, salinity, `uo`, and `vo`. It
receives only the current volume and is trained against the following time
step with the masked Gaussian negative log-likelihood

```text
0.5 * ((target - mean)^2 * exp(-log_variance) + log_variance)
```

The predicted standard deviation is `exp(0.5 * log_variance)`.
`src/inference.py` exposes both the distribution statistics and stochastic
sampling via the reparameterization formula `mean + std * epsilon`.

Large data files are not tracked in Git.

Expected local data paths:

```text
data/raw/copernicus.nc
data/masks/land_sea_mask.nc
```

Colab preparation check (loads and validates the complete pipeline but does
not train):

```bash
git clone https://github.com/arturomarino/thesis_repo.git
cd thesis_repo
pip install -r requirements.txt
# Copia locale: evita di calcolare le statistiche leggendo 18 GB da Drive.
cp /content/drive/MyDrive/Thesis/glorys12_med_test_1994_2003.nc /content/
python src/train.py \
  --data-path /content/glorys12_med_test_1994_2003.nc \
  --mask-path /content/drive/MyDrive/Thesis/land_sea_mask.nc \
  --stats-path /content/drive/MyDrive/Thesis/normalization_stats.nc \
  --time-chunk 16
```

Training starts only with the explicit `--train-model` flag:

```bash
python src/train.py \
  --data-path /content/glorys12_med_test_1994_2003.nc \
  --mask-path /content/drive/MyDrive/Thesis/land_sea_mask.nc \
  --stats-path /content/drive/MyDrive/Thesis/normalization_stats.nc \
  --checkpoint-path /content/drive/MyDrive/Thesis/best_forecaster.pt \
  --reuse-stats \
  --device cuda \
  --batch-size 1 \
  --num-workers 2 \
  --epochs 100 \
  --patience 10 \
  --train-model
```

The best model is saved as `best_forecaster.pt`; the state of every completed
epoch is saved as `best_forecaster_last.pt`. After a Colab interruption,
repeat the same training command and add `--resume` to continue from the next
epoch instead of restarting from zero. At the end of training, Matplotlib also
saves `best_forecaster_learning_curve.png` next to the checkpoint. The graph
compares training and validation Gaussian NLL and highlights the best epoch.

The learning curve can be regenerated directly from the checkpoint history,
without loading the NetCDF dataset:

```bash
python src/train.py \
  --checkpoint-path /content/drive/MyDrive/Thesis/best_forecaster.pt \
  --plot-learning-curve
```

Use `--learning-curve-path /path/curve.png` to choose a different output path.

After model selection is complete, the reserved final year can be evaluated
once with:

```bash
python src/train.py \
  --data-path /content/glorys12_med_test_1994_2003.nc \
  --mask-path /content/drive/MyDrive/Thesis/land_sea_mask.nc \
  --stats-path /content/drive/MyDrive/Thesis/normalization_stats.nc \
  --checkpoint-path /content/drive/MyDrive/Thesis/best_forecaster.pt \
  --reuse-stats \
  --device cuda \
  --evaluate-test
```

Quick checks:

```bash
python src/train.py --smoke-test-normalization
python src/train.py --smoke-test-dataset
python src/train.py --smoke-test-dataloader
python src/train.py --smoke-test-autoencoder
python src/train.py --smoke-test-training
```
