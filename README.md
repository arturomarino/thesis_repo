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
-> Temporal windows: [volume(t-2), volume(t-1), volume(t)] -> volume(t+1)
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
receives three consecutive daily volumes and is trained against the following
time step. The optimization objective combines the masked Gaussian negative
log-likelihood with a masked MSE term on the predicted mean:

```text
0.5 * ((target - mean)^2 * exp(-log_variance) + log_variance)
+ mean_mse_weight * MSE(target, mean)
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
  --checkpoint-path /content/drive/MyDrive/Thesis/best_forecaster_v2.pt \
  --reuse-stats \
  --device cuda \
  --batch-size 1 \
  --num-workers 2 \
  --epochs 100 \
  --patience 15 \
  --context-steps 3 \
  --internal-normalization none \
  --mean-mse-weight 1.0 \
  --gradient-clip-norm 1.0 \
  --learning-curve-directory \
    /content/drive/MyDrive/Thesis/thesis_repo/outputs/learning_curves_v2 \
  --train-model
```

Il nuovo checkpoint usa un nome diverso per non sovrascrivere l'esperimento
precedente. L'early stopping e la selezione del modello migliore usano la
validation RMSE. La persistence viene misurata una volta sulla validation e
resta esclusivamente un riferimento esterno: non entra nel modello o nella
loss.

Dopo ogni epoca il training set viene valutato nuovamente senza aggiornare i
pesi. In questo modo le curve train e validation sono calcolate con lo stesso
modello a pesi fissi. Questo rende il confronto corretto, ma non forza
artificialmente la validation a essere sopra il training: un anno di
validation realmente piu' semplice puo' comunque ottenere una metrica
inferiore.

Dopo ogni epoca viene salvato anche uno snapshot cumulativo nella cartella
`outputs/learning_curves` della repo. Per esempio,
`learning_curve_epoch_011.png` mostra tutta la history dall'asse 0 all'epoca
11, mentre `learning_curve_epoch_012.png` la mostra fino all'epoca 12. In caso
di `--resume`, gli snapshot delle epoche gia' presenti nel checkpoint vengono
rigenerati automaticamente prima di continuare il training.

The best model is saved at the requested checkpoint path; the state of every
completed epoch uses the `_last.pt` suffix. After a Colab interruption,
repeat the same training command and add `--resume` to continue from the next
epoch instead of restarting from zero. At the end of training, Matplotlib also
saves a learning-curve PNG next to the checkpoint. The graph contains NLL and
RMSE, shows the validation persistence reference and highlights the best
validation RMSE epoch.

The learning curve can be regenerated directly from the checkpoint history,
without loading the NetCDF dataset:

```bash
python src/train.py \
  --checkpoint-path /content/drive/MyDrive/Thesis/best_forecaster.pt \
  --learning-curve-directory \
    /content/drive/MyDrive/Thesis/thesis_repo/outputs/learning_curves \
  --plot-learning-curve
```

Use `--learning-curve-path /path/curve.png` to choose a different output path.

## Temperature map

La temperatura marina osservata (`thetao_cglo`) puo' essere visualizzata per
un giorno e un livello di profondita' specifici. La mappa usa il colore per la
temperatura e sovrappone un campione regolare di valori numerici per mantenere
la figura leggibile.

Giorno scelto, primo livello vicino alla superficie:

```bash
python src/plot_temperature_map.py \
  --data-path /content/glorys12_med_test_1994_2003.nc \
  --mask-path /content/land_sea_mask.nc \
  --date 2000-08-15 \
  --output-directory \
    /content/drive/MyDrive/Thesis/thesis_repo/outputs/temperature_maps
```

Per scegliere un giorno casuale riproducibile, omettere `--date`. Per chiedere
il livello disponibile piu' vicino a una profondita' specifica aggiungere, per
esempio, `--depth 10`. `--label-step 8` mostra un valore ogni otto punti della
griglia; valori piu' piccoli producono piu' etichette.

After model selection is complete, the reserved final year can be evaluated
once against both the trained checkpoint and the persistence baseline with:

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

Before freezing the model, the same comparison can be run safely on the
validation year by replacing `--evaluate-test` with
`--evaluate-validation`. The reported RMSE skill score is
`1 - MSE_model / MSE_persistence`: positive values mean that the neural model
outperforms the forecast that simply copies the previous day.

The already-trained checkpoint can also be evaluated specifically for sea
temperature in degrees Celsius, without retraining:

```bash
python src/train.py \
  --data-path /content/glorys12_med_test_1994_2003.nc \
  --mask-path /content/land_sea_mask.nc \
  --stats-path /content/normalization_stats.nc \
  --checkpoint-path /content/drive/MyDrive/Thesis/best_forecaster.pt \
  --reuse-stats \
  --device cuda \
  --temperature-depth 0.5 \
  --evaluate-temperature-validation
```

This reports physical RMSE, MAE, bias, predicted uncertainty and coverage for
all depths and for the available level nearest to the requested depth. Use
`--evaluate-temperature-test` only after model selection is complete.

Quick checks:

```bash
python src/train.py --smoke-test-normalization
python src/train.py --smoke-test-dataset
python src/train.py --smoke-test-dataloader
python src/train.py --smoke-test-autoencoder
python src/train.py --smoke-test-training
```
