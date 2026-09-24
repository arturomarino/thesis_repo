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

The legacy temperature command remains available without retraining:

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

For backward compatibility this command now routes to the multivariable
physical evaluator described below; it reports all four channels rather than
temperature alone. Use `--evaluate-temperature-test` only after model selection
is complete.

## Annual multivariable evaluation

The final-year evaluation can include the complete January 2--December 31
target interval. With a three-day context, the last two days of the previous
year are prepended as inputs only. They are never counted as validation or test
targets. For the non-leap 1998 and 1999 splits this produces 364 forecasts.

The following command evaluates temperature, salinity, zonal velocity and
meridional velocity in their physical units, saves JSON/CSV tables, and creates
four 2x2 surface figures: pointwise annual mean absolute error of the model,
temporal standard deviation of its signed forecast error, mean absolute error
of the persistence baseline, and their MAE difference (model minus
persistence). Each panel uses its 95th percentile as
the color-scale maximum so isolated outliers do not hide the spatial pattern:

```bash
python src/train.py \
  --data-path /content/glorys12_med_test_1994_2003.nc \
  --mask-path /content/land_sea_mask.nc \
  --stats-path /content/normalization_stats.nc \
  --checkpoint-path /content/drive/MyDrive/Thesis/best_forecaster_v3.pt \
  --reuse-stats \
  --device cuda \
  --evaluation-depth 0.5 \
  --evaluation-output-directory \
    /content/drive/MyDrive/Thesis/thesis_repo/outputs/annual_evaluation \
  --evaluate-physical-validation \
  --evaluate-physical-test \
  --plot-annual-errors-test
```

The output directory contains `physical_metrics_validation.json/.csv`,
`physical_metrics_test.json/.csv`, `annual_mae_maps_1999.nc`, and
the figures `annual_mae_maps_1999.png`,
`annual_error_standard_deviation_maps_1999.png`,
`annual_persistence_mae_maps_1999.png`, and
`annual_mae_difference_model_minus_persistence_maps_1999.png`. The NetCDF
stores the model and persistence mean absolute errors, their difference, the
population standard deviation of the model signed error, and number of valid
dates at every grid cell. Positive values in the difference map mean the model
has a larger MAE than persistence. The legacy
`--evaluate-temperature-validation` and `--evaluate-temperature-test` flags
remain accepted as aliases for the multivariable evaluation.

After both JSON files exist, generate the LaTeX tables and copy the figure into
the canonical thesis source with:

```bash
python src/render_thesis_results.py \
  --validation-json outputs/annual_evaluation/physical_metrics_validation.json \
  --test-json outputs/annual_evaluation/physical_metrics_test.json \
  --output-tex tesi/chapter/annual-results-generated.tex \
  --figure-source outputs/annual_evaluation/annual_mae_maps_1999.png \
  --figure-destination tesi/img/annual-mae-maps-1999.png
```

The renderer refuses metric files that do not contain exactly 364 forecasts,
preventing the old 362-pair results from being inserted by mistake.

Quick checks:

```bash
python src/train.py --smoke-test-normalization
python src/train.py --smoke-test-dataset
python src/train.py --smoke-test-dataloader
python src/train.py --smoke-test-autoencoder
python src/train.py --smoke-test-training
```
