# Guida all'esecuzione

Questa guida descrive la pipeline implementata e i comandi per riprodurre training, valutazione e figure. Per obiettivi e risultati consultare il [README](../README.md); per il manoscritto vedere [tesi/main.pdf](../tesi/main.pdf).

## 1. Ambiente e test

L'ambiente locale verificato usa Python 3.11. Creare un virtual environment e installare PyTorch prima delle altre dipendenze:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install torch
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

Per una macchina CUDA usare la build PyTorch adatta al proprio ambiente. In Colab **saltare l'installazione di PyTorch e la creazione del virtual environment**: `requirements.txt` mantiene la build CUDA già disponibile.

Gli smoke test sono eseguibili senza dati esterni:

```bash
python src/train.py --smoke-test-normalization
python src/train.py --smoke-test-dataset
python src/train.py --smoke-test-dataloader
python src/train.py --smoke-test-autoencoder
python src/train.py --smoke-test-training
```

`python src/train.py --help` elenca tutte le opzioni. La raccolta pytest è limitata a `tests/`, così non attraversa dati, esportazioni della tesi o dipendenze locali.

## 2. Dati e checkpoint

I dati reali e i pesi del modello non sono versionati. I file dell'esperimento sono conservati su Google Drive; l'accesso dipende dai permessi già assegnati:

| File | Destinazione locale | Riferimento |
| --- | --- | --- |
| Dataset di rianalisi, circa 18,3 GB | `data/raw/copernicus.nc` | [NetCDF su Drive](https://drive.google.com/file/d/1qkKLNR8whiVD4oJ8JDVDq3NazZlrmJdm/view) |
| Maschera terra–mare | `data/masks/land_sea_mask.nc` | [Maschera su Drive](https://drive.google.com/file/d/18v-wsjsoNKuPPw4MOM7pnhrJqJIEml2u/view) |
| Statistiche di normalizzazione | `data/processed/normalization_stats.nc` | [Statistiche su Drive](https://drive.google.com/file/d/1kF01j5HQXms836AMQVorbmBrLswGC3DH/view) |
| Checkpoint v3 | `checkpoints/best_forecaster_v3.pt` | [Checkpoint su Drive](https://drive.google.com/file/d/1VyqfRkxvAgXmPQyxrZQhpYc-vpVlSQLU/view) |

Il nome originale del dataset è `glorys12_med_test_1994_2003.nc`; il periodo effettivo riportato nell'esperimento è **1994–1999**. La pipeline richiede `thetao_cglo`, `so_cglo`, `uo_cglo`, `vo_cglo` e `zos_cglo`, con coordinate `time`, `depth`, `latitude` e `longitude`. Il previsore usa soltanto le prime quattro variabili volumetriche.

La divisione è cronologica: l'ultimo anno disponibile è il test, il precedente la validation, tutti gli altri il training. Le statistiche sono calcolate sul solo training; riutilizzare quelle salvate soltanto con dati e griglia compatibili.

### Preparazione in locale

Con dataset e maschera nei percorsi predefiniti:

```bash
python src/train.py --time-chunk 16
```

Questo comando prepara e controlla la pipeline, calcola e salva le statistiche, ma avvia il training soltanto se si aggiunge `--train-model`. Se le statistiche compatibili sono già presenti, usare `--reuse-stats`.

### Preparazione in Google Colab

Montare Drive nel notebook, clonare la repository ed eseguire i comandi nella sua directory:

```python
from google.colab import drive
drive.mount('/content/drive')
```

```bash
git clone https://github.com/arturomarino/thesis_repo.git
cd thesis_repo
python -m pip install -r requirements.txt
cp /content/drive/MyDrive/Thesis/glorys12_med_test_1994_2003.nc /content/
python src/train.py \
  --data-path /content/glorys12_med_test_1994_2003.nc \
  --mask-path /content/drive/MyDrive/Thesis/land_sea_mask.nc \
  --stats-path /content/drive/MyDrive/Thesis/normalization_stats.nc \
  --time-chunk 16
```

La copia del NetCDF sul disco della sessione evita letture ripetute da Drive. Negli esempi seguenti i percorsi sono locali; in Colab sostituirli con quelli qui sopra e salvare i checkpoint su Drive per conservarli tra sessioni.

## 3. Training e ripresa

Esempio di un **nuovo esperimento** con il contesto e i pesi della loss della configurazione finale:

```bash
python src/train.py \
  --reuse-stats \
  --checkpoint-path checkpoints/forecaster_new_run.pt \
  --device auto \
  --batch-size 1 \
  --num-workers 0 \
  --epochs 100 \
  --patience 15 \
  --context-steps 3 \
  --internal-normalization none \
  --mean-mse-weight 2.0 \
  --temperature-mse-weight 2.0 \
  --gradient-clip-norm 1.0 \
  --learning-curve-directory outputs/learning_curves_new_run \
  --train-model
```

Le 100 epoche e la patience 15 sono impostazioni dell'esempio: i valori passati al comando storico non sono conservati nel checkpoint. La run descritta nella tesi ha completato 96 epoche e selezionato l'epoca 94.

La loss combina Gaussian NLL mascherata e MSE pesata sulla media. Early stopping e selezione del checkpoint usano la RMSE di validation. La persistence è un riferimento esterno; non entra nella loss.

Dopo ogni epoca vengono salvati:

- il checkpoint migliore nel percorso richiesto;
- lo stato per la ripresa con suffisso `_last.pt`, inclusi ottimizzatore e history;
- una curva cumulativa in `outputs/learning_curves_new_run/`.

Per riprendere aggiungere `--resume` allo stesso comando, mantenendo configurazione e pesi della loss. Le metriche train e validation sono calcolate a pesi fissi dopo ogni epoca; un anno di validation più semplice può avere errore inferiore al training.

Rigenerare le curve senza caricare il NetCDF:

```bash
python src/train.py \
  --checkpoint-path checkpoints/best_forecaster_v3.pt \
  --learning-curve-path outputs/learning_curve_v3.png \
  --learning-curve-directory outputs/learning_curves_v3 \
  --plot-learning-curve
```

Se disponibile, il comando usa la history del checkpoint `_last.pt` per includere tutte le epoche completate.

## 4. Valutazione annuale nelle unità fisiche

Prima della selezione finale del modello usare soltanto la validation:

```bash
python src/train.py \
  --reuse-stats \
  --checkpoint-path checkpoints/best_forecaster_v3.pt \
  --device auto \
  --evaluation-depth 0.506 \
  --evaluation-output-directory outputs/annual_evaluation_v3_corrected \
  --evaluate-physical-validation
```

Dopo aver fissato il modello, valutare il test e produrre le mappe:

```bash
python src/train.py \
  --reuse-stats \
  --checkpoint-path checkpoints/best_forecaster_v3.pt \
  --device auto \
  --evaluation-depth 0.506 \
  --evaluation-output-directory outputs/annual_evaluation_v3_corrected \
  --evaluate-physical-test \
  --plot-annual-errors-test
```

Con tre giorni di contesto, gli ultimi due giorni dell'anno precedente vengono anteposti come soli input. Per 1998 e 1999 risultano 364 target dal 2 gennaio al 31 dicembre, tutti interni allo split valutato.

| Output | Contenuto |
| --- | --- |
| `physical_metrics_validation.json/.csv` | Metriche delle quattro variabili, su tutte le profondità e sul livello scelto |
| `physical_metrics_test.json/.csv` | Stesse metriche sul test |
| `annual_mae_maps_1999.nc` | MAE modello e persistence, differenza, deviazione standard dell'errore assoluto giornaliero e conteggi validi |
| `annual_mae_maps_1999.png` | MAE annuale del modello |
| `annual_error_standard_deviation_maps_1999.png` | Deviazione standard temporale dell'errore assoluto del modello |
| `annual_persistence_mae_maps_1999.png` | MAE annuale della persistence |
| `annual_mae_difference_model_minus_persistence_maps_1999.png` | MAE modello − MAE persistence |

Le mappe non negative usano il 95° percentile di ciascun pannello come massimo della scala; la differenza usa limiti simmetrici al 95° percentile del valore assoluto. Nella differenza, valori positivi indicano un MAE maggiore del modello.

Lo skill score implementato è `1 - MSE_modello / MSE_persistence`: valori positivi indicano un miglioramento. Le metriche aggregano i punti oceanici validi; non applicano pesi per area geografica o spessore dei livelli.

Per un confronto aggregato nello spazio normalizzato usare `--evaluate-validation` o `--evaluate-test`. Gli alias `--evaluate-temperature-validation` e `--evaluate-temperature-test` restano disponibili e richiamano la valutazione fisica multivariata.

## 5. Figure giornaliere e tabelle per la tesi

Una mappa della temperatura della rianalisi:

```bash
python src/plot_temperature_map.py \
  --data-path data/raw/copernicus.nc \
  --mask-path data/masks/land_sea_mask.nc \
  --date 1999-08-15 \
  --depth 0.506 \
  --label-step 8 \
  --output-directory outputs/temperature_maps
```

Omettendo `--date` si sceglie un giorno casuale riproducibile; `--depth` seleziona il livello disponibile più vicino. Per le mappe di previsione giornaliera e del relativo errore consultare `python src/plot_temperature_forecast.py --help`.

Dopo aver prodotto entrambi i JSON dalla stessa run:

```bash
python src/render_thesis_results.py \
  --validation-json outputs/annual_evaluation_v3_corrected/physical_metrics_validation.json \
  --test-json outputs/annual_evaluation_v3_corrected/physical_metrics_test.json \
  --output-tex tesi/chapter/annual-results-generated.tex \
  --figure-source outputs/annual_evaluation_v3_corrected/annual_mae_maps_1999.png \
  --figure-destination tesi/img/annual-mae-maps-1999.png
cp outputs/annual_evaluation_v3_corrected/annual_error_standard_deviation_maps_1999.png \
  tesi/img/annual-error-std-maps-1999.png
cp outputs/annual_evaluation_v3_corrected/annual_persistence_mae_maps_1999.png \
  tesi/img/annual-persistence-mae-maps-1999.png
cp outputs/annual_evaluation_v3_corrected/annual_mae_difference_model_minus_persistence_maps_1999.png \
  tesi/img/annual-mae-difference-1999.png
```

Il renderer richiede esattamente 364 previsioni e otto righe di metriche per split: questa regola è specifica agli anni dell'esperimento della tesi. Per altri periodi va adattata. Figure e tabelle devono provenire dallo stesso checkpoint e dalla stessa valutazione.

I JSON e NetCDF della valutazione storica non incorporano il nome o l'hash del checkpoint; conservare il log della run per una provenienza formale. Il PDF incluso è quello già compilato della tesi finale: rigenerarlo dopo eventuali modifiche ai sorgenti seguendo [tesi/README.md](../tesi/README.md).
