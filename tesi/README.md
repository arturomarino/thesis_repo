# Tesi: manoscritto e materiali della consegna

Questa directory contiene l'unica versione della tesi conservata nella repository, basata sul pacchetto finale del 25 settembre 2026.

- [main.pdf](main.pdf): PDF già compilato, pronto per la consultazione.
- [main.tex](main.tex): documento principale LaTeX.
- `chapter/`: capitoli e tabelle dei risultati.
- `img/`: logo e figure effettivamente usate nel manoscritto.
- `reference.bib`: bibliografia.

## Compilazione

Per Overleaf, creare uno ZIP del **contenuto di questa directory**, importarlo come nuovo progetto e scegliere `main.tex` come documento principale. Non caricare dataset, checkpoint o l'intera repository.

Con una distribuzione LaTeX locale dotata di `latexmk`:

```bash
cd tesi
latexmk -pdf main.tex
```

Il PDF è un materiale della consegna e viene versionato; i file ausiliari della compilazione sono ignorati. Dopo modifiche ai capitoli o alle figure, ricompilare il PDF prima della consegna.

## Provenienza dei risultati

Le quattro figure annuali 1999 provengono dalla valutazione del 24 settembre 2026, `annual_evaluation_v3_corrected`. Rappresentano:

1. MAE del modello.
2. Deviazione standard temporale dell'errore assoluto giornaliero del modello.
3. MAE della persistence.
4. Differenza con segno: MAE modello − MAE persistence.

A 0,506 m, il MAE della temperatura del modello è 0,27683 °C e quello della persistence è 0,14564 °C, su 364 date previste. Questi valori sono riportati nella [tabella sperimentale](chapter/annual-results-generated.tex).

La run è stata fornita come valutazione di `best_forecaster_v3.pt`, selezionato all'epoca 94. I JSON e NetCDF originali non incorporano il nome o l'hash del checkpoint: il log della run è necessario per una provenienza formale. I risultati e il PDF sono conservati qui; dati e pesi restano esterni alla repository.

Per rigenerare tabelle e figure consultare la [guida all'esecuzione](../docs/esecuzione.md).
