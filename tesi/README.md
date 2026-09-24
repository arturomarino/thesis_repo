# Tesi LaTeX

Base in italiano con i cinque capitoli richiesti dal relatore. Riprende dal
progetto fornito classe book, formato A4, corpo 11 pt, margini e interlinea.
Il frontespizio va adattato alle regole ufficiali del proprio corso.
Testi, risultati, bibliografia e dati personali della tesi di esempio non sono
stati trasferiti. Le sezioni proposte sono una traccia modificabile.

## Come iniziare

1. Completare `frontespizio.tex` con ateneo, dipartimento, corso, titolo,
   relatore, laureando, matricola e anno accademico.
2. Scrivere nei file della cartella `chapter/`, sostituendo i segnaposto.
3. Mantenere l'introduzione entro quattro pagine.
4. Inserire le immagini in `img/` e aggiornare le fonti verificate in
   `reference.bib`. La bibliografia è già attiva in `main.tex`; nel testo usare
   `\citep{chiave}` per aggiungere una citazione.

## Compilazione

Su Overleaf, caricare il contenuto di questa cartella e scegliere `main.tex`
come documento principale e pdfLaTeX come compilatore.

In locale, con una distribuzione LaTeX installata, dalla cartella `tesi`:

```sh
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

Non è stata eseguita una compilazione iniziale: nell'ambiente di preparazione
non sono stati trovati pdflatex, latexmk o tectonic nel PATH.
