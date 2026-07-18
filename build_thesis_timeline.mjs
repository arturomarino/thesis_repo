import fs from "node:fs/promises";
import { Workbook, SpreadsheetFile } from "@oai/artifact-tool";

const outDir = "/Users/arturo/Downloads/thesis_repo/outputs/tesi_timeline_2026";
const outFile = `${outDir}/Timeline_fine_tesi_settembre_2026.xlsx`;
const wb = Workbook.create();
const ws = wb.worksheets.add("Timeline");
const data = wb.worksheets.add("Milestone");

const navy = "#17324D";
const blue = "#2F75B5";
const pale = "#EFF6FC";
const light = "#DCEAF7";
const green = "#70AD47";
const red = "#C00000";
const line = "#8DB9E2";

const milestones = [
  [1, new Date(2026, 6, 13), "Piano definitivo", "Indice, obiettivi e domanda di ricerca definiti", "Non iniziato"],
  [2, new Date(2026, 6, 24), "Letteratura chiusa", "Fonti e quadro teorico completati", "Non iniziato"],
  [3, new Date(2026, 7, 7), "Dati pronti", "Raccolta e pulizia concluse", "Non iniziato"],
  [4, new Date(2026, 7, 14), "Metodologia completa", "Capitolo metodologico pronto", "Non iniziato"],
  [5, new Date(2026, 7, 21), "Analisi conclusa", "Risultati verificati e figure pronte", "Non iniziato"],
  [6, new Date(2026, 7, 28), "Risultati scritti", "Capitolo dei risultati completo", "Non iniziato"],
  [7, new Date(2026, 8, 4), "Discussione completa", "Interpretazione dei risultati conclusa", "Non iniziato"],
  [8, new Date(2026, 8, 7), "Bozza al relatore", "Manoscritto completo inviato", "Non iniziato"],
  [9, new Date(2026, 8, 11), "Revisioni integrate", "Feedback del relatore recepito", "Non iniziato"],
  [10, new Date(2026, 8, 14), "PDF finale", "Correzione e impaginazione concluse", "Non iniziato"],
  [11, new Date(2026, 8, 15), "Consegna", "Tesi caricata e ricevuta salvata", "Non iniziato"],
];

// Editable source table.
data.showGridLines = false;
data.getRange("A1:E1").values = [["#", "Data", "Milestone", "Risultato atteso", "Stato"]];
data.getRange("A2:E12").values = milestones;
data.getRange("B2:B12").setNumberFormat("dd mmm yyyy");
data.getRange("A1:E1").format = { fill: navy, font: { bold: true, color: "#FFFFFF" }, horizontalAlignment: "center" };
data.getRange("A2:E12").format.borders = { preset: "inside", style: "thin", color: "#E3EAF2" };
data.getRange("C2:D12").format.wrapText = true;
data.getRange("E2:E12").dataValidation = { rule: { type: "list", values: ["Non iniziato", "In corso", "Completato"] } };
data.getRange("E2:E12").conditionalFormats.add("containsText", { text: "Completato", format: { fill: "#E2F0D9", font: { color: "#375623", bold: true } } });
data.getRange("E2:E12").conditionalFormats.add("containsText", { text: "In corso", format: { fill: "#FFF2CC", font: { color: "#7F6000", bold: true } } });
data.getRange("A1:A12").format.columnWidth = 5;
data.getRange("B1:B12").format.columnWidth = 15;
data.getRange("C1:C12").format.columnWidth = 25;
data.getRange("D1:D12").format.columnWidth = 42;
data.getRange("E1:E12").format.columnWidth = 16;
data.getRange("A2:E12").format.rowHeight = 28;
data.freezePanes.freezeRows(1);

// Single horizontal timeline on the X axis.
ws.showGridLines = false;
ws.mergeCells("A1:W1");
ws.getRange("A1").values = [["Timeline per concludere la tesi · luglio – settembre 2026"]];
ws.getRange("A1:W1").format = { fill: navy, font: { bold: true, color: "#FFFFFF", size: 17 }, horizontalAlignment: "center", verticalAlignment: "center" };
ws.getRange("A1:W1").format.rowHeight = 34;
ws.mergeCells("A2:W2");
ws.getRange("A2").values = [["Una sola linea temporale: le date scorrono da sinistra a destra sull’asse X."]];
ws.getRange("A2:W2").format = { fill: pale, font: { italic: true, color: "#425466" }, horizontalAlignment: "center" };
ws.getRange("A2:W2").format.rowHeight = 22;

// Summary cards.
for (const pair of [["B3:C3", "INIZIO"], ["G3:H3", "FINE"], ["L3:M3", "AVANZAMENTO"]]) {
  ws.mergeCells(pair[0]);
  ws.getRange(pair[0].split(":")[0]).values = [[pair[1]]];
  ws.getRange(pair[0]).format = { fill: light, font: { bold: true, color: navy }, horizontalAlignment: "center" };
}
ws.mergeCells("D3:E3");
ws.mergeCells("I3:J3");
ws.mergeCells("N3:O3");
ws.getRange("D3").formulas = [["='Milestone'!B2"]];
ws.getRange("I3").formulas = [["='Milestone'!B12"]];
ws.getRange("N3").formulas = [["=COUNTIF('Milestone'!E2:E12,\"Completato\")/COUNTA('Milestone'!C2:C12)"]];
ws.getRange("D3:E3").setNumberFormat("dd mmm yyyy");
ws.getRange("I3:J3").setNumberFormat("dd mmm yyyy");
ws.getRange("N3:O3").setNumberFormat("0%");
ws.getRange("D3:E3").format = { font: { bold: true, color: navy }, horizontalAlignment: "center" };
ws.getRange("I3:J3").format = { font: { bold: true, color: navy }, horizontalAlignment: "center" };
ws.getRange("N3:O3").format = { font: { bold: true, color: navy }, horizontalAlignment: "center" };
ws.getRange("B3:O3").format.rowHeight = 24;

// Each milestone spans two columns; labels alternate above and below the axis.
const starts = ["B", "D", "F", "H", "J", "L", "N", "P", "R", "T", "V"];
const ends   = ["C", "E", "G", "I", "K", "M", "O", "Q", "S", "U", "W"];
for (let i = 0; i < 11; i++) {
  const s = starts[i];
  const e = ends[i];
  const src = i + 2;
  const above = i % 2 === 0;
  const titleRow = above ? 5 : 11;
  const descRow = above ? 6 : 12;
  const stemRows = above ? `${s}7:${e}7` : `${s}9:${e}10`;

  ws.mergeCells(`${s}${titleRow}:${e}${titleRow}`);
  ws.getRange(`${s}${titleRow}`).formulas = [[`='Milestone'!C${src}`]];
  ws.getRange(`${s}${titleRow}:${e}${titleRow}`).format = { fill: i === 10 ? "#F4CCCC" : light, font: { bold: true, color: i === 10 ? red : navy, size: 10 }, horizontalAlignment: "center", verticalAlignment: "center", wrapText: true };

  ws.mergeCells(`${s}${descRow}:${e}${descRow}`);
  ws.getRange(`${s}${descRow}`).formulas = [[`='Milestone'!D${src}`]];
  ws.getRange(`${s}${descRow}:${e}${descRow}`).format = { fill: pale, font: { color: "#425466", size: 9 }, horizontalAlignment: "center", verticalAlignment: "center", wrapText: true };

  ws.getRange(stemRows).format.fill = line;
  ws.mergeCells(`${s}8:${e}8`);
  ws.getRange(`${s}8`).values = [["●"]];
  ws.getRange(`${s}8:${e}8`).format = { fill: navy, font: { bold: true, color: i === 10 ? "#FF6666" : "#FFFFFF", size: 14 }, horizontalAlignment: "center", verticalAlignment: "center" };

  ws.mergeCells(`${s}14:${e}14`);
  ws.getRange(`${s}14`).formulas = [[`='Milestone'!B${src}`]];
  ws.getRange(`${s}14:${e}14`).setNumberFormat("dd mmm");
  ws.getRange(`${s}14:${e}14`).format = { font: { bold: true, color: i === 10 ? red : navy, size: 9 }, horizontalAlignment: "center" };
}

ws.mergeCells("B15:W15");
ws.getRange("B15").values = [["ASSE X · tempo →"]];
ws.getRange("B15:W15").format = { font: { bold: true, color: blue }, horizontalAlignment: "right", borders: { bottom: { style: "medium", color: blue } } };
ws.getRange("B15:W15").format.rowHeight = 22;

ws.mergeCells("B17:W17");
ws.getRange("B17").values = [["Modifica date, milestone e stato nel foglio “Milestone”: la timeline si aggiorna automaticamente."]];
ws.getRange("B17:W17").format = { fill: pale, font: { italic: true, color: navy }, horizontalAlignment: "center" };
ws.getRange("B17:W17").format.rowHeight = 25;

ws.getRange("A1:A17").format.columnWidth = 3;
ws.getRange("B1:W17").format.columnWidth = 8.5;
ws.getRange("A5:W5").format.rowHeight = 32;
ws.getRange("A6:W6").format.rowHeight = 42;
ws.getRange("A7:W7").format.rowHeight = 10;
ws.getRange("A8:W8").format.rowHeight = 24;
ws.getRange("A9:W10").format.rowHeight = 10;
ws.getRange("A11:W11").format.rowHeight = 32;
ws.getRange("A12:W12").format.rowHeight = 42;
ws.getRange("A14:W14").format.rowHeight = 22;
ws.freezePanes.freezeRows(3);

const check = await wb.inspect({ kind: "table", range: "Timeline!A1:W17", include: "values,formulas", tableMaxRows: 17, tableMaxCols: 23 });
console.log(check.ndjson);
const errors = await wb.inspect({ kind: "match", searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A", options: { useRegex: true, maxResults: 100 }, summary: "formula errors" });
console.log(errors.ndjson);
const preview = await wb.render({ sheetName: "Timeline", range: "A1:W17", scale: 1.35 });
await fs.mkdir(outDir, { recursive: true });
await fs.writeFile(`${outDir}/preview.png`, new Uint8Array(await preview.arrayBuffer()));
const out = await SpreadsheetFile.exportXlsx(wb);
await out.save(outFile);
console.log(outFile);
