# Risultati Valutazione LSTM Omnivore (Recordings)

## Configurazione
- **Variant**: LSTM
- **Backbone**: Omnivore
- **Split**: Recordings
- **Threshold**: 0.4
- **Checkpoint**: error_recognition_recordings_omnivore_lstm_video_best.pt

## Metriche a Livello di Sub-Step

| Metrica | Valore |
|---------|--------|
| Precision | 0.4619 |
| Recall | 0.5251 |
| F1-Score | 0.4914 |
| Accuracy | 0.6427 |
| AUC-ROC | 0.6670 |
| PR-AUC | 0.3987 |

## Metriche a Livello di Step

| Metrica | Valore |
|---------|--------|
| Precision | 0.4602 |
| Recall | 0.6473 |
| F1-Score | 0.5379 |
| Accuracy | 0.6006 |
| AUC-ROC | 0.6528 |
| PR-AUC | 0.4246 |

## Metriche per Categoria (Step Level)

| Categoria | Precision | Recall | F1-Score | Accuracy | AUC-ROC | PR-AUC |
|-----------|-----------|--------|----------|----------|---------|--------|
| 0 | 0.8053 | 0.4773 | 0.5993 | 0.4560 | 0.3797 | 0.8300 |
| 2 | 0.1032 | 0.7143 | 0.1804 | 0.5261 | 0.6408 | 0.0946 |
| 3 | 0.0206 | 0.8750 | 0.0403 | 0.5037 | 0.6699 | 0.0196 |
| 4 | 0.1091 | 0.7400 | 0.1902 | 0.5305 | 0.6885 | 0.1001 |
| 5 | 0.0501 | 0.6538 | 0.0932 | 0.5067 | 0.5858 | 0.0462 |
| 6 | 0.1121 | 0.6441 | 0.1910 | 0.5201 | 0.6287 | 0.1035 |

## Osservazioni
- Il modello mostra una recall migliore al livello di step rispetto al sub-step
- La categoria 0 ha la più alta precision (0.8053) ma recall più bassa (0.4773)
- Le categorie 3 e 5 hanno recall molto alta ma precision bassa, indicando molti falsi positivi
- Le metriche PR-AUC per la categoria 0 è particolarmente alta (0.8300)

---

# Risultati Valutazione LSTM Omnivore (Split)

## Configurazione
- **Variant**: LSTM
- **Backbone**: Omnivore
- **Split**: Step
- **Threshold**: 0.6
- **Checkpoint**: error_recognition_step_omnivore_lstm_video_best.pt
- **Epochs**: 50
- **Batch Size**: 32
- **Learning Rate**: 1e-3

## Metriche a Livello di Sub-Step

| Metrica | Valore |
|---------|--------|
| Precision | 0.5189 |
| Recall | 0.5835 |
| F1-Score | 0.5493 |
| Accuracy | 0.7320 |
| AUC-ROC | 0.7644 |
| PR-AUC | 0.4194 |

## Metriche a Livello di Step

| Metrica | Valore |
|---------|--------|
| Precision | 0.6514 |
| Recall | 0.5703 |
| F1-Score | 0.6081 |
| Accuracy | 0.7707 |
| AUC-ROC | 0.8256 |
| PR-AUC | 0.5056 |

## Metriche per Categoria (Step Level)

| Categoria | Precision | Recall | F1-Score | Accuracy | AUC-ROC | PR-AUC |
|-----------|-----------|--------|----------|----------|---------|--------|
| 0 | 0.7327 | 0.2301 | 0.3502 | 0.2607 | 0.2603 | 0.8353 |
| 2 | 0.1336 | 0.5918 | 0.2180 | 0.7393 | 0.7577 | 0.1042 |
| 3 | 0.0092 | 0.2500 | 0.0178 | 0.7231 | 0.6473 | 0.0098 |
| 4 | 0.1106 | 0.5714 | 0.1853 | 0.7356 | 0.7361 | 0.0858 |
| 5 | 0.0829 | 0.5294 | 0.1434 | 0.7306 | 0.7024 | 0.0640 |
| 6 | 0.1475 | 0.5161 | 0.2294 | 0.7306 | 0.7090 | 0.1137 |

## Osservazioni
- Questo modello (step split) mostra performance significativamente migliore rispetto al precedente (recordings split)
- L'AUC-ROC a livello di step è notevolmente più alto (0.8256 vs 0.6528)
- La precision a livello di step è migliorata (0.6514 vs 0.4602)
- La categoria 0 mantiene la più alta precision (0.7302) ma con recall più bassa (0.2272)
- Le categorie 2-6 mostrano un pattern simile: recall più alta ma precision più bassa, indicando tendenza a falsi positivi
- Il PR-AUC per la categoria 0 rimane eccezionalmente alto (0.8351)

---

# Risultati Valutazione LSTM SlowFast (Recordings)

## Configurazione
- **Variant**: LSTM
- **Backbone**: SlowFast
- **Split**: Recordings
- **Threshold**: 0.4
- **Checkpoint**: error_recognition_recordings_slowfast_lstm_video_best.pt
- **Epochs**: 50
- **Batch Size**: 32
- **Learning Rate**: 1e-3

## Metriche a Livello di Sub-Step

| Metrica | Valore |
|---------|--------|
| Precision | 0.3677 |
| Recall | 0.6430 |
| F1-Score | 0.4679 |
| Accuracy | 0.5191 |
| AUC-ROC | 0.5761 |
| PR-AUC | 0.3538 |

## Metriche a Livello di Step

| Metrica | Valore |
|---------|--------|
| Precision | 0.3699 |
| Recall | 0.9793 |
| F1-Score | 0.5370 |
| Accuracy | 0.3934 |
| AUC-ROC | 0.5647 |
| PR-AUC | 0.3697 |

## Metriche per Categoria (Step Level)

| Categoria | Precision | Recall | F1-Score | Accuracy | AUC-ROC | PR-AUC |
|-----------|-----------|--------|----------|----------|---------|--------|
| 0 | 0.8455 | 0.8899 | 0.8671 | 0.7675 | 0.4521 | 0.8463 |
| 2 | 0.0781 | 0.9592 | 0.1444 | 0.1699 | 0.6080 | 0.0779 |
| 3 | 0.0133 | 1.0000 | 0.0262 | 0.1148 | 0.5332 | 0.0133 |
| 4 | 0.0781 | 0.9400 | 0.1442 | 0.1684 | 0.5765 | 0.0779 |
| 5 | 0.0432 | 1.0000 | 0.0828 | 0.1416 | 0.5507 | 0.0432 |
| 6 | 0.0914 | 0.9322 | 0.1664 | 0.1788 | 0.5323 | 0.0911 |

## Osservazioni
- SlowFast con split recordings mostra recall estremamente elevata sia a livello step (0.9793) che nelle singole categorie (quasi tutte sopra 0.93)
- La categoria 0 presenta metriche eccellenti con precision 0.8455 e recall 0.8899, risultando ben bilanciata
- Le categorie 2-6 hanno recall quasi perfetta (0.94-1.00) ma precision molto bassa, indicando un bias verso falsi positivi
- L'AUC-ROC complessivo (0.5647) è il più basso tra tutti i modelli testati, suggerendo scarsa capacità discriminativa
- Confrontato con Omnivore recordings (AUC 0.6528), SlowFast ha performance inferiori nonostante l'alta recall
- Il modello sembra predire quasi sempre errore per le categorie 2-6, compromettendo la precision


---

# Risultati Valutazione LSTM SlowFast (Split)

## Configurazione
- **Variant**: LSTM
- **Backbone**: SlowFast
- **Split**: Step
- **Threshold**: 0.6
- **Checkpoint**: error_recognition_step_slowfast_lstm_video_best.pt
- **Epochs**: 50
- **Batch Size**: 32
- **Learning Rate**: 1e-3

## Metriche a Livello di Sub-Step

| Metrica | Valore |
|---------|--------|
| Precision | 0.3174 |
| Recall | 0.7996 |
| F1-Score | 0.4545 |
| Accuracy | 0.4626 |
| AUC-ROC | 0.6281 |
| PR-AUC | 0.3099 |

## Metriche a Livello di Step

| Metrica | Valore |
|---------|--------|
| Precision | 0.3319 |
| Recall | 0.9598 |
| F1-Score | 0.4933 |
| Accuracy | 0.3847 |
| AUC-ROC | 0.6390 |
| PR-AUC | 0.3311 |

## Metriche per Categoria (Step Level)

| Categoria | Precision | Recall | F1-Score | Accuracy | AUC-ROC | PR-AUC |
|-----------|-----------|--------|----------|----------|---------|--------|
| 0 | 0.7905 | 0.2402 | 0.3685 | 0.2870 | 0.3926 | 0.8478 |
| 2 | 0.0762 | 0.3265 | 0.1236 | 0.7155 | 0.6034 | 0.0662 |
| 3 | 0.0190 | 0.5000 | 0.0367 | 0.7368 | 0.6690 | 0.0145 |
| 4 | 0.0952 | 0.4762 | 0.1587 | 0.7343 | 0.6816 | 0.0729 |
| 5 | 0.0619 | 0.3824 | 0.1066 | 0.7268 | 0.5721 | 0.0500 |
| 6 | 0.0714 | 0.2419 | 0.1103 | 0.6967 | 0.5087 | 0.0762 |

## Osservazioni
- SlowFast mostra una recall straordinariamente alta a livello di step (0.9598) ma con precision molto bassa (0.3319)
- Questo indica un modello altamente sensibile che tende a produrre molti falsi positivi
- L'AUC-ROC è inferiore rispetto a Omnivore (0.6388 vs 0.8255), suggerendo performance complessiva peggiore
- La category 0 mantiene il PR-AUC più alto (0.8526) anche con SlowFast
- Le performance sono inferiori a quelle di Omnivore con lo stesso split, suggerendo che Omnivore sia più adatto per questo task

