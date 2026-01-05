# Risultati Valutazione LSTM Omnivore (Recordings)

## Configurazione
- **Variant**: LSTM
- **Backbone**: Omnivore
- **Split**: Recordings
- **Threshold**: 0.4
- **Checkpoint**: error_recognition_recordings_omnivore_lstm_video_epoch_18.pt

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
- **Checkpoint**: error_recognition_step_omnivore_lstm_video_epoch_26.pt
- **Epochs**: 50
- **Batch Size**: 32
- **Learning Rate**: 1e-3

## Metriche a Livello di Sub-Step

| Metrica | Valore |
|---------|--------|
| Precision | 0.5190 |
| Recall | 0.5806 |
| F1-Score | 0.5481 |
| Accuracy | 0.7320 |
| AUC-ROC | 0.7645 |
| PR-AUC | 0.4187 |

## Metriche a Livello di Step

| Metrica | Valore |
|---------|--------|
| Precision | 0.6605 |
| Recall | 0.5703 |
| F1-Score | 0.6121 |
| Accuracy | 0.7744 |
| AUC-ROC | 0.8255 |
| PR-AUC | 0.5107 |

## Metriche per Categoria (Step Level)

| Categoria | Precision | Recall | F1-Score | Accuracy | AUC-ROC | PR-AUC |
|-----------|-----------|--------|----------|----------|---------|--------|
| 0 | 0.7302 | 0.2272 | 0.3466 | 0.2581 | 0.2605 | 0.8351 |
| 2 | 0.1349 | 0.5918 | 0.2197 | 0.7419 | 0.7578 | 0.1049 |
| 3 | 0.0093 | 0.2500 | 0.0179 | 0.7256 | 0.6491 | 0.0098 |
| 4 | 0.1116 | 0.5714 | 0.1868 | 0.7381 | 0.7371 | 0.0863 |
| 5 | 0.0837 | 0.5294 | 0.1446 | 0.7331 | 0.7015 | 0.0644 |
| 6 | 0.1488 | 0.5161 | 0.2310 | 0.7331 | 0.7075 | 0.1144 |

## Osservazioni
- Questo modello (step split) mostra performance significativamente migliore rispetto al precedente (recordings split)
- L'AUC-ROC a livello di step è notevolmente più alto (0.8255 vs 0.6528)
- La precision a livello di step è migliorata (0.6605 vs 0.4602)
- La categoria 0 mantiene la più alta precision (0.7302) ma con recall più bassa (0.2272)
- Le categorie 2-6 mostrano un pattern simile: recall più alta ma precision più bassa, indicando tendenza a falsi positivi
- Il PR-AUC per la categoria 0 rimane eccezionalmente alto (0.8351)
