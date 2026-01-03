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
