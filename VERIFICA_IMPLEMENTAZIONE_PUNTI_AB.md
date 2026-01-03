# Verifica Implementazione Punti A e B

## Sommario Esecutivo

✅ **PUNTO A - PARZIALMENTE IMPLEMENTATO**: Analisi per tipi di errore è presente nel codice
❌ **PUNTO B - NON IMPLEMENTATO CORRETTAMENTE**: LSTM baseline c'è ma non è confrontato con gli altri baseline

---

## PUNTO A: Analisi Performance su Diversi Tipi di Errore

### Stato Attuale: ✅ IMPLEMENTATO (ma incompleto)

#### 1. **Codice di Calcolo delle Metriche per Categoria** ✅
**File**: [base.py](base.py#L394-L456)

Il codice calcola metriche separate per ciascun tipo di errore:

```python
# Linee 394-456: Calcolo metriche per categoria di errore
metrics_per_category = {}
all_categories = set()

# Prima pass: identifica tutte le categorie
for i, (start, end) in enumerate(test_step_start_end_list):
    error_category = test_error_categories[i]
    all_categories.update(error_category)

# Inizializza il dizionario per tutte le categorie
for cat in all_categories:
    metrics_per_category[cat] = {'outputs': [], 'targets': []}

# Seconda pass: aggrega i dati per categoria
for i, (start, end) in enumerate(test_step_start_end_list):
    # ... codice di aggregazione ...
    for cat in all_categories:
        has_category = 1 if cat in error_category else 0
        metrics_per_category[cat]['outputs'].append(mean_step_output)
        metrics_per_category[cat]['targets'].append(has_category)

# Calcolo metriche per categoria di errore (linee 436-450)
category_metrics = {}
for category, data in metrics_per_category.items():
    category_outputs = np.array(data['outputs'])
    category_targets = np.array(data['targets'])
    
    pred_labels = (category_outputs > threshold).astype(int)
    precision = precision_score(category_targets, pred_labels, zero_division=0)
    recall = recall_score(category_targets, pred_labels)
    f1 = f1_score(category_targets, pred_labels)
    accuracy = accuracy_score(category_targets, pred_labels)
    auc = roc_auc_score(category_targets, category_outputs)
    pr_auc = binary_auprc(torch.tensor(pred_labels), torch.tensor(category_targets))
    
    category_metrics[category] = {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'accuracy': accuracy,
        'auc': auc,
        'pr_auc': pr_auc
    }
```

#### 2. **Categorie di Errore Supportate** ✅
**File**: [constants.py](constants.py#L61-L67)

```python
TECHNIQUE_ERROR = "Technique Error"
PREPARATION_ERROR = "Preparation Error"
TEMPERATURE_ERROR = "Temperature Error"
MEASUREMENT_ERROR = "Measurement Error"
TIMING_ERROR = "Timing Error"
```

**File**: [CaptainCookStepDataset.py](dataloader/CaptainCookStepDataset.py#L36-L48)

Le categorie vengono estratte dalle annotazioni e gestite correttamente:

```python
self._error_category_name_label_map = {
    'TechniqueError': const.TECHNIQUE_ERROR,
    'PreparationError': const.PREPARATION_ERROR,
    'TemperatureError': const.TEMPERATURE_ERROR,
    'MeasurementError': const.MEASUREMENT_ERROR,
    'TimingError': const.TIMING_ERROR
}
```

#### 3. **Output delle Metriche per Categoria** ✅
**File**: [base.py](base.py#L489)

```python
print(f"{phase} Step Level Metrics per Category: {category_metrics}")
```

Le metriche vengono stampate a console durante il test.

#### 4. **PROBLEMI IDENTIFICATI** ⚠️

a) **Metriche non salvate su file CSV**
   - Le metriche per categoria vengono stampate a console ma NON salvate nel CSV
   - [base.py](base.py#L87-L105) in `save_results_to_csv()` salva solo metriche globali
   - **Soluzione necessaria**: Estendere `save_results_to_csv()` per salvare metriche per categoria

b) **Metriche non loggabili su Weights & Biases**
   - In `train_model_base()` (linee 246-260), il logging su wandb non include metriche per categoria
   - **Soluzione necessaria**: Aggiungere metriche per categoria al logging wandb

c) **Funzione `test_er_model()` compatta ma incompleta**
   - Calcola bene le metriche per categoria
   - Ma non le integra nel workflow di salvataggio risultati
   - **Flusso incompleto**: calcolo → stampa → ma non salvataggio persistente

---

## PUNTO B: Baseline LSTM Alternativo

### Stato Attuale: ⚠️ PARZIALMENTE IMPLEMENTATO

#### 1. **Modello LSTM Implementato** ✅
**File**: [core/models/er_lstm.py](core/models/er_lstm.py)

Il baseline LSTM è stato implementato:

```python
class ErLSTM(nn.Module):
    """
    LSTM-based baseline per Step-level Error Recognition.
    Restituisce un output per ogni step della sequenza.
    """
    def __init__(self, config, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        input_dim = fetch_input_dim(config)
        
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=256,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )
        
        lstm_out_dim = 256 * 2  # bidirezionale
        self.decoder = MLP(lstm_out_dim, 512, 1)
```

**Caratteristiche**:
- LSTM bidirezionale con 256 unità nascoste
- Output: (B, T, 1) - una predizione per step
- Decoder MLP per conversione output

#### 2. **Integrazione nel Codice di Training** ✅
**File**: [base.py](base.py#L58-L65)

```python
elif config.variant == const.LSTM_VARIANT:
    if config.backbone in [const.OMNIVORE, const.RESNET3D, const.X3D, 
                           const.SLOWFAST, const.IMAGEBIND]:
        model = ErLSTM(config)
```

L'LSTM è:
- ✅ Selezionabile tramite argomento `--variant lstm`
- ✅ Compatibile con tutti i backbone (OMNIVORE, RESNET3D, X3D, SLOWFAST, IMAGEBIND)

#### 3. **Variant Disponibili** ✅
**File**: [constants.py](constants.py#L48-L53)

```python
MLP_VARIANT = "MLP"
TRANSFORMER_VARIANT = "Transformer"
MULTIMODAL_VARIANT = "Multimodal"
LSTM_VARIANT = "lstm"
```

#### 4. **PROBLEMI IDENTIFICATI** ⚠️

a) **LSTM non confrontato con baseline originali**
   - ❌ No risultati comparativi nel README
   - ❌ No script di valutazione dedicato per LSTM
   - ❌ No metriche salvate vs MLP/Transformer

b) **Mancano esperimenti di training su LSTM**
   - Il codice training generale supporta LSTM
   - Ma NON ci sono checkpoint pre-addestrati per LSTM
   - NON ci sono istruzioni per allenarsi con LSTM

c) **No analisi qualitativa**
   - Manca: Quando LSTM funziona meglio/peggio rispetto a MLP/Transformer?
   - Manca: Su quali tipi di errore eccelle?
   - Manca: Analisi tempi di training/inferenza

d) **No ablation study**
   - Non è chiaro se numero layer LSTM (attualmente 1) è ottimale
   - Hidden size (256) non è motivato
   - Bidirezione vs unidirezionale non è motivato

---

## Raccomandazioni per Completamento

### Per PUNTO A (Analisi per Tipo di Errore):

1. **IMMEDIATO**: Estendere `save_results_to_csv()` per salvare metriche per categoria
   ```python
   # Aggiungere colonne nel CSV:
   # - Per ogni categoria: Precision, Recall, F1, AUC, PR_AUC
   # - Ad esempio: "Technique_Error_Precision", "Technique_Error_Recall", ...
   ```

2. **IMPORTANTE**: Aggiungere logging wandb per metriche per categoria
   ```python
   wandb.log({
       "category_metrics": category_metrics,
       "val_metrics": { ... },
       "test_metrics": { ... }
   })
   ```

3. **UTILE**: Creare script di visualizzazione per metriche per categoria
   - Plot F1/AUC per tipo di errore
   - Matrice confusione per categoria
   - Distribuzione errori nel dataset

### Per PUNTO B (Baseline LSTM):

1. **CRITICO**: Aggiungere esperimenti di training su LSTM
   ```bash
   python train_er.py --variant lstm --backbone omnivore --split recordings
   ```

2. **CRITICO**: Salvare checkpoint best model LSTM
   - Creare struttura: `checkpoints/error_recognition_best/lstm/omnivore/`

3. **IMPORTANTE**: Aggiungere tabella comparativa nel README
   ```markdown
   | Modello | Split | F1 (%) | AUC (%) |
   |---------|-------|--------|---------|
   | MLP | Recordings | 55.42 | 63.03 |
   | Transformer | Recordings | 40.73 | 62.27 |
   | LSTM | Recordings | ?? | ?? |
   ```

4. **IMPORTANTE**: Documenti di analisi
   - "LSTM vs Transformer: Quando usare quale?"
   - "Impact dell'LSTM hidden size e num_layers"
   - "Performance per tipo di errore per LSTM"

---

## Checklist di Implementazione

### Punto A:
- [ ] Estendere CSV per metriche per categoria
- [ ] Aggiungere wandb logging per categoria
- [ ] Aggiungere visualizzazione metrica per categoria
- [ ] Documentare risultati analisi per categoria

### Punto B:
- [ ] Addestrare LSTM baseline
- [ ] Salvare checkpoint LSTM best
- [ ] Tabella comparativa nel README
- [ ] Documentazione ablation study LSTM
- [ ] Analisi qualitativa LSTM vs baseline

---

## File Coinvolti

### File che richiedono modifiche:
- [base.py](base.py) - Estendere salvataggio CSV e wandb logging
- [README.md](README.md) - Aggiungere istruzioni e risultati LSTM
- [train_er.py](train_er.py) - (Verificare se OK)

### File già OK:
- [core/models/er_lstm.py](core/models/er_lstm.py) ✅
- [constants.py](constants.py) ✅
- [dataloader/CaptainCookStepDataset.py](dataloader/CaptainCookStepDataset.py) ✅
