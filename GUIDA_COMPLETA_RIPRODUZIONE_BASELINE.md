# Guida Completa: Riproduzione Baseline V1 e V2 del Paper CaptainCook4D

**Task**: SupervisedER - Error Recognition  
**Baseline V1**: MLP-based Model  
**Baseline V2**: Transformer-based Model (ErFormer)  
**Metriche**: Accuracy, Precision, Recall, F1, AUC

---

## Indice

1. [Panoramica del Task](#1-panoramica-del-task)
2. [Architettura del Progetto](#2-architettura-del-progetto)
3. [Setup dell'Ambiente](#3-setup-dellambiente)
4. [Organizzazione delle Feature](#4-organizzazione-delle-feature)
5. [Funzioni Chiave del Codice](#5-funzioni-chiave-del-codice)
6. [Esecuzione della Valutazione](#6-esecuzione-della-valutazione)
7. [Training dei Modelli](#7-training-dei-modelli)
8. [Interpretazione dei Risultati](#8-interpretazione-dei-risultati)

---

## 1. Panoramica del Task

### 1.1 Obiettivo
Il task **SupervisedER (Supervised Error Recognition)** consiste nel rilevare se uno step di una ricetta culinaria contiene errori oppure no. Si tratta di un problema di **classificazione binaria** a livello di step.

### 1.2 Dataset: CaptainCook4D
- **Input**: Feature video pre-estratte da backbone (Omnivore, SlowFast)
- **Granularità**: Segmenti di 1 secondo
- **Annotazioni**: Step-level con label binarie (error/no-error) e categorie di errore
- **Split**: 3 modalità
  - `recordings`: train/val/test divisi per recording diversi
  - `person`: divisi per persona che esegue l'azione
  - `environment`: divisi per ambiente di cucina
  - `step`: divisi per tipo di step (non usato nel paper principale)

### 1.3 Baseline da Riprodurre

| Modello | Split | F1 (%) | AUC (%) | Threshold |
|---------|-------|--------|---------|-----------|
| **V1: MLP (Omnivore)** | Step | 24.26 | 75.74 | 0.6 |
| **V1: MLP (Omnivore)** | Recordings | 55.42 | 63.03 | 0.4 |
| **V2: Transformer (Omnivore)** | Step | 55.39 | 75.62 | 0.6 |
| **V2: Transformer (Omnivore)** | Recordings | 40.73 | 62.27 | 0.4 |

---

## 2. Architettura del Progetto

```
aml-2025-mistake-detection/
├── base.py                          # Funzioni base per train/test
├── train_er.py                      # Script principale per training
├── constants.py                     # Costanti del progetto
├── setup_features_colab.py          # Script per organizzare feature su Colab
├── requirements.txt                 # Dipendenze Python
├── annotations/                     # Annotazioni raw
├── er_annotations/                  # Split JSON per train/val/test
│   ├── recordings_combined_splits.json
│   ├── person_combined_splits.json
│   └── environment_combined_splits.json
├── core/
│   ├── config.py                    # Configurazione argomenti
│   ├── evaluate.py                  # Script di valutazione
│   ├── utils.py                     # Utility functions
│   └── models/
│       ├── blocks.py                # Componenti comuni (MLP, PositionalEncoding)
│       ├── er_former.py             # Baseline V2: Transformer
│       └── er_lstm.py               # Baseline alternativo: LSTM
├── dataloader/
│   ├── CaptainCookStepDataset.py    # Dataset per step-level
│   └── CaptainCookSubStepDataset.py # Dataset per sub-step-level
├── data/                            # Feature estratte (da creare)
│   └── video/
│       ├── omnivore/
│       │   └── {recording_id}_360p.mp4_1s_1s.npz
│       └── slowfast/
│           └── {recording_id}_360p.mp4_1s_1s.npz
└── checkpoints/                     # Modelli pre-addestrati
    └── error_recognition_best/
        ├── MLP/
        │   └── omnivore/
        └── Transformer/
            └── omnivore/
```

---

## 3. Setup dell'Ambiente

### 3.1 Installazione Dipendenze

```bash
# Crea ambiente virtuale
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# oppure
.venv\Scripts\activate     # Windows

# Installa dipendenze
pip install -r requirements.txt
```

**Dipendenze principali**:
- `torch`: Deep learning framework
- `scikit-learn`: Metriche di valutazione
- `torcheval`: Metriche aggiuntive (binary_auprc)
- `numpy`: Manipolazione array
- `tqdm`: Progress bar
- `wandb` (opzionale): Logging esperimenti

### 3.2 Download Risorse

1. **Feature pre-estratte** (obbligatorio):
   - Scaricare per i segments 1s le feature di omnivore e slowfast

2. **Checkpoint pre-addestrati** (per valutazione):
   - Dalla stessa cartella Box, scarica `error_recognition_best/`
   - Posiziona in `checkpoints/error_recognition_best/`

---

## 4. Organizzazione delle Feature

### 4.1 Struttura delle Feature

Le feature sono salvate in formato **`.npz`** (NumPy compressed):

```python
# Esempio di caricamento
features_data = np.load('recording_1_360p.mp4_1s_1s.npz')
features = features_data['arr_0']  # Shape: (num_seconds, feature_dim)
```

**Dimensioni delle feature**:
- **Omnivore**: 1024 dimensioni (backbone multimodale)
- **SlowFast**: 400 dimensioni (slow pathway 256 + fast pathway 144)

### 4.2 Script di Organizzazione (Google Colab)

Lo script `setup_features_colab.py` automatizza l'organizzazione:

```python
# Su Google Colab
!python setup_features_colab.py
```

**Cosa fa lo script**:
1. **Monta Google Drive** (se disponibile)
2. **Estrae `features.zip`** → cartella temporanea
3. **Estrae i nested zip** (`omnivore.zip`, `slowfast.zip`)
4. **Organizza i file `.npz`** in:
   - `data/video/omnivore/*.npz`
   - `data/video/slowfast/*.npz`
   - `data/audio/*.npz` (se presenti)
5. **Verifica** il conteggio file
6. **Cleanup** delle directory temporanee

---

## 5. Funzioni Chiave del Codice

### 5.1 Dataset Loader (`CaptainCookStepDataset`)

**File**: `dataloader/CaptainCookStepDataset.py`

#### 5.1.1 `__init__(config, phase, split)`
Inizializza il dataset caricando annotazioni e split.

**Parametri**:
- `config`: Oggetto di configurazione con backbone, modality, ecc.
- `phase`: `"train"`, `"val"`, o `"test"`
- `split`: `"recordings"`, `"person"`, `"environment"`, o `"step"`

**Cosa fa**:
```python
# 1. Carica annotazioni step (start_time, end_time, step_id)
with open('annotations/annotation_json/step_annotations.json') as f:
    self._annotations = json.load(f)

# 2. Carica annotazioni errori (has_errors, error_categories)
with open('annotations/annotation_json/error_annotations.json') as f:
    self._error_annotations = json.load(f)

# 3. Carica split train/val/test
split_file = f"er_annotations/{split}_combined_splits.json"
with open(split_file) as f:
    self._recording_ids_json = json.load(f)
    self._recording_ids = self._recording_ids_json[phase]
```

#### 5.1.2 `_prepare_recording_step_dictionary(recording_id)`
Prepara un dizionario step_id → lista di (start, end, has_errors, error_categories).

**Output**: 
```python
{
    'step_1': [(10, 25, True, {6, 2})],   # Step con errori
    'step_2': [(30, 45, False, {0})],      # Step senza errori
}
```

#### 5.1.3 `__getitem__(idx)`
Restituisce le feature e label per uno step.

**Processo**:
```python
# 1. Ottieni recording_id e time boundaries
recording_id, step_start_end_list = self._step_dict[idx]

# 2. Carica feature dal file .npz
features_path = f"data/video/{backbone}/{recording_id}_360p.mp4_1s_1s.npz"
recording_features = np.load(features_path)['arr_0']  # Shape: (T, D)

# 3. Estrai segmento temporale
step_features = recording_features[start:end, :]  # Shape: (T_step, D)

# 4. Crea label (1 se ha errori, 0 altrimenti)
step_labels = torch.ones(T_step, 1) if has_errors else torch.zeros(T_step, 1)

return step_features, step_labels, error_categories
```

#### 5.1.4 `collate_fn(batch)`
Combina più sample in un batch concatenando lungo la dimensione temporale.

```python
def collate_fn(batch):
    step_features, step_labels, error_categories = zip(*batch)
    
    # Concatena tutti gli step in un unico tensore
    step_features = torch.cat(step_features, dim=0)  # (Total_T, D)
    step_labels = torch.cat(step_labels, dim=0)      # (Total_T, 1)
    
    return step_features, step_labels, error_categories
```

---

### 5.2 Modelli

#### 5.2.1 **V1: MLP Baseline** (`blocks.py`)

```python
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = torch.relu(self.layer1(x))  # Hidden layer con ReLU
        x = self.layer2(x)               # Output layer (logits)
        return x
```

**Caratteristiche**:
- **Input**: Feature di dimensione 1024 (Omnivore) o 400 (SlowFast)
- **Hidden**: 512 unità
- **Output**: 1 logit (classificazione binaria)
- **Attivazione**: ReLU dopo il primo layer
- **Loss**: BCEWithLogitsLoss (applica sigmoid internamente)

**Istanziazione**:
```python
model = MLP(input_dim=1024, hidden_size=512, output_size=1)
```

#### 5.2.2 **V2: Transformer Baseline** (`er_former.py`)

```python
class ErFormer(nn.Module):
    def __init__(self, config):
        super().__init__()
        input_dim = fetch_input_dim(config)  # 1024 per Omnivore
        
        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_dim,
            nhead=8,
            dim_feedforward=2048,
            batch_first=True
        )
        self.step_encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)
        
        # MLP Decoder
        self.decoder = MLP(1024, 512, 1)

    def forward(self, input_data):
        # 1. Gestione NaN
        input_data = torch.nan_to_num(input_data, nan=0.0)
        
        # 2. Transformer encoding
        encoded_output = self.step_encoder(input_data)  # (B, T, D)
        
        # 3. Decodifica con MLP
        final_output = self.decoder(encoded_output)  # (B, T, 1)
        
        return final_output
```

**Caratteristiche**:
- **Self-attention**: 8 teste di attenzione
- **Feed-forward**: 2048 dimensioni interne
- **Layers**: 1 layer di transformer encoder
- **Decoder**: MLP (1024 → 512 → 1)

**Vantaggi rispetto a MLP**:
- Cattura dipendenze temporali tra frame
- Modella relazioni long-range all'interno dello step

---

### 5.3 Training Loop (`base.py`)

#### 5.3.1 `train_model_base(train_loader, val_loader, config, test_loader)`

Funzione principale per addestrare il modello.

**Processo**:
```python
# 1. Inizializza modello
model = fetch_model(config)  # MLP o Transformer in base a config.variant

# 2. Ottimizzatore e loss
optimizer = optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([2.5]))  # Class balancing

# 3. Scheduler (riduce LR se AUC non migliora)
scheduler = ReduceLROnPlateau(
    optimizer, mode='max', factor=0.1, patience=5, min_lr=1e-7
)

# 4. Training loop
for epoch in range(1, config.num_epochs + 1):
    # Train
    model.train()
    for data, target, _ in train_loader:
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
    
    # Validation
    val_losses, sub_step_metrics, step_metrics = test_er_model(
        model, val_loader, criterion, device, phase='val'
    )
    
    # Update learning rate
    scheduler.step(step_metrics['auc'])
    
    # Save checkpoint
    store_model(model, config, ckpt_name=f"{model_name}_epoch_{epoch}.pt")
    
    # Track best model
    if step_metrics['auc'] > best_auc:
        best_model_state = model.state_dict()
```

**Loss Function**:
```python
criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([2.5]))
```
- `BCEWithLogitsLoss`: Combina sigmoid + binary cross entropy (numericamente stabile)
- `pos_weight=2.5`: Penalizza di più gli errori sugli esempi positivi (class imbalance)

**Gradient Clipping**:
```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```
- Previene exploding gradients limitando la norma dei gradienti a 1.0

---

### 5.4 Evaluation (`test_er_model` in `base.py`)

#### 5.4.1 Calcolo Metriche a Livello Sub-Step

```python
def test_er_model(model, test_loader, criterion, device, phase, threshold=0.6):
    all_outputs = []
    all_targets = []
    
    # Inference
    with torch.no_grad():
        for data, target, error_categories in test_loader:
            output = model(data)
            sigmoid_output = output.sigmoid()  # Converti logits in probabilità
            all_outputs.append(sigmoid_output.cpu().numpy())
            all_targets.append(target.cpu().numpy())
    
    all_outputs = np.concatenate(all_outputs)  # (N_total_frames,)
    all_targets = np.concatenate(all_targets)  # (N_total_frames,)
    
    # Metriche sub-step (frame-level)
    pred_labels = (all_outputs > threshold).astype(int)
    sub_step_precision = precision_score(all_targets, pred_labels)
    sub_step_recall = recall_score(all_targets, pred_labels)
    sub_step_f1 = f1_score(all_targets, pred_labels)
    sub_step_accuracy = accuracy_score(all_targets, pred_labels)
    sub_step_auc = roc_auc_score(all_targets, all_outputs)
```

#### 5.4.2 Aggregazione a Livello Step

```python
    # Metriche step-level (aggregazione per step)
    all_step_outputs = []
    all_step_targets = []
    
    for i, (start, end) in enumerate(test_step_start_end_list):
        step_output = all_outputs[start:end]  # Output del singolo step
        step_target = all_targets[start:end]  # Target del singolo step
        
        # Normalizzazione (opzionale)
        if sub_step_normalization:
            prob_range = np.max(step_output) - np.min(step_output)
            if prob_range > 0:
                step_output = (step_output - np.min(step_output)) / prob_range
        
        # Aggregazione: media delle probabilità
        mean_step_output = np.mean(step_output)
        
        # Target binario: 1 se >95% dei frame sono positivi
        step_target_binary = 1 if np.mean(step_target) > 0.95 else 0
        
        all_step_outputs.append(mean_step_output)
        all_step_targets.append(step_target_binary)
    
    # Normalizzazione globale (opzionale)
    if step_normalization:
        all_step_outputs = (all_step_outputs - min) / (max - min)
    
    # Calcola metriche
    pred_step_labels = (all_step_outputs > threshold).astype(int)
    precision = precision_score(all_step_targets, pred_step_labels)
    recall = recall_score(all_step_targets, pred_step_labels)
    f1 = f1_score(all_step_targets, pred_step_labels)
    auc = roc_auc_score(all_step_targets, all_step_outputs)
```

**Threshold**:
- Il threshold (0.4 o 0.6) determina la soglia di probabilità per classificare come "errore"
- Varia tra split perché il dataset ha bilanciamento diverso

---

### 5.5 Utility Functions

#### 5.5.1 `fetch_model(config)` in `base.py`

Istanzia il modello corretto in base alla configurazione.

```python
def fetch_model(config):
    if config.variant == const.MLP_VARIANT:
        input_dim = fetch_input_dim(config)
        model = MLP(input_dim, 512, 1)
    elif config.variant == const.TRANSFORMER_VARIANT:
        model = ErFormer(config)
    elif config.variant == const.LSTM_VARIANT:
        model = ErLSTM(config)
    
    model.to(config.device)
    return model
```

#### 5.5.2 `fetch_input_dim(config)` in `blocks.py`

Determina la dimensione di input in base al backbone.

```python
def fetch_input_dim(config, decoder=False):
    if config.backbone == const.OMNIVORE:
        return 1024
    elif config.backbone == const.SLOWFAST:
        return 400
    elif config.backbone == const.X3D:
        return 400
    elif config.backbone == const.RESNET3D:
        return 400
    elif config.backbone == const.IMAGEBIND:
        k = len(config.modality)  # Multimodale
        return 1024 * k
```

#### 5.5.3 `store_model(model, config, ckpt_name)` in `base.py`

Salva i pesi del modello.

```python
def store_model(model, config, ckpt_name):
    task_directory = os.path.join(config.ckpt_directory, config.task_name)
    variant_directory = os.path.join(task_directory, config.variant)
    backbone_directory = os.path.join(variant_directory, config.backbone)
    
    os.makedirs(backbone_directory, exist_ok=True)
    
    ckpt_file_path = os.path.join(backbone_directory, ckpt_name)
    torch.save(model.state_dict(), ckpt_file_path)
```

---

## 6. Esecuzione della Valutazione

### 6.1 Script di Evaluation

**File**: `core/evaluate.py`

```python
# Carica configurazione
conf = Config()
conf.split = args.split
conf.backbone = args.backbone
conf.variant = args.variant
conf.ckpt_directory = args.ckpt

# Carica modello
model = fetch_model(conf)
model.load_state_dict(torch.load(conf.ckpt_directory))
model.eval()

# Carica dataset di test
test_dataset = CaptainCookStepDataset(conf, const.TEST, conf.split)
test_loader = DataLoader(test_dataset, batch_size=1, collate_fn=collate_fn)

# Valuta
test_er_model(model, test_loader, criterion, conf.device, phase="test", threshold=args.threshold)
```

### 6.2 Comandi di Esecuzione

#### V1 - MLP (Omnivore, Split: Step)
```bash
python -m core.evaluate \
  --variant MLP \
  --backbone omnivore \
  --ckpt checkpoints/error_recognition_best/MLP/omnivore/error_recognition_MLP_omnivore_step_epoch_43.pt \
  --split step \
  --threshold 0.6
```

**Output atteso**:
```
----------------------------------------------------------------
test Sub Step Level Metrics: {'precision': 0.XX, 'recall': 0.XX, ...}
test Step Level Metrics: {'precision': 0.XX, 'recall': 0.XX, 'f1': 0.2426, 'auc': 0.7574, ...}
----------------------------------------------------------------
```

#### V1 - MLP (Omnivore, Split: Recordings)
```bash
python -m core.evaluate \
  --variant MLP \
  --backbone omnivore \
  --ckpt checkpoints/error_recognition_best/MLP/omnivore/error_recognition_MLP_omnivore_recordings_epoch_XX.pt \
  --split recordings \
  --threshold 0.4
```

**Output atteso**:
- F1: 55.42%
- AUC: 63.03%

#### V2 - Transformer (Omnivore, Split: Step)
```bash
python -m core.evaluate \
  --variant Transformer \
  --backbone omnivore \
  --ckpt checkpoints/error_recognition_best/Transformer/omnivore/error_recognition_Transformer_omnivore_step_epoch_XX.pt \
  --split step \
  --threshold 0.6
```

**Output atteso**:
- F1: 55.39%
- AUC: 75.62%

#### V2 - Transformer (Omnivore, Split: Recordings)
```bash
python -m core.evaluate \
  --variant Transformer \
  --backbone omnivore \
  --ckpt checkpoints/error_recognition_best/Transformer/omnivore/error_recognition_Transformer_omnivore_recordings_epoch_XX.pt \
  --split recordings \
  --threshold 0.4
```

**Output atteso**:
- F1: 40.73%
- AUC: 62.27%

---

## 7. Training dei Modelli

### 7.1 Script di Training

**File**: `train_er.py`

```python
def main():
    conf = Config()
    conf.task_name = const.ERROR_RECOGNITION
    
    if conf.enable_wandb:
        init_logger_and_wandb(conf)
    
    train_step_test_step_er(conf)
    
    if conf.enable_wandb:
        wandb.finish()
```

### 7.2 Comandi di Training

#### Training V1 - MLP
```bash
python train_er.py \
  --variant MLP \
  --backbone omnivore \
  --split step \
  --num_epochs 50 \
  --batch_size 1 \
  --lr 1e-3 \
  --weight_decay 1e-3 \
  --seed 1000 \
  --ckpt_directory ./checkpoints
```

**Parametri importanti**:
- `--variant MLP`: Usa il modello MLP
- `--backbone omnivore`: Usa feature Omnivore (1024-dim)
- `--split step`: Usa step-split
- `--num_epochs 50`: Numero di epoche
- `--lr 1e-3`: Learning rate iniziale
- `--weight_decay 1e-3`: Regolarizzazione L2

#### Training V2 - Transformer
```bash
python train_er.py \
  --variant Transformer \
  --backbone omnivore \
  --split recordings \
  --num_epochs 50 \
  --batch_size 1 \
  --lr 1e-3 \
  --weight_decay 1e-3 \
  --seed 1000 \
  --ckpt_directory ./checkpoints
```

### 7.3 Monitoraggio Training

Durante il training, vengono salvati:

1. **Checkpoint per epoch**:
   ```
   checkpoints/error_recognition/Transformer/omnivore/
   └── error_recognition_Transformer_omnivore_recordings_epoch_1.pt
   └── error_recognition_Transformer_omnivore_recordings_epoch_2.pt
   └── ...
   └── error_recognition_Transformer_omnivore_recordings_best.pt
   ```

2. **Training statistics**:
   ```
   stats/error_recognition/Transformer/omnivore/
   └── error_recognition_Transformer_omnivore_recordings_training_performance.txt
   ```
   
   Contenuto:
   ```
   Epoch, Train Loss, Test Loss, Precision, Recall, F1, AUC
   1, 0.523456, 0.456789, 0.45, 0.32, 0.37, 0.65
   2, 0.489012, 0.432145, 0.48, 0.35, 0.40, 0.67
   ...
   ```

3. **Wandb logs** (se abilitato):
   - Metriche real-time
   - Grafici di loss/accuracy
   - Confronto tra esperimenti

---

## 8. Interpretazione dei Risultati

### 8.1 Metriche Utilizzate

#### 8.1.1 Accuracy
```python
accuracy = (TP + TN) / (TP + TN + FP + FN)
```
- Percentuale di predizioni corrette
- **Limitazione**: Fuorviante con class imbalance

#### 8.1.2 Precision
```python
precision = TP / (TP + FP)
```
- Percentuale di errori predetti che sono realmente errori
- **Alta precision**: Pochi falsi positivi

#### 8.1.3 Recall (Sensitivity)
```python
recall = TP / (TP + FN)
```
- Percentuale di errori reali che sono stati rilevati
- **Alto recall**: Pochi falsi negativi

#### 8.1.4 F1-Score
```python
f1 = 2 * (precision * recall) / (precision + recall)
```
- Media armonica di precision e recall
- **Metrica primaria** per il task (bilancia precision/recall)

#### 8.1.5 AUC (Area Under ROC Curve)
```python
auc = roc_auc_score(y_true, y_scores)
```
- Misura la capacità di distinguere tra classi a tutti i threshold
- **Range**: 0.5 (random) - 1.0 (perfetto)
- **AUC > 0.7**: Buona discriminazione

### 8.2 Confronto V1 vs V2

| Aspetto | V1 (MLP) | V2 (Transformer) |
|---------|----------|------------------|
| **Complessità** | Bassa (2 layer) | Media (1 encoder + decoder) |
| **Parametri** | ~500K | ~2M |
| **Capacità temporale** | Nessuna (frame indipendenti) | Self-attention tra frame |
| **F1 (Step)** | 24.26% | **55.39%** ✓ |
| **AUC (Step)** | 75.74% | 75.62% |
| **F1 (Recordings)** | **55.42%** ✓ | 40.73% |
| **AUC (Recordings)** | **63.03%** ✓ | 62.27% |
| **Training time** | Veloce | Lento |

**Osservazioni**:
- **Transformer > MLP** su step-split (F1 +131% relativo)
- **MLP > Transformer** su recordings-split
- **Possibile causa**: Overfitting del Transformer su recordings (meno dati di train)

### 8.3 Analisi per Categoria di Errore

Il codice calcola anche metriche per categoria:

```python
category_metrics = {
    'Technique Error': {'precision': 0.45, 'recall': 0.52, 'f1': 0.48, ...},
    'Preparation Error': {'precision': 0.38, 'recall': 0.41, 'f1': 0.39, ...},
    'Temperature Error': {'precision': 0.29, 'recall': 0.35, 'f1': 0.31, ...},
    'Measurement Error': {'precision': 0.42, 'recall': 0.39, 'f1': 0.40, ...},
    'Timing Error': {'precision': 0.51, 'recall': 0.48, 'f1': 0.49, ...},
}
```

Questo permette di capire quali errori sono più difficili da rilevare.

### 8.4 Threshold Tuning

Il threshold ottimale varia tra split:

| Split | Threshold Ottimale | Motivazione |
|-------|-------------------|-------------|
| Step | 0.6 | Dataset più bilanciato |
| Recordings | 0.4 | Più esempi negativi |
| Person | 0.5 | Bilanciamento intermedio |
| Environment | 0.5 | Bilanciamento intermedio |

**Come scegliere il threshold**:
1. Plotta la curva Precision-Recall
2. Scegli il punto che massimizza F1
3. Oppure bilancia in base al costo di FP vs FN

---

## 9. Troubleshooting

### 9.1 Problemi Comuni

#### 9.1.1 FileNotFoundError: Cannot find features
**Causa**: Feature non organizzate correttamente

**Soluzione**:
```bash
# Verifica struttura
ls data/video/omnivore/
ls data/video/slowfast/

# Riesegui setup
python setup_features_colab.py
```

#### 9.1.2 CUDA Out of Memory
**Causa**: Batch size troppo grande o sequenze troppo lunghe

**Soluzione**:
```bash
# Riduci batch size
python train_er.py --batch_size 1 ...

# Oppure usa CPU (lento)
python train_er.py --device cpu ...
```

#### 9.1.3 Metriche molto diverse dal paper
**Possibili cause**:
- Seed diverso (`--seed 1000`)
- Threshold non ottimale
- Feature estratte in modo diverso
- Split file corrotto

**Debug**:
```bash
# Verifica split
python -c "import json; print(json.load(open('er_annotations/recordings_combined_splits.json')))"

# Verifica feature shape
python -c "import numpy as np; print(np.load('data/video/omnivore/recording_1_360p.mp4_1s_1s.npz')['arr_0'].shape)"
```

## 10. Riferimenti

### 10.1 Paper Originale
**CaptainCook4D: A Multi-View Video Dataset for 4D Understanding of Cooking Activity**
- Authors: Rohith Peddi, et al.
- Conference: CVPR 2024
- Link: [arXiv](https://arxiv.org/abs/2312.14383)

### 10.2 Repository Ufficiale
- **Error Recognition**: https://github.com/CaptainCook4D/error_recognition
- **Feature Extractors**: https://github.com/CaptainCook4D/feature_extractors

### 10.3 Risorse Aggiuntive
- **Omnivore Model**: https://github.com/facebookresearch/omnivore
- **SlowFast Networks**: https://github.com/facebookresearch/SlowFast

**Fine della Guida**

Per domande o problemi, consulta il README.md o apri un issue sul repository.
