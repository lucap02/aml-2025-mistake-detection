# AML/DAAI 2025 - Mistake Detection Project

## Environment Setup

First of all, create a python environment with 

```
python -m venv .venv
pip install -r requirements.txt
```

Then, download the pre-extracted features for 1s segments and put them in the `data/features` directory.

## Step 1: Baselines reproduction
Download the official best checkpoints from [here](https://utdallas.app.box.com/s/uz3s1alrzucz03sleify8kazhuc1ksl3) (`error_recognition_best` directory) and place them in the `checkpoints`. Then run the evaluation for the error recognition task.

**Example command**:
```
python -m core.evaluate --variant MLP --backbone omnivore --ckpt checkpoints/error_recognition_best/MLP/omnivore/error_recognition_MLP_omnivore_step_epoch_43.pt --split step --threshold 0.6
```

You should be able to reproduce results close to those reported in the paper (Table 2):

| Split | Model | F1 | AUC |
|-------|-------|----|-----|
| Step | MLP (Omnivore) | 24.26 | 75.74 |
| Recordings | MLP (Omnivore) | 55.42 | 63.03 |
| Step | Transf. (Omnivore) | 55.39 | 75.62 |
| Recordings | Transf. (Omnivore) | 40.73 | 62.27 |

**NOTE**: Use the thresholds indicated in the official README.md of project (0.6 for step and 0.4 for recordings steps).

## Acknowledgements

This project builds on many repositories from the CaptainCook4D release. Please refer to the original codebases for more details.

**Error Recognition**: https://github.com/CaptainCook4D/error_recognition

**Features Extraction**: https://github.com/CaptainCook4D/feature_extractors

Struttura:
scaricate le feature, io ho un file feature.zip al cui interno ci sono i file omnivore.zip e slowfast.zip, che contengono a loro volta tutte le rispettive feature. Se avete anche voi una struttura di questo tipo, vi basta eseguire lo script setup_features_colab.py e le sistemerà come segue:

```
data/
├── video/
│   ├── omnivore/
│   │   ├── {recording_id}_360p.mp4_1s_1s.npz  # Feature video 1s per Omnivore
│   │   └── ...
│   └── slowfast/
│       ├── {recording_id}_360p.mp4_1s_1s.npz  # Feature video 1s per SlowFast
│       └── ...
└── audio/
    ├── {recording_id}_audio.npz
    └── ...
```

altrimenti assicuratevi in qualche modo di sistemare le feature secondo questa struttura, altrimenti non partirà l'evaluation.
### NOTA
Assicuratevi che i percorsi nel file [config.py](./core/config.py) siano corretti, in particolare le voci *self.segment_features_directory* e *self.ckpt_directory* 