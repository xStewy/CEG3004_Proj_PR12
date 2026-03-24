# Environmental Sound Classification – CEG3004 DSP Mini-Project

## Overview
This project builds a machine learning system to classify environmental sounds using audio signal processing and supervised learning.

The system processes audio clips, extracts meaningful acoustic features, and trains a classifier to recognise sound categories such as animals, human activities, and environmental noises.

The model is trained using the **ESC-50 derived dataset** provided in the course project and predicts labels for a hidden submission set.

---

## Methodology

### 1. Audio Preprocessing
All audio clips are standardised before feature extraction.

Steps:
- Convert audio to **mono**
- Resample to **16 kHz**
- Trim silence
- Pad or truncate to **5 seconds**
- Normalise amplitude

---

### 2. Feature Extraction
Instead of raw audio, statistical audio features are extracted.

Features used:

**MFCC**
- 40 MFCC coefficients
- Delta (Δ)
- Delta-Delta (Δ²)

**Spectral Features**
- Spectral centroid
- Spectral bandwidth
- Spectral rolloff
- Spectral flatness

**Time-Frequency Features**
- Log-Mel spectrogram (64 Mel bands)

**Other Audio Descriptors**
- Chroma features
- Spectral contrast

---

### 3. Robust Feature Pooling
For each feature matrix, statistical pooling is applied across time:

- Mean
- Standard deviation
- Median
- 10th percentile
- 90th percentile

This converts variable-length spectrogram features into a fixed-length vector suitable for machine learning.

---

### 4. Machine Learning Model

The final classifier used is:

**ExtraTreesClassifier**

Advantages:
- Handles high-dimensional tabular features well
- Robust to noise
- Fast training
- Strong performance for audio feature vectors

Model configuration:

- `n_estimators = 3000`
- `max_features = 'sqrt'`
- `min_samples_split = 2`
- `min_samples_leaf = 1`
- `bootstrap = False`

---

## Repository Structure

