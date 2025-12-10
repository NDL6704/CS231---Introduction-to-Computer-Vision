# Vietnamese Traditional Music Genre Classification

A machine learning and deep learning project for classifying Vietnamese traditional music genres.

## Project Overview

This project classifies 3 Vietnamese traditional music genres:
- **Cải Lương** (cailuong)
- **Ca Trù** (catru)
- **Chèo** (cheo)

The project implements two approaches:
1. **Machine Learning**: SVM and KNN with HOG (Histogram of Oriented Gradients) features
2. **Deep Learning**: CNN with Mel Spectrogram

## Directory Structure

```
CS231/
│
├── datasets/
│   └── image/
│       ├── cailuong/
│       ├── catru/
│       └── cheo/
│
├── SVM&KNN_final.ipynb
├── EDA and train models.ipynb
├── app-CS231.py
├── music_genre_classifier.h5
└── README.md
```

## Installation

### Python Libraries

```bash
pip install opencv-python numpy pandas scikit-image scikit-learn joblib matplotlib librosa pydub soundfile streamlit tensorflow
```

### Additional Requirements

- Python 3.8 or higher
- FFmpeg (for audio processing)

## Usage

### 1. Train SVM/KNN Models

Open and run `SVM&KNN_final.ipynb`:

**Main steps:**
- Load images from `datasets/image/`
- Resize images to 150x150
- Extract HOG features with parameters:
  - orientations=9
  - pixels_per_cell=(8, 8)
  - cells_per_block=(2, 2)
- Train and optimize hyperparameters using GridSearchCV
- Save trained models to `.pkl` files

### 2. Train Deep Learning Model

Open and run `EDA and train models.ipynb` to:
- Perform exploratory data analysis
- Train CNN model
- Save model to `music_genre_classifier.h5`

### 3. Run Prediction Application

Launch the Streamlit web application:

```bash
streamlit run app-CS231.py
```

**Application features:**
- Upload audio files (.wav or .mp3)
- Automatic MP3 to WAV conversion
- Split audio into 30-second segments
- Generate Mel Spectrograms
- Predict genre using voting mechanism
- Display prediction results

## Model Results

### SVM Model
- **Kernel**: RBF/Linear (optimized by GridSearchCV)
- **Features**: HOG features
- **Results**: See classification report in notebook

### KNN Model
- **Neighbors**: 3-9 (optimized by GridSearchCV)
- **Weights**: uniform/distance
- **Metric**: euclidean/manhattan
- **Results**: See classification report in notebook

### CNN Model
- **Input shape**: (369, 496, 3) - Mel Spectrogram
- **Architecture**: Custom CNN
- **Output**: 3 classes (cailuong, catru, cheo)

## Prediction Pipeline

1. **Input**: Audio file (.wav or .mp3)
2. **Preprocessing**:
   - Convert MP3 → WAV (if needed)
   - Split audio into 30-second segments (661500 samples @ 22050 Hz)
3. **Feature Extraction**:
   - Compute STFT (Short-Time Fourier Transform)
   - Generate Mel Spectrogram
   - Convert to dB scale
4. **Prediction**:
   - Predict each segment
   - Apply voting for final result
5. **Output**: Music genre (cailuong/catru/cheo)

## Key Parameters

### Mel Spectrogram
- **Sample rate**: 22050 Hz
- **Hop length**: 256
- **N_mels**: 576
- **Reference**: np.max

### Audio Segmentation
- **Unit length**: 661500 samples (~30 seconds)
- **Voting method**: Majority voting

### HOG Features
- **Orientations**: 9
- **Pixels per cell**: (8, 8)
- **Cells per block**: (2, 2)
- **Block norm**: L2-Hys



