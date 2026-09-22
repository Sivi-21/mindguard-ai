# Mental Health Classification Project

This project implements and compares three different machine learning approaches for mental health status classification based on text statements. The models evaluated are BERT (transformer-based), LSTM (recurrent neural network), and SVM (support vector machine) with TF-IDF features.

## Overview

The project analyzes text statements to classify mental health status using three different machine learning approaches:

- **BERT**: A pre-trained transformer model fine-tuned for sequence classification
- **LSTM**: A recurrent neural network with custom word embeddings
- **SVM**: A linear support vector machine with TF-IDF feature extraction

## Project Structure

```
├── README.md                    # Project documentation
├── data.csv                     # Input dataset (30MB)
├── bert.py                      # BERT model implementation
├── lstm.py                      # LSTM model implementation  
├── svm.py                       # SVM model implementation
├── evaluate.py                  # Model evaluation script
├── preds/                       # Model predictions directory
│   ├── val_predictions_bert.csv # BERT validation predictions
│   ├── val_predictions_lstm.csv # LSTM validation predictions
│   └── val_predictions_svm.csv  # SVM validation predictions
└── results/                     # Saved models directory
    ├── BERT/                    # BERT model artifacts
    ├── LSTM/                    # LSTM model artifacts (lstm_model.pt)
    └── SVM/                     # SVM model artifacts (model + vectorizer)
```

## Dataset

This project uses the **"Sentiment Analysis for Mental Health"** dataset from Kaggle, created by [suchintikasarkar](https://www.kaggle.com/datasets/suchintikasarkar/sentiment-analysis-for-mental-health).

### Dataset Overview
- **Size**: 51,074 instances (30MB CSV file)
- **Source**: Kaggle - A comprehensive, curated collection of mental health statements
- **URL**: https://www.kaggle.com/datasets/suchintikasarkar/sentiment-analysis-for-mental-health

### Dataset Structure
The CSV file (`data.csv`) contains the following columns:
- `statement`: Text statements/comments related to mental health discussions
- `status`: Mental health status labels (target variable for classification)

### Data Source Composition
This dataset is a meticulously curated amalgamation of multiple publicly available mental health datasets from Kaggle, including:
- 3k Conversations Dataset for Chatbot
- Depression Reddit Cleaned
- Human Stress Prediction
- Mental Health Dataset Bipolar
- Reddit Mental Health Data
- Students Anxiety and Depression Dataset
- Suicidal Mental Health Dataset
- Suicidal Tweet Detection Dataset

### Data Preprocessing
The dataset is automatically split into training (70%) and validation (30%) sets with a fixed random seed (42) for reproducibility across all three models.

**Key preprocessing steps:**
- **BERT**: Uses BERT tokenizer with truncation, padding, and max length of 128 tokens
- **LSTM**: Basic tokenization with regex `\b\w+\b`, lowercasing, vocabulary building from top 10,000 words
- **SVM**: Text cleaning (lowercasing, removing special characters), TF-IDF vectorization with max 10,000 features
- **Label encoding**: Consistent LabelEncoder across all models for target classes
- **Data splitting**: 70/30 train/validation split with random_state=42 for reproducibility

## Models

### 1. BERT (`bert.py`)
- Uses pre-trained `bert-base-uncased` model
- Fine-tuned for 2 epochs with batch size 16
- Implements custom PyTorch dataset class
- Saves model to `results/BERT/`
- Saves predictions to `preds/val_predictions_bert.csv`

### 2. LSTM (`lstm.py`)
- Custom LSTM implementation with PyTorch
- Builds vocabulary from top 10,000 most common words
- 128-dimensional embeddings and hidden states
- Trains for 20 epochs with validation after each epoch
- Adam optimizer with learning rate 1e-3
- Saves model to `results/LSTM/lstm_model.pt`

### 3. SVM (`svm.py`)
- Uses LinearSVC from scikit-learn
- TF-IDF vectorization with max 10,000 features
- Includes uni-gram and bi-gram features
- Text preprocessing with regex cleaning
- Saves model and vectorizer to `results/SVM/`

## Setup Instructions

### 1. Install Dependencies

```bash
pip install pandas scikit-learn transformers torch
```

### Key Dependencies:
- `pandas`: Data manipulation and analysis
- `scikit-learn`: SVM model and evaluation metrics
- `transformers`: BERT model and tokenizer
- `torch`: PyTorch for LSTM implementation

### 2. Download Dataset

1. Visit the [Kaggle dataset page](https://www.kaggle.com/datasets/suchintikasarkar/sentiment-analysis-for-mental-health)
2. Download the dataset (requires Kaggle account)
3. Extract the CSV file and place it in your project directory as `data.csv`
4. This is optional as the data used for training the models are present in the repository itself.

** Download using Kaggle API:**
```bash
# Install Kaggle CLI
pip install kaggle

# Download dataset (requires kaggle.json in ~/.kaggle/)
kaggle datasets download -d suchintikasarkar/sentiment-analysis-for-mental-health

# Extract the dataset
unzip sentiment-analysis-for-mental-health.zip
```

## Usage

### Training Individual Models

1. **Train BERT model:**
   ```bash
   python bert.py
   ```
   - Training time: 2 epochs (depends on hardware)
   - GPU recommended for faster training

2. **Train LSTM model:**
   ```bash
   python lstm.py
   ```
   - Training time: 20 epochs
   - Validates after each epoch
   - Custom vocabulary building included

3. **Train SVM model:**
   ```bash
   python svm.py
   ```
   - Fastest training time
   - Uses TF-IDF feature extraction
   - Good baseline for comparison

### Model Evaluation

Run the evaluation script to get detailed metrics:
```bash
python evaluate.py
```

This script calculates:
- Per-class metrics (Precision, Recall, F1-score, Sensitivity, Specificity)
- Overall metrics (Accuracy, Cohen's Kappa, Macro/Micro/Weighted F1-scores)
- Confusion matrix analysis

The evaluation script automatically reads all prediction files from the `preds/` directory.

## Output Files

### Prediction Files (`preds/` directory)
Each model generates validation predictions in the `preds/` folder:
- `preds/val_predictions_bert.csv`: BERT model predictions
- `preds/val_predictions_lstm.csv`: LSTM model predictions  
- `preds/val_predictions_svm.csv`: SVM model predictions

Each CSV contains:
- `true_label`: Actual encoded labels (numerical, see below)  
- `predicted_label`: Model predictions (numerical, see below)

### Mental Health Numerical Categories

The dataset includes 7 mental health categories. All models use consistent label encoding:

| **Label Index** | **Mental Health Category** | **Description** |
|-----------------|----------------------------|-----------------|
| **0** | **Anxiety** | Anxiety disorders and related symptoms |
| **1** | **Bipolar** | Bipolar disorder (manic-depressive episodes) |
| **2** | **Depression** | Depression and depressive episodes |
| **3** | **Normal** | Normal/healthy mental state |
| **4** | **Personality disorder** | Avoidant personality disorder (AvPD) |
| **5** | **Stress** | Stress-related mental health conditions |
| **6** | **Suicidal** | Suicidal ideation and related thoughts |

**Note**: The numerical labels are consistent across all models (BERT, LSTM, SVM) due to alphabetical ordering by sklearn's LabelEncoder

### Saved Models (`results/` directory)
Trained models are saved in organized subdirectories:
- `results/BERT/`: BERT model files (Hugging Face format)
- `results/LSTM/lstm_model.pt`: PyTorch LSTM state dictionary
- `results/SVM/svm_model.joblib`: Scikit-learn SVM model
- `results/SVM/tfidf_vectorizer.joblib`: TF-IDF vectorizer for SVM

## Model Performance

### Performance Results Summary

| Model | Accuracy | F1 Macro | F1 Weighted | Cohen's Kappa |
|-------|----------|----------|-------------|---------------|
| **BERT** | **83.6%** | **0.8106** | **0.8367** | **0.7871** |
| **SVM** | 77.07% | 0.7228 | 0.7673 | 0.6992 |
| **LSTM** | 74.89% | 0.663 | 0.7477 | 0.6727 |
| **XGBoost*** | 70.7% | 0.7 | 0.707 | 0.6113 |
| **Decision Tree*** | 62.54% | 0.619 | 0.625 | 0.5096 |

*_XGBoost and Decision Tree was trained with KNIME workflows (under `knime_workflows` folder) using SMOTE for class balancing_*

### Class-wise Recall Performance

| Mental Health Category | BERT | SVM | LSTM | XGBoost* | Decision Tree* |
|------------------------|------|-----|------|----------|----------------|
| **Depression** | 0.762 | 0.711 | 0.694 | 0.669 | 0.566 |
| **Anxiety** | **0.859** | 0.784 | 0.751 | 0.630 | 0.510 |
| **Bipolar Disorder** | 0.794 | 0.723 | 0.686 | 0.709 | 0.519 |
| **Normal** | **0.962** | 0.955 | 0.917 | 0.916 | 0.874 |
| **Suicidal Thoughts** | 0.731 | 0.658 | 0.691 | 0.615 | 0.485 |
| **Stress** | 0.747 | 0.501 | 0.464 | 0.356 | 0.397 |
| **Personality Disorder** | 0.684 | 0.658 | 0.400 | 0.548 | 0.451 |

### Key Findings

- **BERT** achieved the highest overall performance with 83.6% accuracy and demonstrated superior contextual sensitivity to informal linguistic artifacts
- **SVM** provided competitive performance (77.07% accuracy) with excellent interpretability and balanced class-wise performance  
- **LSTM** offered a computationally efficient alternative with 74.89% accuracy, capturing temporal dependencies effectively
- All models showed excellent performance on the **Normal** class (87.4%-96.2% recall)
- **Anxiety** detection achieved the highest recall across mental health conditions (51.0%-85.9%)
- **Stress** and **Personality Disorder** proved most challenging to classify accurately across all models
- BERT demonstrated particularly strong performance for **Suicidal Thoughts** (0.731 recall) and **Anxiety** (0.859 recall)

## Technical Details

### BERT Configuration
- Model: `bert-base-uncased`
- Max sequence length: 128 tokens
- Training epochs: 2
- Batch size: 16
- Weight decay: 0.01

## Model Versions

### Version 2 (BERT) - Current
- **Model:** bert-base-uncased (fine-tuned)
- **Expected Accuracy:** 85%+
- **Handles:** Context, negations, spelling variations
- **Training Date:** September 2026

### Version 1 (LSTM) - Deprecated
- **Model:** LSTM + BiLSTM
- **Accuracy:** 79.30%
- **Limitations:** Required keyword patches

### LSTM Configuration  
- Vocabulary size: Top 10,000 words + special tokens
- Embedding dimension: 128
- Hidden dimension: 128
- Training epochs: 20
- Learning rate: 1e-3

### SVM Configuration
- Algorithm: Linear SVM
- Features: TF-IDF (max 10,000 features)
- N-grams: Unigrams and bigrams
- Text preprocessing: Lowercase, remove special characters

## Research Applications

This project demonstrates comparative analysis of machine learning approaches for mental health classification, with potential applications in:

- **Clinical Decision Support**: Assisting healthcare professionals in preliminary mental health screening
- **Social Media Monitoring**: Automated detection of mental health concerns in online platforms
- **Research and Analytics**: Large-scale analysis of mental health discourse patterns
- **Educational Tools**: Understanding public sentiment around mental health topics

## Limitations and Ethical Considerations

### Important Limitations
- **Not for Clinical Diagnosis**: This tool is for research purposes only and should not replace professional mental health evaluation
- **Algorithmic Bias**: Models may reflect biases present in training data and could lead to stigmatization
- **Class Imbalance**: Some mental health conditions remain underrepresented, affecting model performance
- **Text-based Only**: Analysis limited to linguistic patterns; does not consider clinical context or history
- **Cultural Sensitivity**: Results may vary across different cultural and linguistic contexts

### Ethical Guidelines
As highlighted in the project research, deploying such models raises significant ethical questions:
- **Consent and Transparency**: Users should be informed before mental health classification
- **Data Privacy**: Strict privacy safeguards required for mental health-related text analysis  
- **Professional Oversight**: Models should only be deployed under supervision of public health professionals
- **Bias Auditing**: Regular evaluation for fairness and potential discrimination
- **Responsible Use**: Technology should support, not replace, human mental health professionals

## Notes

- All models use the same train/validation split (random_state=42) for fair comparison
- The project includes GPU support for BERT and LSTM models when available
- Models are saved for future inference and deployment
- Comprehensive logging and evaluation metrics are provided for model comparison
- For production use, consider additional validation with domain experts
- Regular model retraining recommended as language patterns evolve

## Data Citation

**Dataset**: Sentiment Analysis for Mental Health  
**Author**: suchintikasarkar  
**Source**: Kaggle  
**URL**: https://www.kaggle.com/datasets/suchintikasarkar/sentiment-analysis-for-mental-health  
**License**: Please refer to the Kaggle dataset page for licensing information

## Acknowledgments

This project utilizes the comprehensive mental health dataset created by suchintikasarkar on Kaggle. The dataset represents a valuable amalgamation of multiple mental health-related datasets, providing a rich foundation for mental health classification research.

Special thanks to:
- The original dataset creators whose work was compiled into this comprehensive collection
- The Kaggle community for making mental health datasets publicly available for research
- Contributors to the various source datasets including Reddit mental health data, depression datasets, and suicide detection datasets

🧠 Multi-model machine learning approach for mental health classification from social media text. Compares BERT, LSTM, and SVM models achieving 83.6% accuracy on 7 mental health categories. Research-focused with ethical AI considerations.
## 🎨 Interactive Dashboard

This project includes a Streamlit dashboard for live mental health risk prediction.

### Dashboard Features
- **Live Text Analysis**: Enter social media text and get instant risk prediction
- **Risk Levels**: Low, Moderate, High Risk classification
- **Confidence Scores**: Probability distribution for each prediction
- **Model Performance**: Visual metrics and confusion matrix
- **Data Analysis**: Dataset statistics and distributions


# 🧠 MindGuard AI - Mental Health Risk Detector

## 🎯 Overview
An AI-powered mental health risk detection system using BERT + BiLSTM hybrid model to classify social media text into 3 risk levels.

## 📊 Performance
- **Accuracy:** 88.95%
- **F1-Score:** 89.55%
- **Precision:** 85.66%
- **Recall:** 93.81%

## 🤖 Model Architecture
- **Base:** BERT-base-uncased (110M parameters)
- **Sequential:** 2-layer BiLSTM (256 units × 2 directions)
- **Safety Layer:** 35+ critical keywords + pattern matching
- **Output:** 3 risk classes (Low, Moderate, High)

## 🎨 Features
- Real-time text analysis
- 3-level risk classification
- Crisis resource integration
- Batch CSV analysis
- Sentiment analysis
- Emotion radar chart
- Keyword highlighting
- Download reports
- Dark mode
- Text-to-speech

## 🚀 Quick Start

### 1. Install
```bash
pip install -r requirements.txt


# MindGuard AI - Mental Health Risk Detector

## 🤖 Model: BERT-base-uncased (Fine-tuned)
- **F1-Score:** 0.8481
- **Precision:** 0.8441
- **Recall:** 0.8314
- **Accuracy:** ~85%
- **Training:** 52,681 samples on Kaggle GPU

## 🎯 Features
- Real-time text analysis
- 3-level risk classification
- Crisis resources integration
- Batch CSV analysis
- Premium dashboard UI

## 🛠️ Tech Stack
- Python, Streamlit, PyTorch
- HuggingFace Transformers
- BERT-base-uncased
- Scikit-learn



### Run the Dashboard
```bash
streamlit run app.py