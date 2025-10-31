# Human Activity Recognition Using Hidden Markov Models

## Overview
This project implements a Hidden Markov Model (HMM) to classify human activities (standing, walking, jumping, still) using smartphone accelerometer and gyroscope data. It is designed for personalized health monitoring in elderly care facilities, providing a non-intrusive way to track mobility patterns and detect potential health risks.

## Features
- Data collection from iPhone X accelerometer and gyroscope
- Preprocessing with sliding windows and feature extraction (time-domain, frequency-domain, correlations)
- HMM implementation with Gaussian emissions and Baum-Welch training
- Activity classification using the Viterbi algorithm
- Evaluation with sensitivity, specificity, and F1-score
- Interpretability through transition and emission parameters

## Project Structure
```
├── data/             # Raw and preprocessed sensor data
├── src/              # Code for preprocessing, HMM training, and evaluation
├── figures/          # Plots: transition matrix, emission means, confusion matrix
├── README.md         # Project overview and instructions
└── report.pdf        # Final report with results and discussion
```

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/human-activity-hmm.git
   cd human-activity-hmm
   ```

2. Install required Python packages:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
1. Preprocess raw data:
   ```bash
   python src/preprocess.py
   ```

2. Train HMM and evaluate:
   ```bash
   python src/train_hmm.py
   ```

3. Visualize results:
   ```bash
   python src/visualize.py
   ```

