# Predicting the BPM of Songs

A machine learning project that predicts the **Beats Per Minute (BPM)** of songs using numerical audio-related features.

The main goal of this project was to build a complete regression pipeline, from data exploration and preprocessing to model training, evaluation, and generating predictions for unseen data.

## Project Overview

In this project, the target variable is:

- `BeatsPerMinute` — the tempo of a song in BPM.

The model uses several numerical audio features to make predictions.

### Input Features

- `RhythmScore`
- `AudioLoudness`
- `VocalContent`
- `AcousticQuality`
- `InstrumentalScore`
- `LivePerformanceLikelihood`
- `MoodScore`
- `TrackDurationMs`
- `Energy`

The `id` column is used only as an identifier and is not used as a model feature.

## Dataset

The dataset contains:

- **524,164** training samples
- **174,722** test samples
- **9** input features
- **1** target variable
- No missing values

The training dataset contains the `BeatsPerMinute` target, while the test dataset does not.

## Project Structure

```text
Predicting the BPM of Songs/
├── data/
│   ├── sample_submission.csv
│   ├── train.csv
│   ├── test.csv
│   └── submission.csv
│
├── notebooks/
│   └── exploration.ipynb
│
├── src/
│   ├── data_loader.py
│   ├── evaluate.py
│   ├── model.py
│   └── preprocessing.py
│
├── config.py
├── main.py
└── README.md
```

## Approach

The project follows a simple machine learning workflow:

1. Load the training and test datasets.
2. Explore the dataset and feature distributions.
3. Separate the target variable from the input features.
4. Remove the `id` column from the model inputs.
5. Split the training data into training and validation sets.
6. Train regression models.
7. Evaluate the models using MAE, RMSE, and R².
8. Train the final model on the complete training dataset.
9. Generate predictions for the test dataset.
10. Save the predictions to `data/submission.csv`.

## Models

Two regression approaches were considered during the project.

### Linear Regression

Used as a simple baseline model.

### HistGradientBoostingRegressor

Used as the final model for generating the test predictions.

Configuration:

```python
HistGradientBoostingRegressor(
    max_iter=200,
    learning_rate=0.1,
    max_leaf_nodes=31,
    random_state=42
)
```

## Results

The models were evaluated on a 20% validation split.

| Model | MAE | RMSE | R² |
|---|---:|---:|---:|
| Dummy Mean Baseline | 21.2274 | 26.5196 | -0.0000 |
| Linear Regression | 21.2264 | 26.5178 | 0.0001 |
| HistGradientBoostingRegressor | 21.1832 | 26.4406 | 0.0003 |

The results show that the available features contain **very limited predictive information about BPM**.

The correlations between the target and the available features were also close to zero. As a result, the trained models tend to predict values close to the overall mean BPM.

This is an important result of the project: a machine learning model does not necessarily produce strong predictions when the available features do not contain enough information about the target.

## Test Predictions

After evaluation, the final `HistGradientBoostingRegressor` model was trained using the complete training dataset.

The resulting predictions were saved as:

```text
data/submission.csv
```

The submission contains:

```text
id,BeatsPerMinute
```

with one prediction for each test sample.

## Running the Project

Install the required Python packages:

```bash
pip install pandas numpy scikit-learn
```

Then run:

```bash
python main.py
```

The script will:

- load the data,
- train the model,
- evaluate it,
- train the final model,
- generate test predictions,
- create `data/submission.csv`.

## What I Learned

This project was mainly focused on building a complete and organized machine learning workflow.

Key topics practiced:

- Exploratory Data Analysis
- Pandas
- Train/validation splitting
- Regression
- Baseline models
- Linear Regression
- Gradient Boosting
- Model evaluation
- MAE, RMSE and R²
- Separating project components into modules
- Generating predictions
- Creating a submission file
- Git and GitHub project management

## Future Improvements

Possible improvements for a future version include:

- Feature engineering
- Testing additional regression algorithms
- Hyperparameter tuning
- Cross-validation
- Investigating whether additional audio features contain stronger BPM-related information
- Comparing predictions against a stronger baseline

## Status

**Completed**

This project represents a complete end-to-end machine learning pipeline, from raw training data to a generated test submission.