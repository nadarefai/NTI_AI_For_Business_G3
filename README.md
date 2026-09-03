# NTI_AI_For_Business_G3
# 🧠 Mental Health Score Prediction

A machine learning project investigating the relationship between **social media usage, lifestyle habits, academic characteristics, and mental health** among students.

The project uses several regression models to predict a student's **Mental Health Score** and compares their performance using standard regression metrics.

---

## 📌 Project Overview

Mental health can be influenced by multiple aspects of daily life, including social media usage, sleep, physical activity, academic workload, and stress.

The goal of this project is to use machine learning to identify patterns in these factors and predict the `Mental_Health_Score` of students.

The dataset contains:

* **5,000 observations**
* **13 variables**
* A continuous target variable: `Mental_Health_Score`

---

## 📊 Dataset Features

| Feature                   | Description                                | Type        |
| ------------------------- | ------------------------------------------ | ----------- |
| `Age`                     | Student's age                              | Numerical   |
| `Gender`                  | Student's gender                           | Categorical |
| `Country`                 | Student's country                          | Categorical |
| `Academic_Level`          | High School, Undergraduate, or Graduate    | Categorical |
| `Most_Used_Platform`      | Most frequently used social media platform | Categorical |
| `Purpose_Of_Use`          | Main purpose of social media use           | Categorical |
| `Avg_Daily_Usage_Hours`   | Average daily social media usage           | Numerical   |
| `Daily_Unlocks`           | Number of daily device/app unlocks         | Numerical   |
| `Study_Hours`             | Daily study hours                          | Numerical   |
| `Physical_Activity_Hours` | Daily physical activity hours              | Numerical   |
| `Sleep_Hours_Per_Night`   | Average nightly sleep                      | Numerical   |
| `Stress_Level`            | Low, Medium, High, or Very High            | Ordinal     |
| `Mental_Health_Score`     | Target mental health score                 | Numerical   |

---

## 🔎 Exploratory Data Analysis

The dataset was explored to understand:

* Distribution of the target variable
* Numerical feature distributions
* Categorical feature distributions
* Correlations between numerical variables
* Relationships between lifestyle factors and mental health
* Differences in mental health scores across stress levels
* Relationships between social media usage and mental health

Visualizations include:

* Histograms
* Bar charts
* Scatter plots
* Correlation matrix
* Actual vs. predicted plots
* Residual plots
* Feature importance plots

---

## 🧹 Data Preprocessing

### Country

The `Country` variable contained many categories with very few observations.

To reduce sparse categories, countries with fewer than **70 observations** were grouped into:

```text
Other
```

This reduces the number of extremely small categories while preserving the information from countries with sufficient observations.

### Categorical Encoding

Categorical variables were encoded according to their characteristics.

**Ordinal encoding** was used for variables with a natural order:

* `Academic_Level`
* `Stress_Level`

For example:

```text
Stress_Level:

Low        → 0
Medium     → 1
High       → 2
Very High  → 3
```

**One-hot encoding** was used for nominal variables such as:

* `Country`
* `Most_Used_Platform`
* `Purpose_Of_Use`

### Numerical Features

Numerical variables were passed directly to the tree-based models because models such as Random Forest, Extra Trees, and XGBoost do not require standardization.

---

## 🤖 Machine Learning Models

The project evaluates multiple regression algorithms.

### 1. Linear Regression

Used as a baseline model to establish a simple linear relationship between the input variables and mental health score.

### 2. Random Forest Regressor

An ensemble of decision trees that can capture nonlinear relationships and interactions between variables.

### 3. Extra Trees Regressor

Another tree-based ensemble method that introduces additional randomness when constructing decision trees.

### 4. XGBoost Regressor

A gradient-boosting algorithm that builds trees sequentially to correct errors made by previous trees.

---

## ⚙️ Machine Learning Pipeline

Preprocessing and model training are combined using **scikit-learn pipelines**.

The general structure is:

```text
Raw Dataset
     │
     ▼
Train / Test Split
     │
     ▼
ColumnTransformer
     │
     ├── Ordinal Encoding
     │
     ├── One-Hot Encoding
     │
     └── Numerical Features
     │
     ▼
Machine Learning Model
     │
     ▼
Predictions
     │
     ▼
Model Evaluation
```

Using a pipeline ensures that preprocessing is consistently applied to training and testing data and helps prevent data leakage during cross-validation.

---

## 🔧 Hyperparameter Tuning

Hyperparameter optimization was performed for the tree-based models using:

* `GridSearchCV`
* `RandomizedSearchCV`

For XGBoost, parameters such as the following were explored:

```text
n_estimators
learning_rate
max_depth
min_child_weight
subsample
colsample_bytree
gamma
reg_alpha
reg_lambda
```

Cross-validation was used during tuning to identify parameters that generalize well to unseen data.

---

## 📈 Model Evaluation

Because `Mental_Health_Score` is continuous, this is treated as a **regression problem**.

The following metrics are used:

### MAE — Mean Absolute Error

Measures the average absolute difference between the predicted and actual values.

Lower is better.

### MSE — Mean Squared Error

Penalizes larger prediction errors more strongly.

Lower is better.

### RMSE — Root Mean Squared Error

The square root of MSE and expressed in the same units as the target variable.

Lower is better.

### R² — R-squared

Measures the proportion of variation in the target that is explained by the model.

Higher is better.

---

## 📊 Model Comparison

The models are compared using the same test dataset and evaluation metrics.

---

## 🔍 Model Interpretability

Tree-based models provide feature importance values that can be used to identify which variables contribute most to predictions.

Feature importance is visualized to investigate the role of factors such as:

* Stress level
* Social media usage
* Sleep
* Study hours
* Physical activity
* Daily unlocks
* Academic level
* Platform usage

These results help provide insight into the variables that are most useful for predicting mental health scores.

---

## ⚠️ Limitations

Several limitations should be considered:

* A high predictive score does not mean that a particular variable directly causes changes in mental health.
* Country categories with low observation counts were grouped into `Other`, which reduces geographical detail.
* Machine learning predictions depend on the quality and representativeness of the available dataset.
* The model should not be used as a clinical diagnostic tool.

---

## 🛠️ Technologies Used

* **Python**
* **Pandas** — data manipulation
* **NumPy** — numerical computation
* **Matplotlib** — visualization
* **Scikit-learn** — preprocessing, pipelines, model evaluation, and hyperparameter tuning
* **XGBoost** — gradient boosting regression

---

## 📁 Project Structure

```text
Mental-Health-Prediction/
│
├── data/
│   └── dataset.csv
│
├── notebooks/
│   └── mental_health_prediction.ipynb
│
├── models/
│   └── best_model.pkl
│
├── README.md
│
└── requirements.txt
```

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone <repository-url>
cd Mental-Health-Prediction
```

### 2. Install dependencies

```bash
pip install pandas numpy matplotlib scikit-learn xgboost
```

Or, if a `requirements.txt` file is provided:

```bash
pip install -r requirements.txt
```

### 3. Run the notebook

Open:

```text
notebooks/mental_health_prediction.ipynb
```

and execute the cells to reproduce the analysis and model training.

---

## 🎯 Project Goal

The main objective of this project is to **compare multiple machine learning regression techniques** and determine which approach provides the most effective prediction of mental health scores based on social media usage, lifestyle, academic, and demographic factors.

The project also demonstrates a complete machine learning workflow:

```text
Data
 ↓
Exploration
 ↓
Cleaning
 ↓
Preprocessing
 ↓
Feature Encoding
 ↓
Train/Test Split
 ↓
Model Training
 ↓
Hyperparameter Tuning
 ↓
Evaluation
 ↓
Model Comparison
```

---

## 👥 Contributors

* Yassin Sobhy
* Nada Refai
* Kirollos Kamil
* Hady Kamal
* Menna Zaki


