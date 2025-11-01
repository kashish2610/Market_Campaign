# Market Campaign Prediction and Analysis

## Overview
This project develops a machine learning pipeline to predict customer responses to marketing campaigns. It combines exploratory data analysis (EDA), feature engineering, class imbalance handling, and ensemble learning models to improve campaign efficiency and customer targeting accuracy.

The notebook `Market_campign_final.ipynb` demonstrates a complete process—from raw data to model evaluation—using Python and key machine learning libraries.

---

## Objectives
- Understand customer purchasing behaviour and demographics.
- Identify key features influencing marketing campaign success.
- Build and evaluate robust classification models for campaign response prediction.
- Handle class imbalance and improve model generalization.

---

## Dataset
- **File used:** `superstore_data.csv`
- **Description:** Contains customer demographic and transactional data along with marketing response information.
- **Key features include:**
  - Demographics: `Age`, `Marital_Status`, `Education`, `Income`
  - Campaign and purchase data: `Recency`, `MntWines`, `MntFruits`, `MntGoldProds`
  - Household info: `Kidhome`, `Teenhome`
  - Target variable: `Response` (1 = positive, 0 = negative)
    
---
## Tech Stack  
- **Programming Language:** Python  
- **Libraries / Tools:**  
  - pandas, numpy — for data manipulation  
  - matplotlib, seaborn — for data visualisation  
  - scikit-learn — for model training and evaluation  
  - xgboost — gradient boosting frameworks  
  - optuna — for hyperparameter tuning  
  - Jupyter Notebook / Google Colab — for development and analysis
    
---    

## Exploratory Data Analysis (EDA)
The notebook explores:
- Data distributions using bar and box plots.
- Relationships between income, age, and campaign success.
- Outlier detection and missing value analysis.
- Correlation heatmaps to identify redundant features.

---

## Feature Engineering
Feature engineering was a crucial step to prepare the dataset for modelling:
1. **Feature Reduction and Transformation**
   - Dropped irrelevant or redundant columns: `Id`, `Dt_Customer`.
   - Derived `Age` from `Year_Birth` and removed the original column.
2. **Aggregation**
   - Created a new `Family_Size` feature by combining `Kidhome` and `Teenhome`.
3. **Encoding**
   - Applied label encoding / one-hot encoding on categorical variables such as `Education` and `Marital_Status`.
4. **Scaling**
   - Standardized numerical features using `StandardScaler` to ensure balanced feature contribution.
5. **Class Imbalance Handling**
   - Used **SMOTE (Synthetic Minority Over-sampling Technique)** to balance the dataset and prevent model bias towards the majority class.
6. **Train-Test Split**
   - Split the data into training and testing sets for unbiased model evaluation.

---

## Model Building and Evaluation
 The following models were developed and optimised:  
   - **AdaBoost Classifier** using `GridSearchCV` for hyperparameter tuning.  
   - **Gradient Boosting Classifier** using `RandomizedSearchCV`.  
   - **XGBoost Classifier**, with hyperparameters tuned via `Optuna`.  

### Implementation Details
- Used `GridSearchCV` for hyperparameter tuning.
- Evaluated performance using metrics:
  - **Accuracy**
  - **F1-score**
  - **AUC**
  - **Confusion Matrix**
  - **Classification Report**
- Libraries used:
  - `scikit-learn`
  - `xgboost`
  - `imblearn` for SMOTE handling

### Results
| Model                  | Accuracy | F1-Score |   AUC     |
|-------------------------|----------|----------|----------|
| Adaboost                | 0.87     | 0.92     | 0.84     |
| Gradient Boosting       | 0.88     | 0.93     | 0.88     |
|  XGBoost                | 0.8916   | 0.94     | 0.88     |


### Model Evaluation (XGBoost — Best Model)

The **XGBoost** was tuned and evaluated on the test dataset with an optimal decision threshold of **0.85**.  
The following summarizes the model’s performance metrics and classification outcomes.

#### Performance Metrics
| Metric | Score |
|---------|-------|
| **Threshold** | 0.85 |
| **Accuracy** | 0.8916 |
| **F1-Score** | 0.50 |
| **AUC-ROC** | 0.8815 |

---

#### Confusion Matrix
|               | **Predicted: 0** | **Predicted: 1** |
|---------------|------------------|------------------|
| **Actual: 0** | 371 *(True Negatives)* | 9 *(False Positives)* |
| **Actual: 1** | 39 *(False Negatives)* | 24 *(True Positives)* |

**Interpretation:**
- The model correctly classified **371 non-responders** and **24 responders**.  
- **9 customers** were incorrectly predicted as responders, while **39 actual responders** were missed.  
- Although overall accuracy is high (**~89%**), recall for the positive class is relatively lower due to class imbalance.  
- The threshold of 0.85 favors precision, prioritizing fewer false positives (over-targeting fewer uninterested customers).

---

#### Classification Report
| Class | Precision | Recall | F1-Score | Support |
|--------|-----------|---------|-----------|----------|
| **0 (No Response)** | 0.90 | 0.98 | 0.94 | 380 |
| **1 (Response)** | 0.73 | 0.38 | 0.50 | 63 |
| **Macro Avg** | 0.82 | 0.68 | 0.72 | 443 |
| **Weighted Avg** | 0.88 | 0.89 | 0.88 | 443 |

---



### Observations
- **XGBoost** achieved the best balance of speed and accuracy due to its leaf-wise tree growth strategy and regularization.  
- Feature scaling and SMOTE significantly improved recall on the minority (positive response) class.  
- Important features driving predictions included **Income**, **Recency**, **Age**, and **Family_Size**.
  
---

## Tools and Libraries
- **Python 3.9+**
- `numpy`, `pandas`, `matplotlib`, `seaborn`
- `scikit-learn`
- `xgboost`
- `imblearn`
- `datetime`

---

## How to Run
1. Clone the repository:
   git clone https://github.com/kashish2610/ML_Project_Market_campign.git
   cd ML_Project_Market_campign
 

