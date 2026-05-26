# credit-card-fraud-detection
A machine learning project to detect fraudulent financial transactions within highly imbalanced data.

## Tech Stack & Algorithms
* **Language:** Python
* **Libraries:** Scikit-Learn, XGBoost, PyTorch (TabNet), Pandas
* **Models Evaluated:** Random Forest, Logistic Regression, XGBoost, TabNet

## Key Results
Because fraud datasets are highly imbalanced, models were evaluated strictly using the **F1-Score for the minority class** rather than overall accuracy:
* **Random Forest:** **79.52% F1-Score (Class 1)**
* **Logistic Regression:** **38.67% F1-Score (Class 1)**
* **XGBoost:** **73.87% F1-Score (Class 1)**
* **TabNet:** **56.04% F1-Score (Class 1)**

After comparing multiple models, Random Forest was selected for hyperparameter tuning due to its superior baseline performance.

## Final Model Tuning
While the baseline Random Forest yielded a high F1-score, hyperparameter tuning was performed to optimize the decision threshold, stabilize variance, and minimize false positives. 

The final tuned Random Forest model achieved a highly robust performance profile:
* **F1-Score (Class 1):** **78.23%**
* **Precision (Class 1):** **83.88%**
* **Recall (Class 1):** **73.29%**
* **ROC-AUC:** **0.9875**
* **Overall Accuracy:** **99.84%**

## Data Imbalance Strategies
This project tested two different methodologies to address the problem of severe class imbalance in transaction data:

1. **Cost-Sensitive Learning (`class_weight='balanced'`)**
   * Keeps the original data 100% intact and imposes a larger weight penalty on the model if it misses fraudulent transactions.
   * **Results:** Produced the best performance with a **78.23% F1-Score**.

2. **Data Resampling (Undersampling)**
   * Randomly cuts the amount of normal transaction data to a 50:50 balance with the amount of fraudulent data.
   * **Results:** The training process is significantly faster, but the final performance is still below the class-weight approach due to the loss of information from the discarded normal transaction data.

## Dataset Link
The models were trained using the raw transaction data sourced from the [Kaggle Credit Card Transactions Dataset](https://www.kaggle.com/datasets/priyamchoksi/credit-card-transactions-dataset).
