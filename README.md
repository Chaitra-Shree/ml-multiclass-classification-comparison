# 🍄 Mushroom Classification - ML Assignment 2

This project is a binary classification system designed to predict whether a mushroom is **Edible** or **Poisonous** based on its physical characteristics. It implements six different machine learning models and compares their performance using multiple evaluation metrics.

## Dataset Description
* **Source**: Kaggle/UCI Mushroom Classification Dataset.
* **Instances**: 8,124 mushrooms.
* **Features**: 22 categorical attributes (e.g., cap-shape, odor, habitat).
* **Classes**: Edible (e) and Poisonous (p).
* **Target Distribution**: Edible: 4,208 (51.8%), Poisonous: 3,916 (48.2%).

## Model Performance Comparison
The table below shows the performance of all 6 classification models against the required evaluation metrics.

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Logistic Regression** | 0.9551 | 0.9821 | 0.9598 | 0.9464 | 0.9531 | 0.9101 |
| **Decision Tree** | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| **kNN** | 0.9969 | 1.0000 | 0.9949 | 0.9987 | 0.9968 | 0.9938 |
| **Naive Bayes** | 0.9286 | 0.9506 | 0.9195 | 0.9336 | 0.9265 | 0.8572 |
| **Random Forest (Ensemble)** | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| **XGBoost (Ensemble)** | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |

## Observations on Model Performance
Based on the results above, here are the observations for each model's performance on this dataset:

| ML Model Name | Observation about model performance |
| :--- | :--- |
| **Logistic Regression** | Achieved high accuracy but was outperformed by tree-based models, likely due to some non-linear patterns in the mushroom features. |
| **Decision Tree** | Achieved perfect scores as it successfully identified key features like "odor" which have a very high correlation with mushroom toxicity. |
| **kNN** | Showed exceptional performance by effectively clustering similar mushroom types in the multidimensional feature space. |
| **Naive Bayes** | While still highly accurate, it was the lowest performer because it assumes feature independence, which is not strictly true for biological data. |
| **Random Forest (Ensemble)** | Successfully used multiple decision trees to eliminate classification errors, resulting in a perfect MCC and Accuracy score. |
| **XGBoost (Ensemble)** | Handled the categorical features perfectly through gradient boosting, delivering the most robust and accurate predictions. |

## Project Structure
* **app.py**: The Streamlit frontend for interactive testing and data upload.
* **requirements.txt**: List of Python libraries needed for the app.
* **README.md**: Full project documentation including performance tables.
* **model/**: Directory containing the saved `.pkl` model files and training logic.

## Setup and Execution
1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt

2. **Train and Evaluate**:
   ```bash
   python model/train_models.py

3. **Launch the Web App:**:
   ```bash
   streamlit run app.py