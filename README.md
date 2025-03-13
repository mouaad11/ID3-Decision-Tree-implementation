# Bank Marketing Analysis with ID3 Decision Tree

## Project Overview
This project implements an ID3 decision tree algorithm to predict whether a client will subscribe to a term deposit based on bank marketing campaign data. The model uses the entropy criterion for decision tree induction and includes data preprocessing, exploratory data analysis, and visualization of results.

## Dataset
The dataset contains information about direct marketing campaigns (phone calls) of a Portuguese banking institution. The classification goal is to predict if the client will subscribe to a term deposit (variable 'deposit').

### Key Features:
- **Client demographics**: age, job, marital status, education, etc.
- **Campaign information**: contact communication type, month, day of week
- **Previous campaign data**: pdays (days since last contact), poutcome (previous campaign outcome)
- **Economic indicators**: employment variation rate, consumer price index, etc.

## Implementation Details

### Data Preprocessing
- One-hot encoding for categorical features
- Feature scaling for numerical variables
- Train-test split (80% training, 20% testing)

### Model Building
The model uses the ID3 (Iterative Dichotomiser 3) algorithm, which is implemented using scikit-learn's DecisionTreeClassifier with the entropy criterion. Key parameters:
- `criterion='entropy'` - Uses information gain for feature selection
- `max_depth=5` - Controls tree complexity to prevent overfitting, plus additional parameters.

### Evaluation Metrics
- Accuracy: 81.1%
- Confusion Matrix Analysis
- Classification Report (Precision, Recall, F1-score)

## Key Findings

### Feature Importance
The top features influencing client subscription decisions are:
1. **Duration** (0.63) - Length of the last contact call
2. **Previous outcome success** (0.18) - Whether previous marketing was successful
3. **Unknown contact type** (0.18) - Contact communication method

### Decision Rules
The decision tree reveals important patterns in customer behavior:
- Clients with longer conversation durations are more likely to subscribe
- Previous campaign success is a strong indicator of future subscription
- Contact method significantly impacts campaign effectiveness

## Visualization
The project includes detailed visualizations:
- Feature distributions and relationships
- Target variable distribution
- Decision tree structure showing classification rules
- Feature importance chart

## How to Use
1. Clone the repository
2. Install required packages: `pip install -r requirements.txt`
3. Run the notebook: `jupyter notebook tp_id3.ipynb`

## Requirements
- Python 3.6+
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn

## Future Improvements
- Hyperparameter tuning to optimize model performance
- Feature engineering to create more predictive variables
- Ensemble methods to improve prediction accuracy
- Comparison with other algorithms (Random Forest, Gradient Boosting)

## References
- Bank Marketing Dataset: [UCI Machine Learning Repository]([https://archive.ics.uci.edu/ml/datasets/bank+marketing](https://www.kaggle.com/datasets/janiobachmann/bank-marketing-dataset))
- Decision Trees and ID3 Algorithm: [Machine Learning Mastery](https://machinelearningmastery.com/implement-decision-tree-algorithm-scratch-python/)
