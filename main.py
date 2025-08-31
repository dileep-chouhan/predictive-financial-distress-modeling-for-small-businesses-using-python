import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
# --- 1. Synthetic Data Generation ---
np.random.seed(42) # for reproducibility
# Generate synthetic financial data for 100 small businesses
num_businesses = 100
data = {
    'CurrentRatio': np.random.uniform(0.5, 3, num_businesses),
    'DebtToEquityRatio': np.random.uniform(0.1, 2, num_businesses),
    'ProfitMargin': np.random.uniform(-0.1, 0.3, num_businesses),
    'SalesGrowth': np.random.uniform(-0.2, 0.5, num_businesses),
    'FinancialDistress': np.random.randint(0, 2, num_businesses) # 0: Not distressed, 1: Distressed
}
df = pd.DataFrame(data)
# --- 2. Data Cleaning and Preparation ---
# (In a real-world scenario, this would involve handling missing values, outliers, etc.)
# For this example, we'll assume the data is clean.
# Separate features (X) and target (y)
X = df.drop('FinancialDistress', axis=1)
y = df['FinancialDistress']
# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# --- 3. Model Training and Evaluation ---
# Train a Logistic Regression model (a simple model for demonstration)
model = LogisticRegression(solver='liblinear') # Choose a solver appropriate for smaller datasets
model.fit(X_train, y_train)
# Make predictions on the test set
y_pred = model.predict(X_test)
# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)
print(f"Model Accuracy: {accuracy}")
print("\nConfusion Matrix:\n", conf_matrix)
print("\nClassification Report:\n", class_report)
# --- 4. Visualization ---
# Visualize the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Not Distressed', 'Distressed'],
            yticklabels=['Not Distressed', 'Distressed'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.tight_layout()
plt.savefig('confusion_matrix.png')
print("Plot saved to confusion_matrix.png")
#Visualize feature importance (if applicable and if the model supports it)
#This part is commented out because LogisticRegression doesn't directly provide feature importance in a way that is easily visualized.  
#For tree-based models, this would be straightforward.
# feature_importances = model.feature_importances_
# plt.figure(figsize=(10,6))
# sns.barplot(x=feature_importances, y=X.columns)
# plt.title("Feature Importances")
# plt.xlabel("Importance")
# plt.ylabel("Feature")
# plt.tight_layout()
# plt.savefig('feature_importances.png')
# print("Plot saved to feature_importances.png")