import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score, classification_report
from sklearn.model_selection import train_test_split


def load_and_preprocess_data(file_path, is_train_set=True):
    # Step 1: Parsing the CSV file
    data = pd.read_csv(file_path)

    # Step 2: Preprocessing

    # Converting spaces to NaN and then filling NaN values
    data.replace(' ', pd.NA, inplace=True)
    data.fillna(method='ffill', inplace=True)

    # Encoding categorical variables
    label_encoders = {}
    categorical_columns = [
        'gender', 'Partner', 'Dependents', 'PhoneService',
        'MultipleLines', 'InternetService', 'OnlineSecurity',
        'OnlineBackup', 'DeviceProtection', 'TechSupport',
        'StreamingTV', 'StreamingMovies', 'Contract',
        'PaperlessBilling', 'PaymentMethod'
    ]

    for column in categorical_columns:
        label_encoders[column] = LabelEncoder()
        data[column] = label_encoders[column].fit_transform(data[column])

    # Encoding 'Churn' column conditionally
    if is_train_set:
        label_encoders['Churn'] = LabelEncoder()
        data['Churn'] = label_encoders['Churn'].fit_transform(data['Churn'])

    # Converting columns to appropriate data types before scaling
    data['tenure'] = pd.to_numeric(data['tenure'], errors='coerce')
    data['MonthlyCharges'] = pd.to_numeric(data['MonthlyCharges'], errors='coerce')
    data['TotalCharges'] = pd.to_numeric(data['TotalCharges'], errors='coerce')

    # Filling any new NaN values created due to conversion
    data.fillna(method='ffill', inplace=True)

    # Scaling numerical variables
    scaler = StandardScaler()
    data[['tenure', 'MonthlyCharges', 'TotalCharges']] = scaler.fit_transform(
        data[['tenure', 'MonthlyCharges', 'TotalCharges']])

    return data, label_encoders


# Usage:
train_data, label_encoders = load_and_preprocess_data('data/churn_training.csv', is_train_set=True)
test_data, _ = load_and_preprocess_data('data/churn_holdout.csv', is_train_set=False)

# Splitting the training data into training and validation sets
X = train_data.drop('Churn', axis=1)
y = train_data['Churn']
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Train and evaluate Logistic Regression model
lr_model = LogisticRegression(max_iter=1000, random_state=42)
lr_model.fit(X_train, y_train)
lr_predictions = lr_model.predict(X_val)
lr_f1 = f1_score(y_val, lr_predictions)
print('Logistic Regression Classification Report:')
print(classification_report(y_val, lr_predictions))

# Train and evaluate Decision Tree model
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)
dt_predictions = dt_model.predict(X_val)
dt_f1 = f1_score(y_val, dt_predictions)
print('Decision Tree Classification Report:')
print(classification_report(y_val, dt_predictions))

# Train and evaluate KNN model
knn_model = KNeighborsClassifier()
knn_model.fit(X_train, y_train)
knn_predictions = knn_model.predict(X_val)
knn_f1 = f1_score(y_val, knn_predictions)
print('KNN Classification Report:')
print(classification_report(y_val, knn_predictions))

# Summarizing the performance of the models
print('Model Performance Summary:')
print(f'Logistic Regression F1 Score: {lr_f1:.2f}')
print(f'Decision Tree F1 Score: {dt_f1:.2f}')
print(f'KNN F1 Score: {knn_f1:.2f}')


# Identifying the best model based on F1 Score
best_model = max((lr_model, lr_f1), (dt_model, dt_f1), (knn_model, knn_f1), key=lambda pair: pair[1])[0]

# Preparing the test data
X_test = test_data
X_test = test_data.drop('Churn', axis=1)

# Using the best model to make predictions on the test set
test_predictions = best_model.predict(X_test)

# Decoding the predictions back to original labels (Yes/No)
churn_label_encoder = label_encoders['Churn']
test_predictions_decoded = churn_label_encoder.inverse_transform(test_predictions)

# Saving the predictions to a CSV file
submission_data = pd.DataFrame({'CustomerID': test_data['customerID'], 'Churn': test_predictions_decoded})
submission_data.to_csv('data/churn_predictions.csv', index=False)

print('Test predictions saved to data/churn_predictions.csv')

