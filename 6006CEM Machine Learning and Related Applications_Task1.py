import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA

# Download NLTK resources
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

# Load the dataset
dataset = pd.read_csv('Fraud.csv')

# Display the first few rows of the dataset
print("First few rows of the dataset:")
print(dataset.head())

# Display basic information about the dataset
print("\nDataset information:")
print(dataset.info())

# Display summary statistics of the dataset
print("\nSummary statistics of the dataset:")
print(dataset.describe())

# Display the number of missing values in each column
print("\nNumber of missing values in each column:")
print(dataset.isnull().sum())

# Display the number of unique values in each column:
print("\nNumber of unique values in each column:")
print(dataset.nunique())

# Data Cleaning
# Handling missing data
imputer = SimpleImputer(strategy='mean')
dataset['amount'] = imputer.fit_transform(dataset['amount'].values.reshape(-1, 1))

# Removing duplicate records
dataset = dataset.drop_duplicates()

# Data Transformation
# Normalization/Scaling
scaler = StandardScaler()
numerical_features = ['step', 'amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest']
dataset[numerical_features] = scaler.fit_transform(dataset[numerical_features])

# Encoding categorical variables
label_encoder = LabelEncoder()
dataset['type'] = label_encoder.fit_transform(dataset['type'])

# Dimensionality reduction using PCA
pca = PCA(n_components=5)  # Adjust the number of components as needed
dataset_pca = pca.fit_transform(dataset[numerical_features])
dataset_pca = pd.DataFrame(dataset_pca, columns=[f'pca_{i}' for i in range(1, 6)])
dataset = pd.concat([dataset, dataset_pca], axis=1)

# Save the preprocessed dataset to a new CSV file
dataset.to_csv('preprocessed_fraud_dataset.csv', index=False)

# Data Splitting
# Split the dataset into features (X) and target variable (y)
X = dataset.drop('isFraud', axis=1)
y = dataset['isFraud']

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Drop the specified column in place
x_train_numeric = x_train.select_dtypes(include=[np.number])
x_test_numeric = x_test.select_dtypes(include=[np.number])

# Create and train a Decision Tree classifier with hyperparameter tuning
param_grid_dt = {
    'max_depth': [5, 10, 15],
    'min_samples_split': [2, 5, 10]
}

grid_dt = GridSearchCV(DecisionTreeClassifier(random_state=42), param_grid_dt, cv=5)
grid_dt.fit(x_train_numeric, y_train)

# Print the best hyperparameters for Decision Tree
print("Best Hyperparameters for Decision Tree:", grid_dt.best_params_)

# Use the best model for predictions
y_pred_dt_tuned = grid_dt.predict(x_test_numeric)
accuracy_dt_tuned = accuracy_score(y_test, y_pred_dt_tuned)

# Use a different variant of Naive Bayes for continuous features
naive_bayes = GaussianNB()
naive_bayes.fit(x_train, y_train)

# Improve Neural Network architecture and training
model = Sequential()
model.add(Dense(256, input_dim=x_train.shape[1], activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])

# Train the model for more epochs
history = model.fit(x_train, y_train, epochs=20, batch_size=32, validation_data=(x_test, y_test), verbose=1)

# Make predictions on the test set for each model
y_pred_dt = grid_dt.predict(x_test)
y_pred_rf = grid_rf.predict(x_test)
y_pred_nb = naive_bayes.predict(x_test)
y_pred_nn = (model.predict(x_test) > 0.5).astype("int32")

# Evaluate the models after hyperparameter tuning
accuracy_dt = accuracy_score(y_test, y_pred_dt)
accuracy_rf = accuracy_score(y_test, y_pred_rf)
accuracy_nb = accuracy_score(y_test, y_pred_nb)
accuracy_nn = accuracy_score(y_test, y_pred_nn)

conf_matrix_dt = confusion_matrix(y_test, y_pred_dt)
conf_matrix_rf = confusion_matrix(y_test, y_pred_rf)
conf_matrix_nb = confusion_matrix(y_test, y_pred_nb)
conf_matrix_nn = confusion_matrix(y_test, y_pred_nn)

classification_rep_dt = classification_report(y_test, y_pred_dt)
classification_rep_rf = classification_report(y_test, y_pred_rf)
classification_rep_nb = classification_report(y_test, y_pred_nb, zero_division=1)
classification_rep_nn = classification_report(y_test, y_pred_nn, zero_division=1)

# Print the results for each model after hyperparameter tuning
print("\nDecision Tree Classifier (After Tuning):")
print(f"Accuracy: {accuracy_dt_tuned}")
print("Confusion Matrix:")
print(conf_matrix_dt)
print("Classification Report:")
print(classification_rep_dt)
print("\n\n")

# Print the results for each model after hyperparameter tuning
print("\nRandom Forest Classifier (After Tuning):")
print(f"Accuracy: {accuracy_rf_tuned}")
print("Confusion Matrix:")
print(conf_matrix_rf)
print("Classification Report:")
print(classification_rep_rf)
print("\n\n")

# Print the results for each model after hyperparameter tuning
print("\nNaive Bayes Classifier:")
print(f"Accuracy: {accuracy_nb}")
print("Confusion Matrix:")
print(conf_matrix_nb)
print("Classification Report:")
print(classification_rep_nb)
print("\n\n")

# Print the results for each model after hyperparameter tuning
print("\nNeural Network Classifier:")
print(f"Accuracy: {accuracy_nn}")
print("Confusion Matrix:")
print(conf_matrix_nn)
print("Classification Report:")
print(classification_rep_nn)
print("\n\n")

# Plot Confusion Matrix Heatmap for Decision Tree
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix_dt, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title('Decision Tree Confusion Matrix Heatmap')
plt.show()

# Bar Chart for Model Accuracy
models = ['Decision Tree', 'Random Forest', 'Naive Bayes', 'Neural Network']
accuracies = [accuracy_dt_tuned, accuracy_rf_tuned, accuracy_nb, accuracy_nn]

plt.figure(figsize=(8, 6))
plt.bar(models, accuracies, color=['blue', 'green', 'red', 'purple'])
plt.ylim(0, 1)
plt.ylabel('Accuracy')
plt.title('Model Accuracy Comparison (After Tuning)')
plt.show()

# Learning Curve for Neural Network
plt.figure(figsize=(8, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Neural Network Learning Curve (After Tuning)')
plt.legend()
plt.show()
