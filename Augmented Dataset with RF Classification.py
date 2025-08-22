import pandas as pd                                                                        # This Library will help with cleaning dataset
from sklearn.model_selection import train_test_split                                       # This is used for the splitting of the dataset into training and testing
from sklearn.ensemble import RandomForestClassifier                                        # This is the random forest algorithm used for the model
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix        # This is for the evaluation metrics used for viewing the results

ransom_dataset = pd.read_csv("augmented_dataset.csv")                                      # Used for reading the original dataset

features = ransom_dataset.drop("Class", axis=1)                                     # Splitting the dataset into its features and labels
labels = ransom_dataset["Class"]

features_train, features_test, labels_train, labels_test = train_test_split(                # Splitting the dataset into its testing and training sets with 80% for training and 20% for testing
    features, labels, test_size=0.2, random_state=35, stratify=labels
)

print("Train-test complete")
print(f"Training set: {features_train.shape}, Testing set: {features_test.shape}")

model = RandomForestClassifier(random_state=42)                                             # Model using the Random Forest algorithm
model.fit(features_train, labels_train)

test_pred = model.predict(features_test)                                                    # This will make prediction on the test set while the other set is being used for training

print("\n=== Evaluation ===")                                                               # The model will then be evaluated and show the results
print("Accuracy:", accuracy_score(labels_test, test_pred))                                  # Accuracy
print("\nClassification Report:\n", classification_report(labels_test, test_pred))          # Precision, recall, and F1-score
print("\nConfusion Matrix:\n", confusion_matrix(labels_test, test_pred))                    # Shows breakdown of the predictions

