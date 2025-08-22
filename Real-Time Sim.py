import pandas as pd                                                                          # This Library will help with cleaning dataset
from sklearn.model_selection import train_test_split                                         # This is used for the splitting of the dataset into training and testing
from sklearn.ensemble import RandomForestClassifier                                          # This is the random forest algorithm used for the model
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix          # This is for the evaluation metrics used for viewing the results

ransom_dataset = pd.read_csv("augmented_dataset.csv")                                        # Used for reading the dataset

features = ransom_dataset.drop("Class", axis=1)                                        # Splitting the dataset into its features and labels
labels = ransom_dataset["Class"]

features_train, features_test, labels_train, labels_test = train_test_split(                # Splitting the dataset into its testing and training sets with 80% for training and 20% for testing
    features, labels, test_size=0.2, random_state=35, stratify=labels
)

print(f"Training set: {features_train.shape}, Testing set: {features_test.shape}")

model = RandomForestClassifier(random_state=42)                                             # Model using the Random Forest algorithm
model.fit(features_train, labels_train)

test_pred = model.predict(features_test)                                                      # This will make prediction on the test set while the other set is being used for training

print("Evaluation")                                                                         # The model will then be evaluated and show the results
print("Accuracy:", accuracy_score(labels_test, test_pred))                                  # Accuracy
print("Classification Report:", classification_report(labels_test, test_pred))              # Precision, recall, and F1-score
print("Confusion Matrix:", confusion_matrix(labels_test, test_pred))                        # Shows breakdown of the predictions

print("Real-Time Detection Testing/Results")

sample_from_dataset = [10679, 14594, 7375, 3230, 16589]                                 # This selects the specific samples to test in the dataset. Change samples to test different times
sample_data = features_test.iloc[sample_from_dataset]                                   # Extracts all the features data for the samples
sample_labels = labels_test.iloc[sample_from_dataset]                                   # Extracts the label classification being benign or malware for the samples


sample_predictions = model.predict(sample_data)                                         # Uses the trained model to prdict what each of the sample class will be

for i, index in enumerate(sample_from_dataset):                                        # loops through each sample and compares the actual label of the sample with the prediction
    actual = "Benign" if sample_labels.iloc[i] == 0 else "Malware"                      # converts the numeric label to a readable text
    predicted = "Benign" if sample_predictions[i] == 0 else "Malware"                   # converts the predicted result label into text
    print(f"Sample {index} -> Actual: {actual} | Predicted: {predicted}")               # Displays the results of the actual class and predicted result

