import pandas as pd                                                                                     # This Library will help with cleaning dataset
from sklearn.model_selection import train_test_split                                                    # This is used for the splitting of the dataset into training and testing
from sklearn.ensemble import RandomForestClassifier                                                     # This is the random forest algorithm used for the model
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix                     # This is for the evaluation metrics used for viewing the results

ransom_dataset = pd.read_csv("Final_Dataset_without_duplicate_main.csv")                                    # Used for reading the original dataset

ransom_dataset.drop(columns=["md5", "sha1", "magic_number", "Subsystem", "DllCharacteristics",
    "text_Characteristics", "rdata_Characteristics", "Magic", "AddressOfEntryPoint",
                 "OperatingSystemVersion", "ImageVersion", "Category", "Family"], inplace=True)             # These are the irrelevant columns in the dataset that will be removed as it is not needed for the model

ransom_dataset["Class"] = ransom_dataset["Class"].map({"Benign": 0, "Malware": 1})                      # This converts the classification labels to binary being Benign - 0 and Malware - 1

def convert_hex_to_int(value):                                                                          # The columns in dataset that are hexadecimal will be converted to integers
    try:
        return int(value, 16)
    except:
        return None

                                                                                                 # the list of columns that are stored as hexadecimal strings and need converted
hexadecimal_columns = [
    'EntryPoint', 'bytes_on_last_page', 'pages_in_file', 'relocations',
    'size_of_header', 'min_extra_paragraphs', 'max_extra_paragraphs',
    'init_ss_value', 'init_sp_value', 'init_ip_value', 'init_cs_value',
    'over_lay_number', 'oem_identifier', 'address_of_ne_header',
    'SizeOfCode', 'SizeOfInitializedData', 'SizeOfUninitializedData',
    'BaseOfCode', 'BaseOfData', 'ImageBase', 'SectionAlignment',
    'FileAlignment', 'SizeOfImage', 'SizeOfHeaders', 'Checksum',
    'SizeofStackReserve', 'SizeofStackCommit', 'SizeofHeapCommit',
    'SizeofHeapReserve', 'LoaderFlags',
    'text_VirtualSize', 'text_VirtualAddress', 'text_SizeOfRawData',
    'text_PointerToRawData', 'text_PointerToRelocations',
    'text_PointerToLineNumbers',
    'rdata_VirtualSize', 'rdata_VirtualAddress', 'rdata_SizeOfRawData',
    'rdata_PointerToRawData', 'rdata_PointerToRelocations',
    'rdata_PointerToLineNumbers'
]

                                                                                                          # The conversion will be applied to the selected columns
for columns in hexadecimal_columns:
    ransom_dataset[columns] = ransom_dataset[columns].apply(convert_hex_to_int)

ransom_dataset = pd.get_dummies(ransom_dataset, columns=["file_extension", "PEType", "MachineType"], drop_first=True)               # Convert categorical features into binary columns for the model to read

features = ransom_dataset.drop("Class", axis=1)                                                                     # Splitting the dataset into its features and labels
labels = ransom_dataset["Class"]

features_train, features_test, labels_train, labels_test = train_test_split(                                # Splitting the dataset into its testing and training sets with 80% for training and 20% for testing
    features, labels, test_size=0.2, random_state=40, stratify=labels
)

print(f"Training set: {features_train.shape}, Testing set: {features_test.shape}")

model = RandomForestClassifier(random_state=42)                                                         # Model using the Random Forest algorithm
model.fit(features_train, labels_train)

test_pred = model.predict(features_test)                                                                # This will make prediction on the test set while the other set is being used for training


print("\nEvaluation")                                                                                           # The model will then be evaluated and show the results
print("Accuracy:", accuracy_score(labels_test, test_pred))                                                  # Accuracy
print("\nClassification Report:\n", classification_report(labels_test, test_pred))                          # Precision, recall, and F1-score
print("\nConfusion Matrix:\n", confusion_matrix(labels_test, test_pred))                                     # Shows breakdown of the predictions
