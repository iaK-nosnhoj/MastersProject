import pandas as pd                                                                     # This Library will help with cleaning dataset
import numpy as np                                                                      # This library is used to perform the noise in the dataset

ransom_dataset = pd.read_csv("Final_Dataset_without_duplicate_main.csv")                                # Used for reading the original dataset

ransom_dataset.drop(columns=["md5", "sha1", "magic_number", "Subsystem", "DllCharacteristics",              # These are the irrelevant columns in the dataset that will be removed as it is not needed for the model
    "text_Characteristics", "rdata_Characteristics", "Magic", "AddressOfEntryPoint",
                 "OperatingSystemVersion", "ImageVersion", "Category", "Family"], inplace=True)

ransom_dataset["Class"] = ransom_dataset["Class"].map({"Benign": 0, "Malware": 1})                      # This converts the classification labels to binary being Benign - 0 and Malware - 1

def convert_hex_to_int(value):                                                              # The columns in dataset that are hexadecimal will be converted to integers
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
for column_names in hexadecimal_columns:
    ransom_dataset[column_names] = ransom_dataset[column_names].apply(convert_hex_to_int)

ransom_dataset = pd.get_dummies(ransom_dataset, columns=["file_extension", "PEType", "MachineType"], drop_first=True)       # Convert categorical features into binary columns for the model to read


numeric_cols_for_augmentation = ransom_dataset.select_dtypes(include=[np.number]).columns.tolist()              # This selects all columns in the dataset for augmentation
numeric_cols_for_augmentation.remove("Class")                                                                   # This will make it so it does not modify the class column as its binary and should not be chnaged

number_of_new_copies = 3                                                                                        # total number of times the dataset should be duplicated and augmented which will be 3 times the size of original dataset to over 85000

augmented_data = []

for _ in range(number_of_new_copies):
                                                                                                         # Duplicates the dataset
    copy_ransom_dataset = ransom_dataset.copy()

    for column_names in numeric_cols_for_augmentation:                                                           # Apply random noise to the numeric features
        noise = np.random.normal(loc=1.0, scale=0.05, size=len(copy_ransom_dataset))                            # Generate Gaussian noise with a mean of 1.0 and standard deviation 0.05
        copy_ransom_dataset[column_names] = copy_ransom_dataset[column_names] * noise                           # Multiplies each value by its noise value to chnage the sample
        copy_ransom_dataset[column_names] = copy_ransom_dataset[column_names].round().astype(int)               # Rounds all numbers to whole numbers so values remain integers

    augmented_data.append(copy_ransom_dataset)

                                                                                                      # Adds up all augmented dataset copies and  original dataset together into one big dataset
df_augmented = pd.concat([ransom_dataset] + augmented_data, ignore_index=True)

print("Dataset Augmentation Complete")
print("Original size of Dataset:", len(ransom_dataset))
print("New augmented size:", len(df_augmented))
