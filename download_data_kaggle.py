import os
import zipfile

# Set up Kaggle credentials
os.environ['KAGGLE_CONFIG_DIR'] = os.path.expanduser('~/.kaggle')

# Download the dataset
os.system("kaggle datasets download -d lpitters/yelp-academic-dataset")

dataset_zip = f"yelp-academic-dataset.zip"

if os.path.exists(dataset_zip):
    print("Extracting dataset...")
    with zipfile.ZipFile(dataset_zip, 'r') as zip_ref:
        	zip_ref.extractall("data")
os.remove(dataset_zip)

#handle if dataset alrready exists