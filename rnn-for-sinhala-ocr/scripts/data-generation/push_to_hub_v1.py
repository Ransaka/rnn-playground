import pandas as pd
from pathlib import Path
import os
from multiprocessing import Pool
from functools import partial
from datasets import load_dataset
from tqdm import tqdm

token = 'hf_token'

def delete_file(img_fp, files):
    """Delete file if it's not in the metadata."""
    if img_fp.parts[-1] not in files:
        os.remove(img_fp)
        return img_fp
    return None

def post_processing():
    # Load and clean metadata
    df = pd.read_csv("metadata_v2.csv", header=None, encoding='utf-8', on_bad_lines='skip', names=['text', 'file_name', 'font', 'additional_effects'])
    df.dropna(inplace=True)
    df.to_csv("sinhala_mjsynth_dataset_v2/metadata.csv", index=False)

    # Get list of valid files from the metadata
    files = df['file_name'].tolist()

    # Get list of all images in the dataset folder
    images_available = list(Path("sinhala_mjsynth_dataset_v2").glob("*.png"))

    # Use multiprocessing Pool for parallel file deletion
    N = 0
    with Pool() as pool:
        # Use tqdm to show the progress bar
        results = list(tqdm(pool.starmap(delete_file, [(img_fp, files) for img_fp in images_available]), total=len(images_available), desc="Deleting Files"))

    # Count how many files were deleted
    N = sum(1 for result in results if result is not None)
    print(f"\n{N} Files Removed...")

    # Load the dataset and split into training and test sets
    dataset = load_dataset("imagefolder", data_dir="sinhala_mjsynth_dataset_v2", num_proc=16)
    dataset = dataset['train'].train_test_split(train_size=0.8)
    print(dataset)

    # Push the dataset to the Hugging Face hub
    dataset.push_to_hub(repo_id='Ransaka/SSOCR-V.2', token=token)

post_processing()
