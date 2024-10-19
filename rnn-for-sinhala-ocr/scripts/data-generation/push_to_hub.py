import os
import json
import pandas as pd
import time
from datasets import DatasetDict, Dataset, Image

token = 'hf_token'
texts = []
images = []
fonts = []
effects = []

# with open('sinhala_mjsynth_dataset_v2/metadata.csv', 'r', encoding='utf8') as f:
df = pd.read_csv('sinhala_mjsynth_dataset_v2/metadata.csv')
for i,line in df.iterrows():
    texts.append(line['text'])
    images.append(os.path.join('sinhala_mjsynth_dataset_v2', line['file_name']))
    fonts.append(line['font'])
    effects.append(line['additional_effects'])

# create a Dataset instance from dict
ds = Dataset.from_dict({"image": images, "text": texts, "fonts":fonts, "additional_effects":effects})
# cast the content of image column to PIL.Image
ds = ds.cast_column("image", Image())
# create train split
dataset = DatasetDict({"train": ds})
# save Arrow files locally
dataset.push_to_hub(repo_id='Ransaka/SSOCR-V.2', token=token)