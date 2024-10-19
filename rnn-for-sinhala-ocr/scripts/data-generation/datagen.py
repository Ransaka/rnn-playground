import os
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import random
import numpy as np
import cv2
from multiprocessing import Pool
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import pickle
from uuid import uuid4
from tqdm import tqdm

with open("corpus.lst","rb") as f:
    sinhala_words = pickle.load(f)

def generate_sinhala_word():
    return random.choice(sinhala_words)

def apply_random_blur(image):
    if random.random() < 0.3:  
        radius = random.uniform(1, 3)
        return image.filter(ImageFilter.GaussianBlur(radius=radius)),True
    return image,False

def apply_perspective_distortion(image):
    if random.random() < 0.2:
        img_cv = np.array(image)
        rows, cols = img_cv.shape[:2]
        pts1 = np.float32([[0, 0], [cols, 0], [0, rows], [cols, rows]])
        delta = random.uniform(10, 30)
        pts2 = np.float32([
            [random.uniform(0, delta), random.uniform(0, delta)],
            [cols - random.uniform(0, delta), random.uniform(0, delta)],
            [random.uniform(0, delta), rows - random.uniform(0, delta)],
            [cols - random.uniform(0, delta), rows - random.uniform(0, delta)]
        ])
        M = cv2.getPerspectiveTransform(pts1, pts2)
        dst = cv2.warpPerspective(img_cv, M, (cols, rows))
        return Image.fromarray(dst), True
    return image,False

def create_image(text, font_path, output_dir, min_height=32, max_height=64):
    font_size = random.randint(min_height, max_height)
    font = ImageFont.truetype(font_path, font_size)
    temp_img = Image.new('RGB', (1000, 1000), color='green')
    temp_draw = ImageDraw.Draw(temp_img)
    bbox = temp_draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    padding = max(20, int(font_size * 0.5))
    width = text_width + 2 * padding
    height = text_height + 2 * padding
    image = Image.new('RGB', (width, height), color='white')
    draw = ImageDraw.Draw(image)
    text_x = (width - text_width) // 2
    text_y = (height - text_height) // 2
    draw.text((text_x, text_y), text, font=font, fill='black')
    image,random_blur_flag = apply_random_blur(image)
    image,perspective_distortion_flag = apply_perspective_distortion(image)
    filename = f"{str(uuid4())}_{width}x{height}.png"
    filepath = os.path.join(output_dir, filename)
    image.save(filepath)
    return filename, (random_blur_flag, perspective_distortion_flag)

def generate_single_image(args):
    i, font_paths, output_dir = args
    font_path = random.choice(font_paths)
    word = generate_sinhala_word()
    filepath, (random_blur_flag, perspective_distortion_flag) = create_image(word, font_path, output_dir)
    transformations = []
    if random_blur_flag:
        transformations.append("BLUR")
    if perspective_distortion_flag:
        transformations.append("PERSPECTIVE_TRANSF")
    metadata = f"{word},{filepath},{font_path.parts[-1]},'{transformations}'\n"
    return metadata

def generate_dataset(num_images, font_paths, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    with open("metadata_v2.csv", "w", encoding='utf-8') as f:
        with Pool() as pool:
            args = [(i, font_paths, output_dir) for i in range(num_images)]
            for metadata in tqdm(pool.imap_unordered(generate_single_image, args), 
                                 total=num_images, desc="Generating images..", unit="image"):
                f.write(metadata)

def generate_single_image_wrapper(args):
    return generate_single_image(*args)

if __name__ == "__main__":
    font_paths = list(Path("/home/user/.cache/kagglehub/datasets/ransakaravihara/google-fonts-sinhala/versions/1").glob("*/static/*.ttf"))
    output_dir = "sinhala_mjsynth_dataset_v2"
    num_images = 700_000

    generate_dataset(num_images, font_paths, output_dir)
