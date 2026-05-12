"""
Spanish MNIST Dataset Generator
Generates 28x28 grayscale handwriting-style PNG images for all Spanish characters.

Usage:
    python utils/generate_dataset.py --samples_per_class 100 --output data/raw
"""

import os
import csv
import random
import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter

FONT_PATHS = [
    "/usr/share/fonts/truetype/freefont/FreeSerif.ttf",
    "/usr/share/fonts/truetype/freefont/FreeSerifItalic.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSerif-Italic.ttf",
    "/usr/share/fonts/truetype/crosextra/Caladea-Regular.ttf",
    "/usr/share/fonts/truetype/crosextra/Caladea-Italic.ttf",
]
FONT_PATHS = [f for f in FONT_PATHS if os.path.exists(f)]


def build_char_list():
    """Return the full list of Spanish MNIST character definitions."""
    chars = []
    cid = 0

    for i in range(10):
        chars.append({"id": cid, "char": str(i), "category": "digit", "label": f"digit_{i}"})
        cid += 1
    for c in range(65, 91):
        ch = chr(c)
        chars.append({"id": cid, "char": ch, "category": "uppercase", "label": f"upper_{ch}"})
        cid += 1
    for ch in ["Á", "É", "Í", "Ó", "Ú", "Ü", "Ñ"]:
        chars.append({"id": cid, "char": ch, "category": "accented_upper", "label": f"upper_{ch}"})
        cid += 1
    for c in range(97, 123):
        ch = chr(c)
        chars.append({"id": cid, "char": ch, "category": "lowercase", "label": f"lower_{ch}"})
        cid += 1
    for ch in ["á", "é", "í", "ó", "ú", "ü", "ñ"]:
        chars.append({"id": cid, "char": ch, "category": "accented_lower", "label": f"lower_{ch}"})
        cid += 1
    for ch, name in [("¡", "inv_exclaim"), ("¿", "inv_question"), ("«", "left_angle"), ("»", "right_angle")]:
        chars.append({"id": cid, "char": ch, "category": "special", "label": name})
        cid += 1

    return chars


def generate_image(char: str, seed: int, size: int = 28) -> Image.Image:
    """
    Generate a single synthetic handwritten-style image of a character.

    Args:
        char : The character to render.
        seed : Random seed for reproducibility.
        size : Output image size in pixels (default: 28).

    Returns:
        PIL Image (grayscale, size x size).
    """
    rng = random.Random(seed)
    np_rng = np.random.RandomState(seed)

    CANVAS = size * 2  # larger canvas for rotation headroom

    img = Image.new("L", (CANVAS, CANVAS), color=0)
    draw = ImageDraw.Draw(img)

    font_path = rng.choice(FONT_PATHS) if FONT_PATHS else None
    font_size = rng.randint(int(size * 1.0), int(size * 1.3))

    try:
        font = ImageFont.truetype(font_path, font_size) if font_path else ImageFont.load_default()
    except Exception:
        font = ImageFont.load_default()

    bbox = draw.textbbox((0, 0), char, font=font)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]

    x = (CANVAS - tw) // 2 - bbox[0] + rng.randint(-3, 3)
    y = (CANVAS - th) // 2 - bbox[1] + rng.randint(-3, 3)

    brightness = rng.randint(200, 255)
    draw.text((x, y), char, fill=brightness, font=font)

    # Random rotation
    angle = rng.uniform(-12, 12)
    img = img.rotate(angle, resample=Image.BILINEAR, center=(CANVAS // 2, CANVAS // 2))

    # Crop to center
    left = (CANVAS - size) // 2
    top  = (CANVAS - size) // 2
    img = img.crop((left, top, left + size, top + size))

    # Add Gaussian noise
    arr = np.array(img, dtype=np.float32)
    noise = np_rng.normal(0, rng.uniform(4, 10), arr.shape)
    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
    img = Image.fromarray(arr)

    # Optional blur
    if rng.random() > 0.4:
        img = img.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.3, 0.7)))

    return img


def generate_dataset(output_dir: str, samples_per_class: int = 10, seed: int = 42):
    """
    Generate the full Spanish MNIST dataset to disk.

    Args:
        output_dir       : Root output directory (images/ folder created inside).
        samples_per_class: Number of samples to generate per character class.
        seed             : Base random seed.
    """
    output_dir = Path(output_dir)
    img_dir = output_dir / "images"
    chars = build_char_list()
    csv_rows = []

    print(f"Generating {len(chars)} classes × {samples_per_class} samples = "
          f"{len(chars) * samples_per_class} images")
    print(f"Output: {output_dir}")

    for i, char_info in enumerate(chars):
        char_dir = img_dir / char_info["label"]
        char_dir.mkdir(parents=True, exist_ok=True)

        for s in range(samples_per_class):
            img_seed = seed + char_info["id"] * 10_000 + s
            img = generate_image(char_info["char"], img_seed)
            filename = f"{char_info['label']}_{s:04d}.png"
            filepath = char_dir / filename
            img.save(filepath)

            csv_rows.append({
                "class_id":    char_info["id"],
                "character":   char_info["char"],
                "category":    char_info["category"],
                "label":       char_info["label"],
                "sample_index": s,
                "filename":    f"images/{char_info['label']}/{filename}",
                "unicode":     f"U+{ord(char_info['char']):04X}",
                "width":       28,
                "height":      28,
                "color_mode":  "grayscale",
            })

        if (i + 1) % 20 == 0 or (i + 1) == len(chars):
            print(f"  [{i+1}/{len(chars)}] Done: {char_info['label']}")

    csv_path = output_dir / "spanish_mnist.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=csv_rows[0].keys())
        writer.writeheader()
        writer.writerows(csv_rows)

    print(f"\n✅ Dataset generated: {len(csv_rows)} images")
    print(f"   CSV:    {csv_path}")
    print(f"   Images: {img_dir}")


def main():
    parser = argparse.ArgumentParser(description="Spanish MNIST Dataset Generator")
    parser.add_argument("--output", type=str, default="data/raw",
                        help="Output directory (default: data/raw)")
    parser.add_argument("--samples_per_class", type=int, default=10,
                        help="Number of samples per character class (default: 10)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    args = parser.parse_args()

    generate_dataset(args.output, args.samples_per_class, args.seed)


if __name__ == "__main__":
    main()
