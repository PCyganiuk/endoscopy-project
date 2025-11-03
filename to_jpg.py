import os
from PIL import Image, UnidentifiedImageError
import shutil

input_folder = "/mnt/e/galar"
output_folder = "/mnt/e/galar_jpg"

def convert_png_to_jpg_preserve_structure(input_folder, output_folder):
    for root, dirs, files in os.walk(input_folder):
        rel_path = os.path.relpath(root, input_folder)
        if rel_path.split('/')[0].isdigit():
            if int(rel_path.split('/')[0]) <= 64:
                continue
        if os.path.basename(root).lower() == "labels":
            rel_path = os.path.relpath(root, input_folder)
            target_folder = os.path.join(output_folder, rel_path)
            os.makedirs(target_folder, exist_ok=True)
            
            for file in files:
                src_path = os.path.join(root, file)
                dst_path = os.path.join(target_folder, file)
                shutil.copy2(src_path, dst_path)
            print(f"Copied 'labels' folder: {root} -> {target_folder}")
            continue

        # Otherwise, convert PNGs to JPG
        #rel_path = os.path.relpath(root, input_folder)
        target_folder = os.path.join(output_folder, rel_path)
        os.makedirs(target_folder, exist_ok=True)
        
        for file in files:
            if file.lower().endswith(".png"):
                png_path = os.path.join(root, file)
                jpg_path = os.path.join(target_folder, os.path.splitext(file)[0] + ".jpg")
                
                try:
                    with Image.open(png_path) as img:
                        rgb_img = img.convert("RGB")
                        rgb_img.save(jpg_path, "JPEG")
                    print(f"Converted: {png_path} -> {jpg_path}")
                except UnidentifiedImageError:
                    print(f"Skipped (not an image or corrupted): {png_path}")

if __name__ == "__main__":
    convert_png_to_jpg_preserve_structure(input_folder, output_folder)
