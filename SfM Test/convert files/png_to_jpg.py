from PIL import Image
import os

def convert_png_to_jpg(input_dir, output_dir):
    for filename in os.listdir(input_dir):
        if filename.lower().endswith('.png'):
            png_path = os.path.join(input_dir, filename)
            jpg_path = os.path.join(output_dir, filename.replace('.png', '.jpg'))
            
            try:
                img = Image.open(png_path)
                # Convert RGBA to RGB if needed
                if img.mode == 'RGBA':
                    img = img.convert('RGB')
                img.save(jpg_path, 'JPEG')
                print(f"Converted {filename} to JPG")
            except Exception as e:
                print(f"Error converting {filename}: {e}")

# Usage
png_dir = 'C:\\Users\\fabia\\Desktop\\dioscuri\\images'
jpg_output_dir = 'C:\\Users\\fabia\\Desktop\\test_1\\test_conv_images'

os.makedirs(jpg_output_dir, exist_ok=True)
convert_png_to_jpg(png_dir, jpg_output_dir)