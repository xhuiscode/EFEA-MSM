from PIL import Image
import os


def convert_jpg_to_png(input_path, output_path):
    image = Image.open(input_path)
    image.save(output_path, "PNG")


jpg_folder = r"D:\edgexiazai\EfficientFace1\RAF-DB\train\1"
png_folder = r"D:\edgexiazai\EfficientFace1\raf_db_png"
file_list = os.listdir(jpg_folder)
for file_name in file_list:
    if file_name.endswith(".jpg"):
        jpg_path = os.path.join(jpg_folder, file_name)
        png_path = os.path.join(png_folder, file_name[:-4] + ".png")
        convert_jpg_to_png(jpg_path, png_path)

