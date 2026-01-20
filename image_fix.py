from PIL import Image
import os


def check_image(image_path):
    try:
        with Image.open(image_path) as img:
            img.verify()
        return True
    except Exception as e:
        print(f"erro : {image_path} |: {str(e)}")
        return False

image_dir = "/data2/qfli/before_data2_image"
png_files = [f for f in os.listdir(image_dir) if f.lower().endswith(".png")]
total_files = len(png_files)
deleted_count = 0



for i, img_name in enumerate(png_files, 1):
    img_path = os.path.join(image_dir, img_name)


    print(f"\r🔄 进度: {i}/{total_files} ({i / total_files:.1%})", end="", flush=True)

    if not check_image(img_path):
        try:
            os.remove(img_path)
            deleted_count += 1
            print(f"\ delete n: {img_path}")
        except Exception as e:
            print(f"\n erro: {img_path} | erro: {str(e)}")

print(f"\n ok")