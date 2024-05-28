from PIL import Image
import os


input_dir = "data/input"
output_dir = "data/output"


target_width = 1640
target_height = 590


for folder_name in os.listdir(input_dir):
    input_folder = os.path.join(input_dir, folder_name)
    output_folder = os.path.join(output_dir, folder_name)

    if os.path.isdir(input_folder):

        os.makedirs(output_folder, exist_ok=True)


        for filename in os.listdir(input_folder):
            input_image_path = os.path.join(input_folder, filename)
            output_image_path = os.path.join(output_folder, filename)

            if os.path.isfile(input_image_path):
                try:

                    with Image.open(input_image_path) as img:

                        resized_img = img.resize((target_width, target_height), resample=Image.BILINEAR)
                        

                        resized_img.save(output_image_path)
                        print("Resized and saved:", output_image_path)
                except Exception as e:
                    print("Error processing", input_image_path, ":", str(e))

