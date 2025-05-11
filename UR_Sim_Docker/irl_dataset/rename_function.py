import cv2
import os
import yaml
from tqdm import tqdm

input_folder = '4_channels_perfect_cropped'        # 输入图片文件夹
output_folder = 'Train'   # 输出保存文件夹

os.makedirs(output_folder, exist_ok=True)
for filename in tqdm(os.listdir(input_folder)):
    image_number = filename.split('_')[1]
    light_number = filename.split('_')[3][0]
    new_filename = f"clean_{image_number}_light_{light_number}.png"

    # Save the image with the new filename
    image = cv2.imread(os.path.join(input_folder, filename))
    cv2.imwrite(os.path.join(output_folder, new_filename), image)
