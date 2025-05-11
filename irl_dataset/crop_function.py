import cv2
import os
import yaml

input_folder = '4_channels_perfect'        # 输入图片文件夹
output_folder = '4_channels_perfect_cropped'   # 输出保存文件夹
x, y, w, h = 790, 540, 350, 350    # 裁剪区域坐标

os.makedirs(output_folder, exist_ok=True)

files_skipped = []
for filename in os.listdir(input_folder):
    # or not #image_100_light"filename.lower() == "":
    if not filename.lower().endswith('0.png') or filename.lower() in os.listdir(output_folder):
        continue

    input_path = os.path.join(input_folder, filename)
    image = cv2.imread(input_path)
    _x, _y, _w, _h = 500, 300, 1100, 1000
    image = image[_y:_y+_h, _x:_x+_w]
    img_height, img_width = image.shape[:2]
    preview = image.copy()

    # Detect contours in image
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Blurring the image
    gray = cv2.GaussianBlur(gray, (11, 11), 50)
    # Use canny edge detection
    _, thresh = cv2.threshold(gray, 15, 255, cv2.THRESH_BINARY)
    # Apply local histogram equalization

    gray = cv2.equalizeHist(gray)
    # Apply Gaussian blur
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    # Apply adaptive thresholding
    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)

    # Show image

    edges = cv2.Canny(thresh, 1, 200)

    # show image
    if False:
        cv2.imshow('Threshold', gray)
        cv2.waitKey(0)
        cv2.imshow('Threshold', thresh)
        cv2.waitKey(0)
        cv2.imshow('Threshold', edges)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    contours, _ = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Filter contours based on area

    # Show contours on the image
    cv2.drawContours(preview, contours, -1, (0, 255, 0), 2)
    # Show the image with contours
    # Keep contours with the area closes to w*h
    filtered_contours = sorted(
        contours, key=lambda cnt: abs(cv2.contourArea(cnt) - (w * h)))
    # Largest contour
    filtered_contours = sorted(
        filtered_contours, key=cv2.contourArea, reverse=True)
    # Keep only the top contour
    if len(filtered_contours) > 0:
        filtered_contours = filtered_contours[:1]
    else:
        print(f"[!] No contours found in: {input_path}")
        continue
    # Draw contours on the image
    for cnt in filtered_contours:
        cv2.drawContours(preview, [cnt], -1, (0, 255, 0), 2)

    # Place the rectangle on the image with same center as contour
    for cnt in filtered_contours:
        M = cv2.moments(cnt)
        if M["m00"] == 0:
            continue
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        _x, _y, _w, _h = cv2.boundingRect(cnt)
        x = int(cX - w / 2)
        y = int(cY - h / 2)
        # Draw the rectangle on the image
        cv2.rectangle(preview, (x, y), (x + w, y + h), (255, 0, 0), 2)

    x, y, w, h = 360, 190, 350, 350    # 裁剪区域坐标
    new_preview = image.copy()
    cv2.rectangle(new_preview, (x, y), (x + w, y + h), (255, 0, 0), 2)
    # Show image with coordinates of cursor

    # Show the image with contours
    # cv2.imshow('Rectangle', new_preview)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    if image is None:
        print(f"[!] Failed to read: {input_path}")
        continue

    # Draw the red crop rectangle for preview
    top_left = (x, y)
    bottom_right = (x + w, y + h)
    #
    cv2.rectangle(preview, top_left, bottom_right, (0, 0, 255), 2)

    # Show preview
    cv2.imshow(
        'Crop Preview - Press "s" to save, any other key to skip', new_preview)
    key = cv2.waitKey(0) & 0xFF
    cv2.destroyAllWindows()

    if key == ord('s'):
        # Perform the actual crop and save
        for i in range(4):
            file_i = filename
            file_i = file_i[:-5] + str(i) + file_i[-4:]
            input_path = os.path.join(input_folder, file_i)
            image = cv2.imread(input_path)
            _x, _y, _w, _h = 500, 300, 1100, 1000
            image = image[_y:_y+_h, _x:_x+_w]
            cropped = image[y:y+h, x:x+w]

            output_path = os.path.join(output_folder, file_i)
            cv2.imwrite(output_path, cropped)
            print(f"[+] Saved: {output_path}")
    else:
        print(f"[-] Skipped: {filename}")
        files_skipped.append(filename)

# Save skipped files to a YAML file
skipped_file_path = os.path.join(
    output_folder, 'skipped_files.yaml')
with open(skipped_file_path, 'w') as f:
    yaml.dump(files_skipped, f)
print(f"Skipped files saved to {skipped_file_path}")
print("Done.")
