import os

# Define the directory containing the files
directory = r"Datasets/IRL_3_channel_dataset/Test_Small_defects/"

# Initialize a counter for renaming
counter = 0

# Iterate through all files in the directory
for filename in os.listdir(directory):
    # Check if the filename matches the pattern of a four-digit number followed by ".png"
    if filename.endswith(".png") and filename[:4].isdigit() and len(filename[:4]) == 4:
        print(f"Renaming '{filename}' to 'good{counter:04d}.png'")
        # Construct the new filename
        new_name = f"good{counter:04d}.png"
        # Get the full paths
        old_path = os.path.join(directory, filename)
        new_path = os.path.join(directory, new_name)
        # Rename the file
        os.rename(old_path, new_path)
        # Increment the counter
        counter += 1