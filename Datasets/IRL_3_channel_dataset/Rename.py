import os

# Define the directory containing the files
directory = r"Datasets/IRL_3_channel_dataset/Test/Defects/"

# Ensure the directory exists
if not os.path.exists(directory):
    print(f"Directory '{directory}' does not exist.")
    print("Current working directory:", os.getcwd())
    exit()

# Get a sorted list of all files in the directory
files = sorted(os.listdir(directory))

# Rename each file with a zero-padded number
for index, filename in enumerate(files):
    # Construct the new filename
    new_name = f"defect{index:04d}.png"
    print(f"Renaming '{filename}' to '{new_name}'")
    # Get the full paths for the old and new filenames
    old_path = os.path.join(directory, filename)
    new_path = os.path.join(directory, new_name)
    
    # Rename the file
    os.rename(old_path, new_path)

print("Files have been renamed successfully.")