import os
import subprocess
from PIL import Image

def convert_heic_to_png(folder_path):
    if not os.path.isdir(folder_path):
        print("Folder does not exist:", folder_path)
        return

    for filename in os.listdir(folder_path):
        if filename.lower().endswith(".heic"):
            # Construct the full file path
            file_path = os.path.join(folder_path, filename)
            new_filename = filename.rsplit('.', 1)[0] + '.png'
            new_file_path = os.path.join(folder_path, new_filename)

            # Use ImageMagick to convert the file
            try:
                subprocess.run(["magick", "convert", file_path, new_file_path], check=True)
                print(f"Converted {filename} to {new_filename}")
            except subprocess.CalledProcessError as e:
                print(f"Could not convert {filename}. Error: {e}")

def delete_heic_files(folder_path):
    # Check if the folder exists
    if not os.path.isdir(folder_path):
        print("Folder does not exist:", folder_path)
        return
    
    # Count of deleted files
    deleted_count = 0

    # Loop through all files in the folder
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(".heic"):
            # Construct the full file path
            file_path = os.path.join(folder_path, filename)
            # Delete the file
            os.remove(file_path)
            print(f"Deleted {filename}")
            deleted_count += 1

    if deleted_count == 0:
        print("No .heic files found to delete.")
    else:
        print(f"Total deleted .heic files: {deleted_count}")


folder_path = "/Users/kaiqiz/Documents/mbp-data/Documents/NEU-MSCS-Align/2024 Spring/CS5330/Hw+Proj/cv-project3/database/highlighter"
convert_heic_to_png(folder_path)
delete_heic_files(folder_path)
