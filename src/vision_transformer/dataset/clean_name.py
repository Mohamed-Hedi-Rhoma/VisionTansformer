import os
import re

def sanitize_name(name):
    """Remove forbidden characters from filename/folder name"""
    # Replace apostrophes and other special characters with underscore
    forbidden_chars = r'[<>:"/\\|?*\']'
    sanitized = re.sub(forbidden_chars, '_', name)
    # Remove multiple underscores
    sanitized = re.sub(r'_+', '_', sanitized)
    # Remove leading/trailing underscores
    sanitized = sanitized.strip('_')
    return sanitized

def rename_folders_and_files(root_path):
    """Rename ALL folders and files to remove forbidden characters"""
    # Walk from bottom to top (topdown=False) to rename children before parents
    for dirpath, dirnames, filenames in os.walk(root_path, topdown=False):
        
        # Rename all FILES
        for filename in filenames:
            old_path = os.path.join(dirpath, filename)
            new_filename = sanitize_name(filename)
            
            if new_filename != filename:
                new_path = os.path.join(dirpath, new_filename)
                print(f"Renaming file: {filename} -> {new_filename}")
                os.rename(old_path, new_path)
        
        # Rename all DIRECTORIES (folders)
        for dirname in dirnames:
            old_path = os.path.join(dirpath, dirname)
            new_dirname = sanitize_name(dirname)
            
            if new_dirname != dirname:
                new_path = os.path.join(dirpath, new_dirname)
                print(f"Renaming directory: {dirname} -> {new_dirname}")
                os.rename(old_path, new_path)

# Replace with your actual path
root_directory = "ai_training_data"
rename_folders_and_files(root_directory)
print("Done! All folders and files renamed.")