import os
from langchain_core.tools import tool

@tool
def files_in_directory(directory: str = './data/data_model') -> list[str] | str:
    '''Return the List all files in a directory and its subdirectories.'''
    all_files = []
    try:
        print("Calling files_in_directory...")
        print("Getting files...")
        for root, dirs, files in os.walk(directory):
        # Skip the .git directory
            if '.git' in dirs:
                dirs.remove('.git')

            # Write only the files
            all_files.extend([os.path.join(root, file) for file in files])
        print(f"Got {len(all_files)} files.")
        return all_files
    except Exception as e:
        return f"Error in getting files: {e}"