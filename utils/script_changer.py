import os
import re

def process_scripts(folder_path="../data/scripts"):
    # Check if the folder exists
    if not os.path.exists(folder_path):
        print(f"Error: The folder '{folder_path}' does not exist in the current directory.")
        return

    # Loop through all files in the directory
    for filename in os.listdir(folder_path):
        old_filepath = os.path.join(folder_path, filename)

        # Skip directories, only process files
        if not os.path.isfile(old_filepath):
            continue

        # Extract the number from the old filename (e.g., 'abc0.script' -> '0')
        match = re.search(r'(\d+)', filename)
        if match:
            number = match.group(1)
            new_filename = f"script{number}.txt"
        else:
            # Fallback just in case a file doesn't have a number in its name
            name_without_ext = os.path.splitext(filename)[0]
            new_filename = name_without_ext.replace('abc', 'script') + ".txt"

        new_filepath = os.path.join(folder_path, new_filename)

        try:
            # Read all lines from the current file
            with open(old_filepath, 'r', encoding='utf-8') as file:
                lines = file.readlines()

            # Remove the first 2 lines and the last 8 lines
            if len(lines) > 10:
                modified_lines = lines[2:-8]
            else:
                modified_lines = [] 

            # Write the modified lines to the NEW file
            with open(new_filepath, 'w', encoding='utf-8') as file:
                file.writelines(modified_lines)

            # Delete the old file (only if the new name is actually different)
            if old_filepath != new_filepath:
                os.remove(old_filepath)

            # --- VERIFICATION STEP ---
            actual_lines = len(modified_lines)
            if actual_lines == 20:
                verify_msg = "✅ Verified: Exactly 20 lines."
            else:
                verify_msg = f"⚠️ Warning: Contains {actual_lines} lines (expected 20)."

            print(f"Processed: '{filename}' -> '{new_filename}' | {verify_msg}")

        except Exception as e:
            print(f"Failed to process '{filename}': {e}")

if __name__ == "__main__":
    process_scripts()