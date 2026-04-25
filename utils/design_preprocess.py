import os
import subprocess
from pathlib import Path


def convert_bench_to_aig(folder_path):
    """
    Scans a folder for .bench files and converts them to .aig files using abc.
    
    Args:
        folder_path (str): Path to the folder containing .bench files
    """
    folder_path = Path(folder_path)
    
    if not folder_path.exists():
        print(f"Error: Folder '{folder_path}' does not exist.")
        return
    
    if not folder_path.is_dir():
        print(f"Error: '{folder_path}' is not a directory.")
        return
    
    # Find all .bench files
    bench_files = list(folder_path.glob("**/*.bench"))
    
    if not bench_files:
        print(f"No .bench files found in '{folder_path}'")
        return
    
    print(f"Found {len(bench_files)} .bench file(s)")
    
    # Convert each .bench file to .aig using abc
    for bench_file in bench_files:
        aig_file = bench_file.with_suffix(".aig")
        
        try:
            # Use abc to convert bench to aig
            cmd = [
                "abc",
                "-c",
                f"read {bench_file};strash; write {aig_file}; quit"
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            if result.returncode == 0:
                print(f"✓ Converted: {bench_file} -> {aig_file}")
            else:
                print(f"✗ Failed to convert {bench_file}: {result.stderr}")
        
        except subprocess.TimeoutExpired:
            print(f"✗ Timeout while converting {bench_file}")
        except FileNotFoundError:
            print("✗ abc tool not found. Please ensure abc is installed and in PATH.")
            break
        except Exception as e:
            print(f"✗ Error converting {bench_file}: {str(e)}")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python design_preprocess.py <folder_path>")
        sys.exit(1)
    
    folder = sys.argv[1]
    convert_bench_to_aig(folder)
