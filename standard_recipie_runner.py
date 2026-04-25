#!/usr/bin/env python3
import subprocess
import re
import sys
import csv
from pathlib import Path

# List of all aliases (will be resolved via abc.rc)
ALIASES = [
    "resyn", "resyn2", "resyn2a", "resyn3",
    "compress", "compress2",
    "choice", "choice2",
    "rwsat", "drwsat2",
    "share", "addinit", "blif2aig",
    "v2p", "g2p",
    "&sw_", "&fx_", "&dc3", "&dc4",
    "src_rw", "src_rs", "src_rws",
    "resyn2rs", "r2rs", "compress2rs", "c2rs",
    "&resyn2rs", "&compress2rs"
]

def extract_power(abc_output):
    # Regex to capture the numeric value assigned to 'Power'
    match = re.search(r'Power\s*=\s*([0-9\.eE\-\+]+)', abc_output, re.IGNORECASE)
    if match:
        return float(match.group(1))
    return None

def process_design(design_path, lib_path, output_dir):
    """Process a single design and run all aliases."""
    design_name = design_path.stem
    output_csv = output_dir / f"{design_name}.csv"
    
    results = []
    
    print(f"\nRunning all aliases for {design_path}...")
    
    for alias in ALIASES:
        print(f"  Processing alias: {alias}...", end=" ", flush=True)
        
        # Construct the full ABC command using the alias directly
        abc_commands = [
            f"read_lib {lib_path}",
            f"read {design_path}",
            alias,
            "map",
            "print_stats -p"
        ]
        
        abc_cmd_string = "; ".join(abc_commands)
        
        try:
            # Run ABC and capture stdout
            result = subprocess.run(
                ["abc", "-c", abc_cmd_string],
                capture_output=True,
                text=True,
                check=True,
                timeout=300
            )
            
            power_val = extract_power(result.stdout)
            
            if power_val is not None:
                results.append({"alias": alias, "power": power_val})
                print(f"Power={power_val}")
            else:
                print("ERROR: Could not extract Power")
                
        except subprocess.TimeoutExpired:
            print("ERROR: Timeout")
        except subprocess.CalledProcessError as e:
            print(f"ERROR: ABC execution failed")
        except FileNotFoundError:
            print("Error: 'abc' executable not found in PATH.", file=sys.stderr)
            sys.exit(1)
    
    # Write results to CSV
    if results:
        with open(output_csv, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=["alias", "power"])
            writer.writeheader()
            writer.writerows(results)
        print(f"  Results saved to {output_csv}")
    else:
        print(f"  WARNING: No results to save for {design_name}.")

def main():
    lib_path = Path("nangate45.lib")
    
    if not lib_path.exists():
        print(f"Error: Library file '{lib_path}' not found in the current directory.", file=sys.stderr)
        sys.exit(1)

    # Create output directory if it doesn't exist
    output_dir = Path("abcStats")
    output_dir.mkdir(exist_ok=True)
    
    # Find all .aig files in data/designs
    designs_dir = Path("data/designs")
    
    if not designs_dir.exists():
        print(f"Error: Designs directory '{designs_dir}' not found.", file=sys.stderr)
        sys.exit(1)
    
    design_files = sorted(designs_dir.glob("*.aig"))
    
    if not design_files:
        print(f"Error: No .aig files found in '{designs_dir}'.", file=sys.stderr)
        sys.exit(1)
    
    print(f"Found {len(design_files)} design(s) to process")
    
    # Process each design
    for design_path in design_files:
        process_design(design_path, lib_path, output_dir)

if __name__ == "__main__":
    main()