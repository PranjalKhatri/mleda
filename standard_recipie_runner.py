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

def main():
    if len(sys.argv) != 2:
        print("Usage: python standard_recipie_runner.py <design_path>", file=sys.stderr)
        print("Example: python standard_recipie_runner.py data/designs/i2c.aig", file=sys.stderr)
        sys.exit(1)
    
    design_path = Path(sys.argv[1])
    lib_path = Path("nangate45.lib")
    
    if not design_path.exists():
        print(f"Error: Design file '{design_path}' not found.", file=sys.stderr)
        sys.exit(1)
        
    if not lib_path.exists():
        print(f"Error: Library file '{lib_path}' not found in the current directory.", file=sys.stderr)
        sys.exit(1)

    # Create output directory if it doesn't exist
    output_dir = Path("abcStats")
    output_dir.mkdir(exist_ok=True)
    
    # Generate output CSV filename based on design name
    design_name = design_path.stem
    output_csv = output_dir / f"{design_name}.csv"
    
    results = []
    
    print(f"Running all aliases for {design_path}...")
    
    for alias in ALIASES:
        print(f"  Processing alias: {alias}...", end=" ", flush=True)
        
        # Construct the full ABC command using the alias directly
        abc_commands = [
            f"read_lib {lib_path}",
            f"read {design_path}",
            alias,
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
        print(f"\nResults saved to {output_csv}")
    else:
        print("Error: No results to save.", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()