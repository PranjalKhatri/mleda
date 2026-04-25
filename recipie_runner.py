#!/usr/bin/env python3
import argparse
import subprocess
import ast
import sys

def main():
    parser = argparse.ArgumentParser(description="Execute an ABC synthesis recipe read from stdin.")
    parser.add_argument("lib_path", help="Path to the technology library (e.g., ./testing/nangate45.lib)")
    parser.add_argument("design_path", help="Path to the input design (e.g., ./data/designs/i2c.aig)")
    
    args = parser.parse_args()
    
    # Read the raw string from standard input
    raw_input = sys.stdin.read().strip()
    
    if not raw_input:
        print("Error: No recipe received on stdin.", file=sys.stderr)
        sys.exit(1)
    
    # Parse the input as comma-separated or newline-separated commands
    if ',' in raw_input:
        # Comma-separated format
        recipe_list = [cmd.strip() for cmd in raw_input.split(',') if cmd.strip()]
    else:
        # Newline-separated format (or try to parse as Python list as fallback)
        lines = raw_input.split('\n')
        if len(lines) > 1:
            recipe_list = [cmd.strip() for cmd in lines if cmd.strip()]
        else:
            # Try Python list format as fallback
            try:
                parsed = ast.literal_eval(raw_input)
                if isinstance(parsed, list):
                    recipe_list = parsed
                else:
                    raise ValueError("Not a list")
            except (ValueError, SyntaxError):
                # Single command case
                recipe_list = [raw_input] if raw_input else []
    
    if not recipe_list:
        print("Error: Recipe list is empty.", file=sys.stderr)
        sys.exit(1)
        
    # Construct the sequence of ABC commands
    abc_commands = [
        f"read_lib {args.lib_path}",
        f"read {args.design_path}",
        "print_stats -p"  # Initial stats before optimization
    ]
    
    # Append the steps from the parsed recipe list
    abc_commands.extend(recipe_list)
    
    # Add final stats to compare QoR
    abc_commands.append("print_stats -p")
    
    # Join everything into a single semi-colon separated string for ABC
    abc_cmd_string = "; ".join(abc_commands)
    
    print("[*] Launching ABC...\n" + "-"*40)
    
    # Execute the command
    try:
        subprocess.run(["abc", "-c", abc_cmd_string], check=True)
    except FileNotFoundError:
        print("Error: The 'abc' executable was not found. Please ensure it is compiled and in your system's PATH.", file=sys.stderr)
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        print(f"\n[!] ABC execution failed with return code {e.returncode}", file=sys.stderr)
        sys.exit(e.returncode)

if __name__ == "__main__":
    main()