import subprocess
import json
import os
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)

def generate_datasets(num_runs, desired_dataset_size, chunk_size, temp_dir, timeout=120):
    temp_dir_path = Path(temp_dir)
    temp_dir_path.mkdir(parents=True, exist_ok=True)
    output_files = []

    for i in range(num_runs):
        output_file = temp_dir_path / f"dataset_part_{i+1}.json"
        try:
            subprocess.run([
                'python', 'generate_dataset.py',
                '--size', str(desired_dataset_size),
                '--chunk_size', str(chunk_size),
                '--output', str(output_file)
            ], check=True, timeout=timeout)
            output_files.append(output_file)
        except subprocess.CalledProcessError as e:
            logging.error(f"Failed to generate dataset part {i+1}: {e}")
        except subprocess.TimeoutExpired:
            logging.error(f"Timeout expired for dataset part {i+1}. Skipping this part.")
            continue  # Skip to the next iteration

    return output_files

def combine_datasets(output_files, combined_output_file):
    combined_data = []

    for file in output_files:
        try:
            with open(file, 'r') as f:
                data = json.load(f)
                combined_data.extend(data)
        except Exception as e:
            logging.error(f"Failed to read or parse {file}: {e}")
            continue  # Skip to the next file

    with open(combined_output_file, 'w') as f:
        json.dump(combined_data, f, indent=4)

if __name__ == "__main__":
    # Parameters
    num_runs = 10
    desired_dataset_size = 4000
    chunk_size = 200
    temp_dir = "temp_datasets"
    combined_output_file = "combined_dataset.json"

    # Generate datasets
    output_files = generate_datasets(num_runs, desired_dataset_size, chunk_size, temp_dir)

    # Combine datasets
    combine_datasets(output_files, combined_output_file)

    # Clean up temporary files (optional)
    for file in output_files:
        try:
            os.remove(file)
        except Exception as e:
            logging.error(f"Failed to delete {file}: {e}")