#!/usr/bin/env python3
"""
Convert ChestAgentBench data to the JSON format required by the MaAS evaluation tool.

This script converts ChestAgentBench data to the JSON format required by the MaAS evaluation tool:
[
  {
    "query": "What pathological condition does this chest X-ray show?",
    "image_path": "data/images/chest_xray_001.jpg"
  }
]

Usage:
python convert_chestagentbench_to_maas.py --input MedRAX/chestagentbench/metadata.jsonl --output maas_eval_data.json --figures-dir MedRAX/figures --limit 10
"""

import os
import json
import argparse
from datasets import load_dataset

def convert_chestagentbench_to_maas(
    input_path: str,
    output_path: str,
    figures_dir: str = "MedRAX/figures",
    limit: int = None
) -> None:
    """Convert ChestAgentBench data to MaAS evaluation format
    
    Args:
        input_path: Path to the ChestAgentBench metadata file
        output_path: Path to the output MaAS evaluation data file
        figures_dir: Directory of the image files
        limit: Limit the number of samples to process
    """
    # Load ChestAgentBench data
    dataset = load_dataset("json", data_files=input_path)
    train_dataset = dataset["train"]
    
    # Limit the number of samples
    if limit:
        train_dataset = train_dataset.select(range(min(limit, len(train_dataset))))
    
    # Convert format
    maas_data = []
    for example in train_dataset:
        # Extract image paths
        image_paths = example['images']
        if isinstance(image_paths, str):
            image_paths = [image_paths]
        elif isinstance(image_paths[0], list):  # Handle nested lists
            image_paths = [path for sublist in image_paths for path in sublist]
        
        # Use only the first image
        if image_paths and isinstance(image_paths[0], str):
            img_path = image_paths[0].replace('figures/', '')
            full_path = os.path.join(figures_dir, img_path)
            
            # Verify that the image file exists
            if os.path.exists(full_path):
                # Create an entry in MaAS format
                maas_entry = {
                    "query": example['question'],
                    "image_path": full_path
                }
                maas_data.append(maas_entry)
    
    # Save as a JSON file
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(maas_data, f, ensure_ascii=False, indent=2)
    
    print(f"Converted {len(maas_data)} samples to {output_path}")

def parse_args():
    parser = argparse.ArgumentParser(description="Convert ChestAgentBench data to MaAS evaluation format")
    parser.add_argument("--input", type=str, default="MedRAX/chestagentbench/metadata.jsonl",
                       help="Path to the ChestAgentBench metadata file")
    parser.add_argument("--output", type=str, default="maas_eval_data.json",
                       help="Path to the output MaAS evaluation data file")
    parser.add_argument("--figures-dir", type=str, default="MedRAX/figures",
                       help="Directory of the image files")
    parser.add_argument("--limit", type=int, default=None,
                       help="Limit the number of samples to process")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    convert_chestagentbench_to_maas(
        input_path=args.input,
        output_path=args.output,
        figures_dir=args.figures_dir,
        limit=args.limit
    )