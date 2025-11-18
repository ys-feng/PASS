#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
eval_maas.py - Simplified script for evaluating the Medical Multi-Agent System (MaAS)

This is a wrapper script for evaluate_maas.py, providing a more concise interface to evaluate the MaAS controller.
"""

import os
import sys
import argparse
import asyncio
from pathlib import Path

def parse_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(description="Evaluate MaAS controller performance")
    
    # Required arguments
    parser.add_argument("--data", "-d", type=str, required=True,
                       help="Path to the evaluation dataset JSON file")
    parser.add_argument("--checkpoint", "-c", type=str, required=True,
                       help="Path to the controller checkpoint file")
    
    # Optional arguments (with reasonable defaults)
    parser.add_argument("--model", "-m", type=str, default="gpt-4o",
                       help="Name of the LLM model to use")
    parser.add_argument("--output", "-o", type=str, default="output",
                       help="Output directory for evaluation results")
    parser.add_argument("--baseline", "-b", type=str, default=None,
                       help="Baseline results file (for comparison)")
    parser.add_argument("--device", type=str, default="cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu",
                       help="Device to run on (cuda/cpu)")
    parser.add_argument("--concurrent", type=int, default=5,
                       help="Maximum number of concurrent evaluations")
    
    # Advanced arguments
    parser.add_argument("--temperature", type=float, default=0.7,
                       help="LLM sampling temperature")
    parser.add_argument("--top_p", type=float, default=0.95,
                       help="LLM sampling top_p value")
    parser.add_argument("--model_dir", type=str, default="model-weights",
                       help="Directory for model weights")
    parser.add_argument("--temp_dir", type=str, default="temp",
                       help="Directory for temporary files")
    
    return parser.parse_args()

async def main():
    """Main function"""
    args = parse_args()
    
    # Validate paths
    if not os.path.exists(args.data):
        print(f"Error: Evaluation data file not found: {args.data}")
        return 1
    
    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint file not found: {args.checkpoint}")
        return 1
    
    # Import functions from evaluate_maas.py
    try:
        from medrax.evaluate_maas import (
            create_workflow, 
            load_evaluation_data,
            evaluate_dataset,
            generate_evaluation_report,
            compare_with_baseline
        )
    except ImportError as e:
        print(f"Error: Could not import evaluation module: {str(e)}")
        return 1
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Collect OpenAI API configuration
    openai_kwargs = {}
    if api_key := os.getenv("OPENAI_API_KEY"):
        openai_kwargs["api_key"] = api_key
    else:
        print("Warning: OPENAI_API_KEY environment variable is not set")
    
    if base_url := os.getenv("OPENAI_BASE_URL"):
        openai_kwargs["base_url"] = base_url
    
    # Select the tools to use
    tools_to_use = [
        "ChestXRayClassifierTool",
        "ChestXRaySegmentationTool",
        "ChestXRayReportGeneratorTool",
        "XRayVQATool",
        "XRayPhraseGroundingTool",
        "ImageVisualizerTool",
        "DicomProcessorTool"
    ]
    
    try:
        # Create workflow
        print(f"Loading controller checkpoint: {args.checkpoint}")
        workflow = await create_workflow(
            checkpoint_path=args.checkpoint,
            tools_to_use=tools_to_use,
            model_name=args.model,
            temperature=args.temperature,
            top_p=args.top_p,
            device=args.device,
            model_dir=args.model_dir,
            temp_dir=args.temp_dir,
            openai_kwargs=openai_kwargs
        )
        
        # Load evaluation data
        print(f"Loading evaluation data: {args.data}")
        evaluation_data = await load_evaluation_data(args.data)
        
        # Run evaluation
        print(f"Starting evaluation of {len(evaluation_data)} examples with concurrency: {args.concurrent}")
        results = await evaluate_dataset(workflow, evaluation_data, args.concurrent)
        
        # Generate report
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = output_dir / f"evaluation_report_{timestamp}.json"
        
        print(f"Generating evaluation report...")
        report = await generate_evaluation_report(results, report_path)
        
        # Compare with baseline
        if args.baseline and os.path.exists(args.baseline):
            print(f"Comparing with baseline: {args.baseline}")
            comparison = await compare_with_baseline(results, args.baseline)
            
            if comparison:
                report["baseline_comparison"] = comparison
                with open(report_path, "w") as f:
                    import json
                    json.dump(report, f, indent=2)
        
        print(f"Evaluation complete! Report saved to: {report_path}")
        return 0
    
    except Exception as e:
        import traceback
        print(f"An error occurred during evaluation: {str(e)}")
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(asyncio.run(main()))