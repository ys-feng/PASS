#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import argparse
import asyncio
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="PASS - Progressive Agent System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train the agent
  python main.py train --data_root data --train_split train.json --epochs 5
  
  # Evaluate the agent
  python main.py evaluate --checkpoint outputs/best_model.pth --eval_split test.json
  
  # Quick evaluation with limited samples
  python main.py evaluate --checkpoint outputs/model.pth --max_samples 100
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')
    
    train_parser = subparsers.add_parser('train', help='Train the agent')
    
    # Data arguments
    train_parser.add_argument("--data_root", type=str, default="data", 
                            help="Root directory of the dataset")
    train_parser.add_argument("--train_split", type=str, default="new_train.json", 
                            help="Training set filename")
    train_parser.add_argument("--val_split", type=str, default="new_validate.json", 
                            help="Validation set filename")
    train_parser.add_argument("--output_dir", type=str, default="./outputs/train_agent", 
                            help="Output directory")
    
    # Training arguments
    train_parser.add_argument("--epochs", type=int, default=1, 
                            help="Number of training epochs")
    train_parser.add_argument("--batch_size", type=int, default=48, 
                            help="Batch size for training")
    train_parser.add_argument("--learning_rate", type=float, default=5e-5, 
                            help="Learning rate")
    train_parser.add_argument("--cost_weight", type=float, default=0.0003, 
                            help="Weight of cost in utility function")
    
    # Warm-up arguments
    train_parser.add_argument("--warmup_size", type=int, default=500, 
                            help="Number of samples for supervised warm-up")
    train_parser.add_argument("--warmup_epochs", type=int, default=3, 
                            help="Number of epochs for supervised warm-up")
    train_parser.add_argument("--disable_warmup", action='store_true', 
                            help="Disable supervised warm-up phase")
    
    # Contrastive Path Ranking arguments
    train_parser.add_argument("--cpr_epochs", type=int, default=2, 
                            help="Number of epochs for Contrastive Path Ranking")
    train_parser.add_argument("--cpr_lr", type=float, default=1e-5, 
                            help="Learning rate for CPR phase")
    
    # Controller arguments
    train_parser.add_argument("--controller_hidden_dim", type=int, default=32, 
                            help="Controller hidden layer dimension")
    train_parser.add_argument("--controller_num_layers", type=int, default=4, 
                            help="Number of controller layers")
    train_parser.add_argument("--encoder_model", type=str, 
                            default="sentence-transformers/all-MiniLM-L6-v2",
                            help="Sentence encoder model")
    
    # Device arguments
    train_parser.add_argument("--device", type=str, default="cuda", 
                            help="Training device (cuda or cpu)")
    train_parser.add_argument("--use_multi_gpu", action='store_true', 
                            help="Use multi-GPU parallel training")
    train_parser.add_argument("--gpu_ids", type=str, default=None, 
                            help="GPU IDs to use, comma-separated (e.g., 0,1,2)")
    train_parser.add_argument("--distributed_tools", action='store_true', 
                            help="Distribute tools to different GPUs")
    train_parser.add_argument("--offload_large_tools", action='store_true', 
                            help="Offload large tools to other GPUs")
    train_parser.add_argument("--large_tool_threshold", type=float, default=4.0, 
                            help="VRAM threshold (GB) for large tools")
    
    # Logging and checkpointing
    train_parser.add_argument("--log_tools", action='store_true', 
                            help="Log tool executions")
    train_parser.add_argument("--log_dir", type=str, default="logs/agent_train", 
                            help="Directory for logs")
    train_parser.add_argument("--checkpoint_to_load", type=str, default=None,
                            help="Path to checkpoint to resume training from")
    train_parser.add_argument("--eval_every_n_epochs", type=int, default=1, 
                            help="Evaluate every N epochs")
    train_parser.add_argument("--save_every_n_epochs", type=float, default=1, 
                            help="Save checkpoint every N epochs")
    
    # Other arguments
    train_parser.add_argument("--max_concurrent_runs", type=int, default=10, 
                            help="Maximum concurrent workflow executions")
    train_parser.add_argument("--llm_eval_model", type=str, default="gpt-4o-mini", 
                            help="LLM model for evaluation")
    train_parser.add_argument("--model_dir", type=str, default="model-weights", 
                            help="Directory for model weights")
    train_parser.add_argument("--temp_dir", type=str, default="temp", 
                            help="Directory for temporary files")
    train_parser.add_argument("--seed", type=int, default=42, 
                            help="Random seed")
    train_parser.add_argument("--use_wandb", action='store_true', 
                            help="Enable Weights & Biases logging")
    train_parser.add_argument("--wandb_project", type=str, default="PASS", 
                            help="W&B project name")
    train_parser.add_argument("--wandb_entity", type=str, default=None, 
                            help="W&B entity name")
    train_parser.add_argument("--wandb_run_name", type=str, default=None, 
                            help="W&B run name")
    train_parser.add_argument("--log_every_n_batches", type=int, default=1, 
                            help="Log metrics every N batches")
    
    #Evaluate subcommand
    eval_parser = subparsers.add_parser('evaluate', help='Evaluate the agent')
    
    # Required arguments
    eval_parser.add_argument("--checkpoint", "-c", type=str, required=True,
                           help="Path to the controller checkpoint file (.pth)")
    
    # Data arguments
    eval_parser.add_argument("--data_root", type=str, default="data", 
                           help="Root directory of the dataset")
    eval_parser.add_argument("--eval_split", type=str, default="new_validate.json", 
                           help="Evaluation set filename")
    eval_parser.add_argument("--output_dir", type=str, default="./outputs/eval_agent", 
                           help="Output directory")
    eval_parser.add_argument("--max_samples", type=int, default=None, 
                           help="Maximum number of samples to evaluate (for testing)")
    
    # Controller arguments
    eval_parser.add_argument("--controller_hidden_dim", type=int, default=32, 
                           help="Controller hidden layer dimension")
    eval_parser.add_argument("--controller_num_layers", type=int, default=4, 
                           help="Number of controller layers")
    eval_parser.add_argument("--encoder_model", type=str, 
                           default="sentence-transformers/all-MiniLM-L6-v2",
                           help="Sentence encoder model")
    
    # Device arguments
    eval_parser.add_argument("--device", type=str, default="cuda", 
                           help="Evaluation device (cuda or cpu)")
    eval_parser.add_argument("--gpu_ids", type=str, default=None, 
                           help="GPU IDs to use, comma-separated")
    eval_parser.add_argument("--distributed_tools", action='store_true', 
                           help="Distribute tools to different GPUs")
    eval_parser.add_argument("--offload_large_tools", action='store_true', 
                           help="Offload large tools to other GPUs")
    eval_parser.add_argument("--large_tool_threshold", type=float, default=4.0, 
                           help="VRAM threshold (GB) for large tools")
    
    # Logging arguments
    eval_parser.add_argument("--log_tools", action='store_true', 
                           help="Log tool executions")
    eval_parser.add_argument("--log_dir", type=str, default="logs/agent_eval", 
                           help="Directory for logs")
    eval_parser.add_argument("--max_concurrent_runs", type=int, default=10, 
                           help="Maximum concurrent workflow executions")
    eval_parser.add_argument("--llm_eval_model", type=str, default="gpt-4o-mini", 
                           help="LLM model for evaluation")
    
    # Other arguments
    eval_parser.add_argument("--model_dir", type=str, default="model-weights", 
                           help="Directory for model weights")
    eval_parser.add_argument("--temp_dir", type=str, default="temp", 
                           help="Directory for temporary files")
    
    # Parse arguments
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        sys.exit(1)
    
    if args.command == 'train':
        print("Starting training...")
        from train_agent import main as train_main
        
        sys.argv = ['train_agent.py']
        for key, value in vars(args).items():
            if key == 'command':
                continue
            if isinstance(value, bool):
                if value:
                    sys.argv.append(f'--{key}')
            elif value is not None:
                sys.argv.extend([f'--{key}', str(value)])
        
        asyncio.run(train_main())
        
    elif args.command == 'evaluate':
        print("Starting evaluation...")
        from evaluate_agent import main as eval_main
        
        sys.argv = ['evaluate_agent.py']
        for key, value in vars(args).items():
            if key == 'command':
                continue
            if isinstance(value, bool):
                if value:
                    sys.argv.append(f'--{key}')
            elif value is not None:
                sys.argv.extend([f'--{key}', str(value)])
        
        asyncio.run(eval_main())

if __name__ == "__main__":
    main()
