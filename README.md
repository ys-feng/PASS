# PASS

## Environment Setup

### Prerequisites

- CUDA-capable GPU (recommended for model training and inference)
- Python 3.10

### Installation Steps

1. **Clone the repository:**
```bash
git clone https://github.com/ys-feng/PASS-clean.git
cd PASS-clean
```

2. **Create conda environment from the provided yml file:**
```bash
conda env create -f environment.yml
```

3. **Activate the environment:**
```bash
conda activate medrax
```

4. **Verify installation:**
```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
```

5. **Set up environment variables:**

Create a `.env` file in the project root or set the following environment variables:

```bash
export OPENAI_API_KEY="your-openai-api-key"
export OPENAI_BASE_URL="https://api.openai.com/v1"  # Optional, if using a proxy
```

For proxy support (if needed):
```bash
export ALL_PROXY="socks5://username:password@host:port"
```

6. **Download model weights (if needed):**

Place model weights in the `model-weights/` directory. The system will automatically download required models on first use.

### Environment Details

For a complete list of dependencies, see `environment.yml`.

## Quick Start

### View Help

```bash
python main.py --help
python main.py train --help
python main.py evaluate --help
```

### Training

**Basic Training:**
```bash
python main.py train
```

**Custom Training Parameters:**
```bash
python main.py train \
  --data_root data \
  --train_split new_train.json \
  --val_split new_validate.json \
  --epochs 5 \
  --batch_size 32 \
  --learning_rate 1e-4 \
  --output_dir ./outputs/my_training
```

**Multi-GPU Training:**
```bash
python main.py train \
  --use_multi_gpu \
  --gpu_ids 0,1,2,3 \
  --distributed_tools
```

**Using Weights & Biases:**
```bash
python main.py train \
  --use_wandb \
  --wandb_project PASS \
  --wandb_run_name my_experiment
```


### Evaluation

**Basic Evaluation:**
```bash
python main.py evaluate \
  --checkpoint /path/to/checkpoint.pth
```

**Full Evaluation:**
```bash
python main.py evaluate \
  --checkpoint outputs/best_model.pth \
  --data_root data \
  --eval_split test.json \
  --output_dir ./outputs/my_evaluation
```

**Quick Test (Limited Samples):**
```bash
python main.py evaluate \
  --checkpoint outputs/model.pth \
  --max_samples 50
```

**Multi-GPU Evaluation:**
```bash
python main.py evaluate \
  --checkpoint outputs/model.pth \
  --gpu_ids 0,1 \
  --distributed_tools
```

## Common Parameters

### Training Parameters

- `--data_root`: Dataset root directory (default: `data`)
- `--train_split`: Training set filename (default: `new_train.json`)
- `--val_split`: Validation set filename (default: `new_validate.json`)
- `--output_dir`: Output directory (default: `./outputs/train_agent`)
- `--epochs`: Number of training epochs (default: `1`)
- `--batch_size`: Batch size (default: `48`)
- `--learning_rate`: Learning rate (default: `5e-5`)
- `--warmup_epochs`: Warmup epochs (default: `3`)
- `--disable_warmup`: Disable warmup phase
- `--use_multi_gpu`: Use multi-GPU training
- `--gpu_ids`: GPU IDs to use, comma-separated
- `--use_wandb`: Enable W&B logging

### Evaluation Parameters

- `--checkpoint`, `-c`: Checkpoint file path (**required**)
- `--data_root`: Dataset root directory (default: `data`)
- `--eval_split`: Evaluation set filename (default: `new_validate.json`)
- `--output_dir`: Output directory (default: `./outputs/eval_agent`)
- `--max_samples`: Maximum number of evaluation samples (for quick testing)
- `--gpu_ids`: GPU IDs to use
- `--distributed_tools`: Distribute tools to different GPUs

## Output Files

### Training Output

- `controller_epoch_N.pth`: Model checkpoint for each epoch
- `best_controller_epoch_N_score_X.pth`: Best model checkpoint
- `controller_epoch_N_history.json`: Training history

### Evaluation Output

- `evaluation_results_TIMESTAMP.json`: Detailed evaluation results
- `evaluation_summary_TIMESTAMP.csv`: Evaluation results CSV summary

## Project Structure

```
PASS-clean/
├── core/                       # Core agent components
│   ├── agent_controller.py     # Neural controller
│   ├── agent_workflow.py       # Workflow engine
│   ├── agent_optimizer.py      # Training optimizer
│   ├── agent_operators.py      # Operator management
│   └── agent_utils.py          # Utility functions
├── tools/                      # Medical imaging tools
│   ├── classification.py       # X-ray classification
│   ├── segmentation.py         # Image segmentation
│   ├── report_generation.py   # Report generation
│   ├── xray_vqa.py            # Visual Q&A
│   └── llava_med.py           # LLaVA-Med integration
├── main.py                     # Main entry point
├── train_agent.py             # Training script
├── evaluate_agent.py          # Evaluation script
├── environment.yml            # Conda environment
└── README.md                  # This file
```
