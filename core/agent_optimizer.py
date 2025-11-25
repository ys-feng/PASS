import os
import json
import torch
import logging
import math
from typing import List, Dict, Any, Optional
from datetime import datetime
from pathlib import Path

# Import calculate_score and calculate_cost
from .agent_utils import calculate_score, calculate_cost 

logger = logging.getLogger(__name__)

class AgentOptimizer:
    """Agent optimizer for training the controller network."""
    
    def __init__(
        self,
        controller: torch.nn.Module,
        operator_embeddings: torch.Tensor,
        operator_names: List[str],
        learning_rate: float = 0.01,
        batch_size: int = 4,
        log_dir: str = "logs",
        cost_weight: float = 0, # Add cost weight parameter
        device = None,
        lr_decay_factor: float = 0.95,
        lr_decay_every_n_batches: int = 10
    ):
        """
        Initializes the AgentOptimizer.
        
        Args:
            controller: The controller network.
            operator_embeddings: The operator embedding matrix.
            operator_names: A list of operator names.
            learning_rate: The learning rate.
            batch_size: The batch size.
            log_dir: The log directory.
            cost_weight: The weight of the cost in the utility function.
            device: The computation device.
        """
        self.controller = controller
        self.operator_embeddings = operator_embeddings
        self.operator_names = operator_names
        self.batch_size = batch_size
        self.log_dir = Path(log_dir)
        self.cost_weight = cost_weight # Save the cost weight
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.lr_decay_factor = lr_decay_factor
        self.lr_decay_every_n_batches = lr_decay_every_n_batches
        self.batch_count = 0  # Track the number of processed batches
        self.initial_lr = learning_rate  # Save the initial learning rate
        self.learning_rate = learning_rate  # Add this line for convenient access later
        self.warmup_optimizer = torch.optim.Adam(
            self.controller.parameters(), 
            lr=self.learning_rate
        )
        
        # Initialize the Adam optimizer
        self.optimizer = torch.optim.Adam(self.controller.parameters(), lr=learning_rate)
        
        # Ensure the log directory exists
        self.log_dir.mkdir(exist_ok=True, parents=True)
        
        # Record training history
        self.history = {
            "losses": [],
            "scores": [],
            "costs": [],
            "utilities": [] # Add utility record
        }
        
        logger.info(f"AgentOptimizer initialized, learning_rate: {learning_rate}, batch_size: {batch_size}, cost_weight: {cost_weight}")

    def adjust_learning_rate_if_needed(self, current_batch_count):
        """Adjusts the learning rate if needed based on the current batch count."""
        if self.lr_decay_every_n_batches > 0 and current_batch_count % self.lr_decay_every_n_batches == 0:
            current_lr = self.adjust_learning_rate()
            # Record learning rate
            if 'learning_rates' not in self.history:
                self.history['learning_rates'] = []
            self.history['learning_rates'].append(current_lr)
            return current_lr
        return None

    def adjust_learning_rate(self):
        """Decays the learning rate."""
        # Apply decay factor to each parameter group in the optimizer
        for param_group in self.optimizer.param_groups:
            param_group['lr'] *= self.lr_decay_factor
        
        current_lr = self.optimizer.param_groups[0]['lr']
        logger.info(f"Learning rate adjusted to: {current_lr:.6f}")
        return current_lr

    def update(self, log_probs_batch: List[torch.Tensor], scores_batch: List[float], costs_batch: List[float]) -> Optional[float]:
        """
        Updates the controller parameters.
        
        Args:
            log_probs_batch: Log probabilities for each sample in the batch (Tensors).
            scores_batch: Scores for each sample in the batch.
            costs_batch: Costs for each sample in the batch.
            
        Returns:
            Optional[float]: The loss value, or None if the update failed.
        """
        if len(log_probs_batch) == 0:
            logger.warning("Empty batch, skipping update.")
            return None
            
        try:
            # Ensure all inputs are valid tensors or floats
            valid_log_probs = []
            valid_scores = []
            valid_costs = []
            
            for i, (log_prob, score, cost) in enumerate(zip(log_probs_batch, scores_batch, costs_batch)):
                # Check if log_prob is valid
                if not isinstance(log_prob, torch.Tensor) or torch.isnan(log_prob).any() or torch.isinf(log_prob).any():
                    logger.warning(f"Log probability for sample {i} is invalid, skipping: {log_prob}")
                    continue
                    
                # Check if score and cost are valid
                if not isinstance(score, (int, float)) or not isinstance(cost, (int, float)) or math.isnan(score) or math.isnan(cost) or math.isinf(score) or math.isinf(cost):
                    logger.warning(f"Score or cost for sample {i} is invalid, skipping: score={score}, cost={cost}")
                    continue
                
                # Ensure log_prob is on the correct device and requires grad
                valid_log_probs.append(log_prob.to(self.device).requires_grad_(True))
                valid_scores.append(score)
                valid_costs.append(cost)
            
            # If there are no valid samples, return None
            if not valid_log_probs:
                logger.warning("No valid samples in the batch, skipping update.")
                return None
                
            # Stack batch log probabilities
            # Since log_probs may come from architectures with different numbers of layers, direct stacking might cause shape mismatch.
            # Usually, the RL loss is for the log probability of the entire trajectory. We assume here that the cumulative log probability for each sample is passed.
            # If the log probability for each layer is passed, they need to be summed first.
            # Assuming valid_log_probs already contains the cumulative log probability for each sample (scalar Tensor).
            try:
                 log_probs = torch.stack(valid_log_probs)
            except RuntimeError as e:
                 logger.error(f"Error while stacking log probabilities (possible shape mismatch): {e}. Log Probs: {valid_log_probs}")
                 # Attempting to handle, e.g., by taking only the first element (if all are scalars).
                 try:
                     log_probs = torch.tensor([lp.item() for lp in valid_log_probs], device=self.device, requires_grad=True)
                 except Exception as inner_e:
                     logger.error(f"Could not convert log probabilities to a scalar tensor: {inner_e}")
                     return None

            scores = torch.tensor(valid_scores, dtype=torch.float32, device=self.device)
            costs = torch.tensor(valid_costs, dtype=torch.float32, device=self.device)
            
            # Calculate utility (score minus weighted cost)
            utilities = scores - self.cost_weight * costs
            # Detach utility to make it a constant that does not participate in gradient calculation (as required by the REINFORCE algorithm).
            utilities = utilities.detach()
            
            # Check for NaN or Inf
            if torch.isnan(utilities).any() or torch.isinf(utilities).any():
                logger.error(f"Utility calculation resulted in NaN or Inf values, skipping update. Utilities: {utilities}")
                return None
            if torch.isnan(log_probs).any() or torch.isinf(log_probs).any():
                logger.error(f"Log probabilities contain NaN or Inf values, skipping update. Log Probs: {log_probs}")
                return None
            
            # Calculate loss (mean of the negative product of log probabilities and utility) - REINFORCE loss.
            # loss = -(log_probs * utilities).mean()
            utilities_mean = utilities - utilities.mean()
            loss = -(log_probs * utilities_mean).mean()
            
            # Check if the loss is valid
            if torch.isnan(loss) or torch.isinf(loss):
                logger.error(f"Invalid loss calculation result: {loss.item()}")
                return None
            
            # Gradient descent
            self.optimizer.zero_grad()
            loss.backward()
            
            # Check if gradients are valid
            nan_inf_grad = False
            for name, param in self.controller.named_parameters():
                if param.grad is not None:
                    if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                        logger.error(f"Gradient for parameter {name} contains NaN or Inf values: {param.grad}")
                        nan_inf_grad = True
            
            if nan_inf_grad:
                 logger.error("Gradient contains NaN or Inf, skipping update.")
                 # Clear invalid gradients
                 self.optimizer.zero_grad()
                 return None
            
            # Clip gradients to prevent explosion
            torch.nn.utils.clip_grad_norm_(self.controller.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # Record training history
            loss_value = loss.item()
            current_scores = scores.mean().item()
            current_costs = costs.mean().item()
            current_utilities = utilities.mean().item()
            
            self.history["losses"].append(loss_value)
            self.history["scores"].append(current_scores)
            self.history["costs"].append(current_costs)
            self.history["utilities"].append(current_utilities)
            
            logger.info(f"Updating controller, loss: {loss_value:.4f}, avg score: {current_scores:.4f}, avg cost: {current_costs:.4f}, avg utility: {current_utilities:.4f}")
    
            return loss_value
            
        except Exception as e:
            logger.error(f"An error occurred while updating the controller: {str(e)}", exc_info=True)
            return None
        
    def save_checkpoint(self, name: str = "agent_controller") -> str:
        """
        Saves the controller parameters.
        
        Args:
            name: The checkpoint name prefix.
            
        Returns:
            str: The path to the saved file.
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = self.log_dir / f"{name}_{timestamp}.pth"
        
        try:
            torch.save({
                'controller_state_dict': self.controller.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'operator_names': self.operator_names,
                'history': self.history,
                'timestamp': timestamp,
                'cost_weight': self.cost_weight # Save the cost weight
            }, save_path)
            
            # Also save the training history as JSON
            history_path = self.log_dir / f"{name}_history_{timestamp}.json"
            with open(history_path, 'w') as f:
                # Convert to a serializable format
                serializable_history = {k: [float(v) for v in values] for k, values in self.history.items()}
                json.dump(serializable_history, f, indent=2)
            
            logger.info(f"Saved controller checkpoint to {save_path}")
            return str(save_path)
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}", exc_info=True)
            return ""
        
    def load_checkpoint(self, checkpoint_path: str) -> None:
        """
        Loads the controller parameters.
        
        Args:
            checkpoint_path: The path to the checkpoint file.
        """
        if not Path(checkpoint_path).exists():
            logger.error(f"Checkpoint file not found: {checkpoint_path}")
            return
            
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            
            self.controller.load_state_dict(checkpoint['controller_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            # Restore training history
            if 'history' in checkpoint:
                self.history = checkpoint['history']
                # Ensure all lists are lists of numbers
                for key in self.history:
                    if isinstance(self.history[key], list):
                       self.history[key] = [v for v in self.history[key] if isinstance(v, (int, float))] 
            
            # Load cost weight (if it exists)
            if 'cost_weight' in checkpoint:
                self.cost_weight = checkpoint['cost_weight']
                logger.info(f"Loaded cost weight from checkpoint: {self.cost_weight}")
            
            # Verify that operator names match
            loaded_operator_names = checkpoint.get('operator_names', [])
            if set(loaded_operator_names) != set(self.operator_names):
                logger.warning(f"Loaded operator names do not match current ones! Loaded: {loaded_operator_names}, Current: {self.operator_names}")
                
            logger.info(f"Loaded controller from checkpoint {checkpoint_path}")
        except Exception as e:
             logger.error(f"Failed to load checkpoint {checkpoint_path}: {e}", exc_info=True)
    
    def get_training_stats(self) -> Dict[str, Any]:
        """
        Gets training statistics.
        
        Returns:
            Dict: The training statistics.
        """
        batches = len(self.history["losses"])
        
        if batches == 0:
            return {"batches": 0, "message": "No training data yet."}
        
        return {
            "batches": batches,
            "current_loss": self.history["losses"][-1] if self.history["losses"] else None,
            "current_score": self.history["scores"][-1] if self.history["scores"] else None,
            "current_cost": self.history["costs"][-1] if self.history["costs"] else None,
            "current_utility": self.history["utilities"][-1] if self.history["utilities"] else None,
            "min_loss": min(self.history["losses"]) if self.history["losses"] else None,
            "max_score": max(self.history["scores"]) if self.history["scores"] else None,
            "min_cost": min(self.history["costs"]) if self.history["costs"] else None,
            "max_utility": max(self.history["utilities"]) if self.history["utilities"] else None,
            "avg_loss": sum(self.history["losses"]) / batches if batches else 0,
            "avg_score": sum(self.history["scores"]) / batches if batches else 0,
            "avg_cost": sum(self.history["costs"]) / batches if batches else 0,
            "avg_utility": sum(self.history["utilities"]) / batches if batches else 0,
        }
    
    def apply_text_gradient(self, prompt_template: str, improvement: str) -> str:
        """
        Applies text gradient to improve the prompt template (currently a placeholder).
        
        Args:
            prompt_template: The original prompt template.
            improvement: The suggested improvement.
            
        Returns:
            str: The improved prompt template.
        """
        logger.warning("apply_text_gradient is currently a placeholder and does not actually implement text gradient optimization.")
        
        # In a real application, an LLM could be used to generate an improved prompt template.
        # This is a simple example that directly replaces with the passed improvement.
        improved_template = f"{prompt_template}\n\n# Text Gradient Improvement Suggestion:\n{improvement}"
        
        return improved_template
