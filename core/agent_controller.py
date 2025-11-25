import torch
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Tuple, Any, Optional
import logging

logger = logging.getLogger(__name__)

class SentenceEncoder:
    """A sentence encoder for encoding text into embedding vectors."""
    
    def __init__(self, model_name: str = 'sentence-transformers/all-MiniLM-L6-v2', device=None):
        """
        Initializes the SentenceEncoder.
        
        Args:
            model_name: The name of the pretrained model to use.
            device: The computation device.
        """
        self.model = SentenceTransformer(model_name)
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # Ensure the model is on the correct device.
        try:
            self.model = self.model.to(self.device)
        except Exception as e:
            logger.warning(f"Could not move SentenceTransformer to device {self.device}: {e}")
            # Fall back to CPU.
            self.device = torch.device('cpu')
            self.model = self.model.to(self.device)
         
    def encode(self, sentence: str) -> torch.Tensor:
        """
        Encodes a sentence into an embedding vector.
         
        Args:
            sentence: The input sentence.
             
        Returns:
            torch.Tensor: The sentence embedding vector.
        """
        # Encode using CPU to avoid device conflicts.
        with torch.no_grad():
            embeddings = self.model.encode(sentence, convert_to_tensor=False)
            # Ensure the returned tensor is on the correct device.
            if isinstance(embeddings, torch.Tensor):
                return embeddings.to(self.device)
            else:
                return torch.tensor(embeddings, device=self.device, dtype=torch.float32)
         
    def batch_encode(self, sentences: List[str]) -> torch.Tensor:
        """
        Encodes sentences in a batch.
         
        Args:
            sentences: A list of input sentences.
             
        Returns:
            torch.Tensor: An embedding matrix with shape [batch_size, embedding_dim].
        """
        with torch.no_grad():
            embeddings = self.model.encode(sentences, convert_to_tensor=False)
            if isinstance(embeddings, torch.Tensor):
                return embeddings.to(self.device)
            else:
                return torch.tensor(embeddings, device=self.device, dtype=torch.float32)
     
    def to(self, device):
        """Moves the model to a specified device."""
        self.device = device
        try:
            self.model = self.model.to(device)
        except Exception as e:
            logger.warning(f"Could not move SentenceTransformer to device {device}: {e}")
        return self

class OperatorSelector(torch.nn.Module):
    """A single-layer operator selector."""
     
    def __init__(self, input_dim: int = 384, hidden_dim: int = 32, device=None, is_first_layer: bool = False):
        """
        Initializes the OperatorSelector.
         
        Args:
            input_dim: The input embedding dimension.
            hidden_dim: The hidden layer dimension.
            device: The computation device.
            is_first_layer: Whether this is the first layer selector.
        """
        super().__init__()
        self.is_first_layer = is_first_layer
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
         
        # Use different encoders for the first layer and subsequent layers.
        if self.is_first_layer:
            self.operator_encoder = torch.nn.Linear(input_dim, hidden_dim).to(self.device)
        else:
            self.operator_encoder = torch.nn.Linear(input_dim * 2, hidden_dim).to(self.device)
             
        self.query_encoder = torch.nn.Linear(input_dim, hidden_dim).to(self.device)
        self.activation = torch.nn.Tanh().to(self.device)
         
    def forward(self, query_embedding: torch.Tensor, operator_embeddings: torch.Tensor, prev_op_embedding: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass, calculates the matching score between the query and operators.
         
        Args:
            query_embedding: The query embedding, shape [embedding_dim].
            operator_embeddings: The operator embedding matrix, shape [num_operators, embedding_dim].
            prev_op_embedding: The previous layer's operator embedding, shape [embedding_dim].
             
        Returns:
            torch.Tensor: The operator probability distribution, shape [num_operators].
        """
        # Ensure inputs are on the correct device.
        query_embedding = query_embedding.to(self.device)
        operator_embeddings = operator_embeddings.to(self.device)
        if prev_op_embedding is not None:
            prev_op_embedding = prev_op_embedding.to(self.device)
         
        # Encode the query.
        query_proj = self.activation(self.query_encoder(query_embedding))
         
        # Process the operator embeddings.
        if self.is_first_layer or prev_op_embedding is None:
            # The first layer uses the operator embeddings directly.
            operator_proj = self.activation(self.operator_encoder(operator_embeddings))
        else:
            # Other layers consider the previous layer's operator embedding.
            # Expand and concatenate the previous operator embedding to each operator embedding.
            batch_size = operator_embeddings.size(0)
            prev_op_expanded = prev_op_embedding.unsqueeze(0).expand(batch_size, -1)
            combined = torch.cat([operator_embeddings, prev_op_expanded], dim=1)
            operator_proj = self.activation(self.operator_encoder(combined))
             
        # Calculate dot product similarity.
        similarity = torch.matmul(query_proj, operator_proj.t())
         
        # Convert to a probability distribution.
        probs = torch.softmax(similarity, dim=0)
        return probs

class MultiLayerController(torch.nn.Module):
    """A multi-layer controller network for dynamically selecting operator architectures."""
     
    def __init__(self, input_dim: int = 384, hidden_dim: int = 32, num_layers: int = 4, device=None):
        """
        Initializes the MultiLayerController.
         
        Args:
            input_dim: The input embedding dimension.
            hidden_dim: The hidden layer dimension.
            num_layers: The number of controller layers.
            device: The computation device.
        """
        super().__init__()
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_layers = num_layers
         
        # Create multiple layer selectors.
        self.layers = torch.nn.ModuleList([
            OperatorSelector(input_dim, hidden_dim, self.device, is_first_layer=(i == 0)) 
            for i in range(num_layers)
        ])
         
        # Lazy initialize text_encoder to avoid DataParallel issues.
        self._text_encoder = None
        self._text_encoder_device = self.device
         
    @property
    def text_encoder(self):
        """Lazy initialized text encoder."""
        if self._text_encoder is None:
            # Get the current device (it might be changed by DataParallel).
            current_device = next(self.parameters()).device if list(self.parameters()) else self.device
            self._text_encoder = SentenceEncoder(device=current_device)
            self._text_encoder_device = current_device
        elif self._text_encoder_device != (next(self.parameters()).device if list(self.parameters()) else self.device):
            # If the device has changed, re-initialize.
            current_device = next(self.parameters()).device if list(self.parameters()) else self.device
            self._text_encoder = self._text_encoder.to(current_device)
            self._text_encoder_device = current_device
        return self._text_encoder
     
    def to(self, device):
        """Moves the model to a specified device."""
        result = super().to(device)
        result.device = device
        result._text_encoder_device = device
        if result._text_encoder is not None:
            result._text_encoder = result._text_encoder.to(device)
        return result

    def predict_operator_sequence(
        self,
        query: str,
        operator_embeddings: torch.Tensor,
        operator_names: List[str],
        target_sequence: List[str]
    ) -> List[torch.Tensor]:
        """
        Used for supervised warm-up, predicts multiple operators sequentially (no sampling).
         
        Args:
            query: The input question.
            operator_embeddings: Embeddings for all operators.
            operator_names: A list of operator names.
            target_sequence: The ground truth sequence of operator names.
         
        Returns:
            List[torch.Tensor]: A list of softmax distributions (logits) for each layer.
        """
        # Ensure operator embeddings are on the correct device.
        operator_embeddings = operator_embeddings.to(self.device)
         
        # Encode the query.
        query_embedding = self.text_encoder.encode(query).to(self.device)
        prev_op_embedding = None
        probs_list = []

        logger.info(f"Predicting operator sequence: {target_sequence}")

        for i, target_op in enumerate(target_sequence):
            if i >= self.num_layers:
                logger.info(f"Reached max layers {self.num_layers}, stopping prediction.")
                break # Exceeded max number of layers.
                 
            layer = self.layers[i]
            if layer.is_first_layer:
                current_prev_op_embedding = None
            else:
                current_prev_op_embedding = prev_op_embedding
                 
            logger.debug(f"[predict_operator_sequence] Layer {i} | is_first_layer: {layer.is_first_layer} | prev_op_embedding: {prev_op_embedding.shape if prev_op_embedding is not None else None}")
             
            try:
                probs = layer(query_embedding, operator_embeddings, current_prev_op_embedding)
                probs_list.append(probs)
                 
                # Update prev_op_embedding to the embedding of the current target operator.
                if target_op in operator_names:
                    target_idx = operator_names.index(target_op)
                    prev_op_embedding = operator_embeddings[target_idx]
                    logger.debug(f"Layer {i}: Using embedding of target operator '{target_op}'")
                else:
                    logger.warning(f"Layer {i}: Target operator '{target_op}' not in the known operator list.")
                    # If the target operator is not found, use the one with the highest probability as a substitute.
                    target_idx = torch.argmax(probs).item()
                    prev_op_embedding = operator_embeddings[target_idx]
                    logger.debug(f"Layer {i}: Using embedding of the highest probability operator '{operator_names[target_idx]}' as a substitute.")
            except Exception as e:
                logger.error(f"Error in layer {i} prediction: {str(e)}")
                if i > 0:
                    # If it's not the first layer that erred, return the existing results.
                    return probs_list
                else:
                    # If the first layer erred, return an empty list.
                    return []

        return probs_list
         
    def forward(self, query: str, operator_embeddings: torch.Tensor, operator_names: List[str], threshold: float = 0.25) -> Tuple[List[torch.Tensor], List[List[str]]]:
        """
        Forward pass, samples a multi-layer operator architecture.
         
        Args:
            query: The user query text.
            operator_embeddings: The operator embedding matrix, shape [num_operators, embedding_dim].
            operator_names: A list of operator names.
            threshold: The cumulative probability threshold for sampling.
             
        Returns:
            Tuple[List[torch.Tensor], List[List[str]]]: 
                - A list of log probabilities for each layer.
                - A list of selected operator names for each layer.
        """
        # Ensure operator embeddings are on the correct device.
        operator_embeddings = operator_embeddings.to(self.device)
         
        # Encode the query.
        query_embedding = self.text_encoder.encode(query).to(self.device)
         
        # Store the log probabilities and selected operators for each layer.
        log_probs_layers = []
        selected_names_layers = []
        prev_op_embedding = None
         
        # Select operators layer by layer.
        for i, layer in enumerate(self.layers):
            # Get the operator probability distribution for the current layer.
            probs = layer(query_embedding, operator_embeddings, prev_op_embedding)
             
            # Sample operators.
            selected_indices = self._sample_operators(probs, threshold)
             
            # If it's the first layer, ensure at least one core tool is included.
            if i == 0 and len(selected_indices) == 0:
                # Find the classification or segmentation tool with the highest probability.
                core_tool_indices = []
                for name in ["ChestXRayClassify", "ChestXRayReport"]:
                    if name in operator_names:
                        idx = operator_names.index(name)
                        core_tool_indices.append(idx)
                 
                if core_tool_indices:
                    # Select the core tool with the highest probability.
                    core_probs = probs[torch.tensor(core_tool_indices, device=self.device)]
                    max_idx = core_tool_indices[core_probs.argmax().item()]
                    selected_indices = torch.tensor([max_idx], device=self.device)
             
            # Collect the names of the selected operators.
            try:
                selected_names = [operator_names[idx.item()] for idx in selected_indices]
            except Exception as e:
                logger.error(f"Index error: {str(e)}, selected_indices: {selected_indices}, device: {selected_indices.device}")
                selected_names = []
             
            # If an early stop operator is included, do not add subsequent layers.
            if "EarlyStop" in selected_names:
                selected_names_layers.append(selected_names)
                try:
                    log_probs_layers.append(torch.log(probs[selected_indices]).sum())
                except Exception as e:
                    logger.error(f"Error calculating log probability: {str(e)}")
                    log_probs_layers.append(torch.tensor(0.0, device=self.device))
                logger.info(f"Layer {i}: Early stop selected, skipping subsequent layers.")
                break
                 
            # Store the results of the current layer.
            selected_names_layers.append(selected_names)
            try:
                log_probs_layers.append(torch.log(probs[selected_indices]).sum())
            except Exception as e:
                logger.error(f"Error calculating log probability: {str(e)}")
                log_probs_layers.append(torch.tensor(0.0, device=self.device))
             
            # If no operator was selected, end early.
            if len(selected_indices) == 0:
                logger.info(f"Layer {i}: No operator selected, ending early.")
                break
                 
            # Update the previous layer's operator embedding (using the mean of the selected operator embeddings).
            try:
                prev_op_embedding = operator_embeddings[selected_indices].mean(dim=0)
            except Exception as e:
                logger.error(f"Error calculating mean of operator embeddings: {str(e)}")
                prev_op_embedding = operator_embeddings[0] if operator_embeddings.size(0) > 0 else None
             
        # Output the sampled architecture information.
        logger.info(f"Sampled architecture: {selected_names_layers}")
         
        return log_probs_layers, selected_names_layers
     
    def _sample_operators(self, probs: torch.Tensor, threshold: float = 0.25) -> torch.Tensor:
        """
        Sample operators based on a probability distribution until a cumulative threshold is reached.
         
        Args:
            probs: The operator probability distribution.
            threshold: The cumulative probability threshold.
             
        Returns:
            torch.Tensor: Indices of the selected operators.
        """
        device = probs.device
        probs = probs.detach()  # Detach from gradient computation to avoid affecting backpropagation.
         
        num_ops = probs.size(0)
        if num_ops == 0:
            return torch.tensor([], dtype=torch.long, device=device)
             
        # Initialize lists for selected and remaining operators.
        selected = torch.tensor([], dtype=torch.long, device=device)
        cumulative = 0.0
        remaining = torch.arange(num_ops, device=device)
         
        # Check if all probabilities are zero.
        if probs.sum() == 0:
            # If all probabilities are zero, select the first operator and return.
            return torch.tensor([0], dtype=torch.long, device=device)
         
        # Iteratively sample until the cumulative probability reaches the threshold or no operators are left.
        max_iterations = num_ops  # Prevent infinite loops.
        iteration = 0
         
        while cumulative < threshold and remaining.numel() > 0 and iteration < max_iterations:
            iteration += 1
            # Sample from the remaining operators.
            remaining_probs = probs[remaining]
             
            # If all remaining probabilities are zero, break the loop.
            if remaining_probs.sum() == 0:
                break
             
            # Ensure remaining_probs is on the GPU.
            if remaining_probs.device != device:
                remaining_probs = remaining_probs.to(device)
                 
            try:
                sampled = torch.multinomial(remaining_probs, num_samples=1)
            except Exception as e:
                logger.error(f"Sampling error: {str(e)}, remaining_probs: {remaining_probs}, device: {remaining_probs.device}, probs device: {probs.device}")
                break
                 
            idx = remaining[sampled].squeeze()
             
            # Ensure no duplicates are selected.
            if selected.numel() == 0:
                selected = idx.unsqueeze(0) if idx.dim() == 0 else idx
                cumulative += probs[idx].item()
            elif not torch.any(selected == idx):
                selected = torch.cat([selected, idx.unsqueeze(0) if idx.dim() == 0 else idx])
                cumulative += probs[idx].item()
             
            # Remove the sampled operator from the remaining list.
            mask = torch.ones_like(remaining, dtype=torch.bool)
            mask[sampled] = False
            remaining = remaining[mask]
         
        # If no operator is selected, choose the one with the highest probability.
        if selected.numel() == 0 and num_ops > 0:
            top_idx = torch.argmax(probs).unsqueeze(0)
            return top_idx
             
        return selected.to(device)  # Ensure the result is on the correct device.
     

    def get_path_log_prob(
        self,
        query: str,
        operator_embeddings: torch.Tensor,
        operator_names: List[str],
        path: List[str]
    ) -> torch.Tensor:
        """
        Calculates the total log probability of a given path.
        """
        operator_embeddings = operator_embeddings.to(self.device)
        query_embedding = self.text_encoder.encode(query).to(self.device)
         
        total_log_prob = torch.tensor(0.0, device=self.device)
        prev_op_embedding = None

        for i, op_name in enumerate(path):
            if i >= self.num_layers:
                break
             
            layer = self.layers[i]
             
            # Get the probability distribution for the current layer.
            probs = layer(query_embedding, operator_embeddings, prev_op_embedding)
             
            # Find the index of the current step's operator.
            if op_name not in operator_names:
                # If the operator in the path does not exist, return a very low log probability.
                return torch.tensor(-1e9, device=self.device)
                 
            op_idx = operator_names.index(op_name)
             
            # Accumulate the log probability of this step.
            # Add a small epsilon to prevent log(0).
            total_log_prob = total_log_prob + torch.log(probs[op_idx] + 1e-9)
             
            # Update the previous operator's embedding.
            prev_op_embedding = operator_embeddings[op_idx]
             
        return total_log_prob
