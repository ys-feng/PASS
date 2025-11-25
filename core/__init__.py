"""
Core Agent Module

This module provides the core agent functionality including:
- MultiLayerController: Neural controller for operator selection
- AgentWorkflow: Workflow execution engine
- AgentOptimizer: Training and optimization
- Operator classes and utilities

The agent system supports multi-layer operator architectures for
complex task execution and optimization.
"""

from .agent_controller import (
    SentenceEncoder,
    OperatorSelector,
    MultiLayerController
)

from .agent_workflow import (
    AgentWorkflow,
    CustomJSONEncoder
)

from .agent_optimizer import (
    AgentOptimizer
)

from .agent_operators import (
    Operator,
    ChestXRayClassify,
    ChestXRaySegment,
    ChestXRayReport,
    VQAnalyze,
    GroundFindings,
    LlaVAMed,
    EarlyStop,
    build_operator_mapping,
    get_operator_descriptions,
    get_operator_cost_models,
    calculate_dynamic_cost
)

from .agent_utils import (
    get_sentence_embedding,
    get_operator_embeddings,
    calculate_score,
    calculate_cost,
    get_text_similarity,
    sample_operators
)

__all__ = [
    # Controller
    'SentenceEncoder',
    'OperatorSelector',
    'MultiLayerController',
    
    # Workflow
    'AgentWorkflow',
    'CustomJSONEncoder',
    
    # Optimizer
    'AgentOptimizer',
    
    # Operators
    'Operator',
    'ChestXRayClassify',
    'ChestXRaySegment',
    'ChestXRayReport',
    'VQAnalyze',
    'GroundFindings',
    'LlaVAMed',
    'EarlyStop',
    'build_operator_mapping',
    'get_operator_descriptions',
    'get_operator_cost_models',
    'calculate_dynamic_cost',
    
    # Utils
    'get_sentence_embedding',
    'get_operator_embeddings',
    'calculate_score',
    'calculate_cost',
    'get_text_similarity',
    'sample_operators',
]

__version__ = '1.0.0'
