# ============================================================================
# ENTROPHY © 2025 by Workfabric
# Licensed under Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International
# https://creativecommons.org/licenses/by-nc-sa/4.0/
# ============================================================================

import os
import json
import argparse
import logging
import numpy as np
from sklearn.metrics import precision_recall_fscore_support
import matplotlib.pyplot as plt
import re
from typing import List, Dict, Tuple, Any, Optional
from data_processor import WorkflowDataProcessor
import tqdm
import datetime
import time
import yaml

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("workflow_segmentation")

# ================================
# PROMPTS AND DEFINITIONS
# ================================

SEGMENTATION_PROMPT = """\
Your task is to precisely segment a sequence of user interactions that come from MULTIPLE WORKFLOWS concatenated together.

Here are the workflow process definitions you should consider:
{process_definitions}

Your task:
1. Analyze the entire sequence carefully
2. Identify where one workflow ends and another begins
3. Mark the exact positions (indices) of these boundaries

User interaction sequence (0-indexed):
{sequence}

Step-by-step approach:
1. First, review the entire sequence to understand the overall pattern
2. Look for clear transitions between different workflows
3. Pay attention to workflow beginning and completion signals
4. When identifying a boundary, note its exact index position
5. Ensure all segments together cover the complete sequence

Format requirements:
- Provide a JSON array where each workflow segment has:
  - "start_index": starting position (0-indexed)
  - "end_index": ending position (inclusive, 0-indexed)
- First segment should always have start_index = 0
- Each segment's end_index should be exactly one less than the next segment's start_index
- Last segment's end_index should be the last index in the sequence
- The answer must be enclosed in <answer> tags

Example:
For a sequence with 3 workflows, a valid response might be:
<answer>
[
  {{"start_index": 0, "end_index": 5}},
  {{"start_index": 6, "end_index": 12}},
  {{"start_index": 13, "end_index": 17}}
]
</answer>

Before finalizing your answer:
- Verify that your segments correctly capture workflow transitions
- Check that all indices are within the sequence bounds
- Confirm that segments are contiguous (no gaps or overlaps)
"""

SYSTEM_PROMPT = "You are a workflow segmentation assistant that analyzes sequences of user interactions and identifies where different workflows begin and end."

# ================================
# CONFIGURATION
# ================================

def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from a YAML file.
    
    Args:
        config_path: Path to the YAML configuration file
        
    Returns:
        Dictionary containing configuration parameters
    """
    with open(config_path, 'r') as f:
        if config_path.endswith('.yaml') or config_path.endswith('.yml'):
            config = yaml.safe_load(f)
        else:
            raise ValueError(f"Unsupported configuration file format: {config_path} (must be .yaml or .yml)")
    logger.info(f"Loaded configuration from {config_path}")
    return config

def parse_args():
    """
    Parse command line arguments.
    """
    parser = argparse.ArgumentParser(description="Workflow Segmentation")
    
    # Config file argument
    parser.add_argument("--config", type=str, help="Path to YAML configuration file")
    
    # Data arguments
    parser.add_argument("--data_path", type=str, help="Path to input JSON data file")
    parser.add_argument("--output_dir", type=str, help="Directory for saving outputs")
    
    # Model arguments
    parser.add_argument("--model_name", type=str, help="Foundation model name")
    parser.add_argument("--provider", type=str, help="Provider for workflow segmentation")
    parser.add_argument("--api_key", type=str, help="API key for workflow segmentation")
    parser.add_argument("--api_base", type=str, help="API base for workflow segmentation")
    parser.add_argument("--max_tokens", type=int, help="Maximum tokens for workflow segmentation")
    parser.add_argument("--temperature", type=float, help="Temperature for workflow segmentation")
    parser.add_argument("--seed", type=int, help="Random seed")
    parser.add_argument("--gpus", type=str, help="Comma-separated list of GPU IDs to use (e.g., '0,1,2')")
    parser.add_argument("--batch_size", type=int, help="Batch size for segmentation")
    parser.add_argument("--wait_time", type=float, help="Wait time between segmentations to avoid rate limiting")
    
    # Segmentation specific arguments
    parser.add_argument("--min_workflows", type=int, help="Minimum number of workflows to concatenate")
    parser.add_argument("--max_workflows", type=int, help="Maximum number of workflows to concatenate")
    parser.add_argument("--num_samples", type=int, help="Number of samples to create")
    parser.add_argument("--tolerance", type=int, help="Tolerance for boundary detection")
    args = parser.parse_args()
    
    # If a config file is provided, load it and update args
    if args.config:
        config = load_config(args.config)
        # Update args with values from config file
        for key, value in config.items():
            if key != 'config' and not getattr(args, key, None):  # Don't override explicitly passed args
                setattr(args, key, value)
    
    # Check required arguments after merging config
    required_args = ['data_path', 'model_name', 'provider']
    missing_args = [arg for arg in required_args if not getattr(args, arg, None)]
    if missing_args:
        parser.error(f"The following required arguments are missing: {', '.join(missing_args)}")
    
    # Set defaults for optional arguments if not provided
    if not args.output_dir:
        args.output_dir = "./outputs/segmentation"
    if not args.max_tokens:
        args.max_tokens = 2000  # Larger for segmentation task
    if not args.temperature:
        args.temperature = 0.6  # Lower for more deterministic results
    if not args.wait_time:
        args.wait_time = 0.0
    if not args.seed:
        args.seed = 2404
    if not args.min_workflows:
        args.min_workflows = 2
    if not args.max_workflows:
        args.max_workflows = 4
    if not args.num_samples:
        args.num_samples = 100
    if not args.tolerance:
        args.tolerance = 3
    return args

def set_seed(seed):
    """
    Set random seed for reproducibility.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# ================================
# SEGMENTATION
# ================================

class WorkflowSegmenter:
    """
    Workflow segmentation using foundation models.
    Supports:
    - Local Hugging Face models
    - API-based models (OpenAI, Anthropic, Google)
    """
    
    def __init__(
        self,
        model_name: str,
        provider: str,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        max_tokens: int = 1000,
        temperature: float = 0.2,
        gpu_ids: Optional[str] = None,
        batch_size: int = 8,
        output_dir: Optional[str] = None
    ):
        """
        Initialize the workflow segmenter.
        
        Args:
            model_name: Name of the model to use
            provider: One of "openai", "anthropic", "google", "huggingface"
            api_key: API key for the provider
            api_base: API base URL (if using a deployment other than default)
            max_tokens: Maximum number of tokens to generate
            temperature: Temperature for generation (0.0 for deterministic)
            gpu_ids: IDs of GPUs to use (via CUDA_VISIBLE_DEVICES)
            batch_size: Batch size for Hugging Face inference
            output_dir: Directory to save prompt and generated text
        """
        self.model_name = model_name
        self.provider = provider.lower()
        self.api_key = api_key
        self.api_base = api_base
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.batch_size = batch_size
        self.output_dir = output_dir
        
        # Only set CUDA_VISIBLE_DEVICES if it hasn't already been set
        if gpu_ids and "CUDA_VISIBLE_DEVICES" not in os.environ:
            os.environ["CUDA_VISIBLE_DEVICES"] = gpu_ids
        
        # Set up client based on provider
        if self.provider == "openai":
            try:
                import openai
                self.client = openai.OpenAI(api_key=api_key, base_url=api_base) if api_key else openai.OpenAI()
            except ImportError:
                raise ImportError("OpenAI Python package not installed. Run `pip install openai`.")
                
        elif self.provider == "anthropic":
            try:
                import anthropic
                self.client = anthropic.Anthropic(api_key=api_key)
            except ImportError:
                raise ImportError("Anthropic Python package not installed. Run `pip install anthropic`.")
                
        elif self.provider == "google":
            try:
                from google import genai
                self.client = genai.Client(api_key=api_key)
            except ImportError:
                raise ImportError("Google genai package not installed. Run `pip install google-genai`.")
                
        elif self.provider == "huggingface":
            try:
                import torch
                from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
            except ImportError:
                raise ImportError("Transformers package not installed. Run `pip install transformers`.")

            self.tokenizer = AutoTokenizer.from_pretrained(model_name, token=api_key)

            model_load_kwargs = {"torch_dtype": "auto"}
            # Base arguments for pipeline
            pipeline_init_kwargs = {
                "tokenizer": self.tokenizer,
            }

            can_use_cuda = torch.cuda.is_available() and torch.cuda.device_count() > 0

            if can_use_cuda:
                logger.info(f"CUDA is available with {torch.cuda.device_count()} visible devices")
                cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "")
                if cuda_visible_devices:
                    logger.info(f"Using GPUs specified by CUDA_VISIBLE_DEVICES: {cuda_visible_devices}")
                
                if gpu_ids and "," in gpu_ids:
                    logger.info(f"Multi-GPU: Model will be loaded with device_map='auto'")
                    model_load_kwargs["device_map"] = "auto"
                elif gpu_ids:
                    logger.info(f"Single-GPU: Pipeline will target device 0")
                    pipeline_init_kwargs["device"] = 0 
                else:
                    logger.info(f"No specific GPUs (user) & CUDA available: Using device_map='auto'")
                    model_load_kwargs["device_map"] = "auto"
            else:
                # No CUDA available
                logger.info("CUDA not available, using CPU")
                pipeline_init_kwargs["device"] = "cpu"
                if "device_map" in model_load_kwargs:
                    del model_load_kwargs["device_map"]

            self.model = AutoModelForCausalLM.from_pretrained(model_name, token=api_key, **model_load_kwargs)
            pipeline_init_kwargs["model"] = self.model
            self.client = pipeline("text-generation", **pipeline_init_kwargs)
        else:
            raise ValueError(f"Unsupported provider: {provider}. Use 'openai', 'anthropic', 'google', or 'huggingface'.")
                
    def segment(
        self,
        events_description: List[str],
        process_definitions_text: str
    ) -> List[Dict[str, Any]]:
        """
        Segment a workflow sequence into constituent workflows.
        
        Args:
            events_description: List of event descriptions in the sequence
            
        Returns:
            List of segments with start_index and end_index
        """
        
        
        # Construct prompt for the LLM
        # Number each event for clearer reference in the model's response
        sequence_text = "\n".join([f"{i}. {event}" for i, event in enumerate(events_description)])
        
        prompt = SEGMENTATION_PROMPT.format(
            process_definitions=process_definitions_text,
            sequence=sequence_text
        )
        
        # Handle different providers
        if self.provider == "openai":
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=self.max_tokens,
                temperature=self.temperature
            )
            generated_text = response.choices[0].message.content.strip()
            
        elif self.provider == "anthropic":
            response = self.client.messages.create(
                model=self.model_name,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=self.max_tokens,
                temperature=self.temperature
            )
            generated_text = response.content[0].text.strip()
            
        elif self.provider == "google":
            from google.genai import types
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_PROMPT,
                    max_output_tokens=self.max_tokens,
                    temperature=self.temperature,
                ),
            )
            generated_text = response.text.strip()
            
        elif self.provider == "huggingface":
            outputs = self.client(
                [{"role": "user", "content": prompt}],
                max_new_tokens=self.max_tokens,
                temperature=self.temperature,
                do_sample=self.temperature > 0,
            )
            generated_text = outputs[0]["generated_text"]
            try:
                # Extract content from the model's response
                if isinstance(generated_text, list) and len(generated_text) > 1:
                    generated_text = generated_text[1]["content"].strip()
            except (IndexError, KeyError, TypeError):
                logger.warning("Could not properly extract Hugging Face response content")
        
        # Save prompt and generated text to a file
        if self.output_dir:
            # Create a record for this segmentation
            record = {
                "prompt": prompt,
                "generated_text": generated_text,
                "timestamp": datetime.datetime.now().isoformat()
            }
            
            # Path for the JSON file
            json_path = os.path.join(self.output_dir, "prompts_and_responses.json")
            
            # Check if file exists and load existing records
            if os.path.exists(json_path):
                try:
                    with open(json_path, "r") as f:
                        records = json.load(f)
                except json.JSONDecodeError:
                    records = []
            else:
                records = []
            
            # Append new record
            records.append(record)
            
            # Save updated records
            with open(json_path, "w") as f:
                json.dump(records, f, indent=4)
        
        # Parse the response to extract segments
        return self._parse_segments(generated_text)
    
    def _parse_segments(self, generated_text: str) -> List[Dict[str, Any]]:
        """
        Parse the generated text to extract segmentation information.
        
        Args:
            generated_text: Text generated by the model
            
        Returns:
            List of segments with start_index and end_index
        """
        # Look for the answer pattern in the generated text
        answer_pattern = r'<answer>(.*?)</answer>'
        match = re.search(answer_pattern, generated_text, re.DOTALL)
        
        if match:
            try:
                # Extract the JSON-formatted answer
                json_str = match.group(1).strip()
                segments = json.loads(json_str)
                
                # Validate the structure of each segment
                valid_segments = []
                for segment in segments:
                    if all(k in segment for k in ['start_index', 'end_index']):
                        valid_segments.append({
                            'start_index': int(segment['start_index']),
                            'end_index': int(segment['end_index'])
                        })
                    else:
                        logger.warning(f"Skipping invalid segment: {segment}")
                
                return valid_segments
            except (json.JSONDecodeError, ValueError) as e:
                logger.error(f"Failed to parse JSON from model output: {e}")
                return []
        else:
            logger.warning("No answer pattern found in model output")
            return []

# ================================
# EVALUATION METRICS
# ================================

def evaluate_boundary_detection(true_boundaries: List[int], 
                               pred_boundaries: List[int], 
                               sequence_length: int,
                               tolerance: int = 0) -> Dict[str, float]:
    """
    Evaluate boundary detection performance.
    
    Args:
        true_boundaries: List of true boundary indices
        pred_boundaries: List of predicted boundary indices
        sequence_length: Length of the event sequence
        tolerance: Number of positions to tolerate errors (0 for exact matches)
        
    Returns:
        Dictionary with precision, recall, and F1 score
    """
    # For boundary detection with tolerance
    if tolerance > 0:
        # Count a predicted boundary as correct if it's within tolerance of a true boundary
        true_positives = 0
        for pred_b in pred_boundaries:
            if any(abs(pred_b - true_b) <= tolerance for true_b in true_boundaries):
                true_positives += 1
        
        precision = true_positives / len(pred_boundaries) if pred_boundaries else 0
        recall = true_positives / len(true_boundaries) if true_boundaries else 0
    else:
        # For exact boundary detection
        # Convert to binary arrays where 1 indicates a boundary
        true_array = np.zeros(sequence_length)
        pred_array = np.zeros(sequence_length)
        
        for b in true_boundaries:
            if 0 <= b < sequence_length:
                true_array[b] = 1
        
        for b in pred_boundaries:
            if 0 <= b < sequence_length:
                pred_array[b] = 1
        
        # Calculate precision, recall, F1
        precision, recall, f1, _ = precision_recall_fscore_support(
            true_array, pred_array, average='binary', zero_division=0
        )
    
    # Calculate F1 score
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

def calculate_segmentation_edit_distance(true_boundaries: List[int], 
                                        pred_boundaries: List[int]) -> int:
    """
    Calculate the edit distance between true and predicted segmentations.
    
    Args:
        true_boundaries: List of true boundaries
        pred_boundaries: List of predicted boundaries
        
    Returns:
        Edit distance (number of operations to transform prediction to truth)
    """
    # Calculate Levenshtein distance
    m, n = len(true_boundaries), len(pred_boundaries)
    
    # Create a matrix of size (m+1) x (n+1)
    dp = [[0 for _ in range(n + 1)] for _ in range(m + 1)]
    
    # Initialize first row and column
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    
    # Fill the matrix
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if true_boundaries[i-1] == pred_boundaries[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(dp[i-1][j],      # Deletion
                                  dp[i][j-1],      # Insertion
                                  dp[i-1][j-1])    # Substitution
    
    # Return the edit distance
    return dp[m][n]

def visualize_segmentation(true_segments: List[Dict[str, Any]], 
                          pred_segments: List[Dict[str, Any]], 
                          sequence_length: int,
                          output_path: str = None) -> None:
    """
    Visualize the segmentation results.
    
    Args:
        true_segments: List of true segments
        pred_segments: List of predicted segments
        sequence_length: Length of the event sequence
        output_path: Path to save the visualization
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    
    # Plot ground truth segmentation
    for i, seg in enumerate(true_segments):
        start = seg['start_index']
        end = seg['end_index']
        width = end - start + 1
        
        ax1.barh(y=0, width=width, left=start, height=0.5, 
                color=plt.cm.tab10(i % 10), alpha=0.7)
        ax1.text(start + width/2, 0, f"Segment {i+1}", 
                ha='center', va='center', fontsize=9)
    
    ax1.set_yticks([])
    ax1.set_title('Ground Truth Segmentation')
    
    # Plot predicted segmentation
    for i, seg in enumerate(pred_segments):
        start = seg['start_index']
        end = seg['end_index']
        width = end - start + 1
        
        ax2.barh(y=0, width=width, left=start, height=0.5, 
                color=plt.cm.tab10(i % 10), alpha=0.7)
        ax2.text(start + width/2, 0, f"Segment {i+1}", 
                ha='center', va='center', fontsize=9)
    
    ax2.set_yticks([])
    ax2.set_title('Predicted Segmentation')
    ax2.set_xlabel('Event Index')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Visualization saved to {output_path}")
    
    plt.close()

# ================================
# MAIN EVALUATION FUNCTION
# ================================

def evaluation(processor, args):
    """
    Evaluate workflow segmentation using foundation models.
    
    Args:
        processor: WorkflowDataProcessor instance
        args: Command line arguments
    """
    output_dir = os.path.join(args.output_dir, "segmentation", args.model_name.split("/")[-1])
    os.makedirs(output_dir, exist_ok=True)
    
    # Load and prepare segmentation data
    logger.info("Preparing segmentation data...")
    segmentation_data = processor.prepare_workflow_segmentation_data(
        min_workflows=args.min_workflows,
        max_workflows=args.max_workflows,
        random_state=args.seed,
        num_samples=args.num_samples
    )
    
    # Initialize segmenter
    segmenter = WorkflowSegmenter(
        model_name=args.model_name,
        provider=args.provider,
        api_key=args.api_key,
        api_base=args.api_base,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        gpu_ids=args.gpus,
        batch_size=args.batch_size,
        output_dir=output_dir
    )
    
    # Load process definitions from JSON file
    try:
        with open("data/process_definitions.json", "r") as f:
            process_defs = json.load(f)
        
        # Format process definitions for the prompt
        process_definitions_text = ""
        for process in process_defs:
            if process['name'].lower() in [p.lower() for p in processor.process_names]:
                process_definitions_text += f"- {process['name']}: {process['description']}\n"
    except Exception as e:
        logger.warning(f"Could not load process definitions: {e}")
        process_definitions_text = "No process definitions available."
    
    logger.info(f"Process definitions: {process_definitions_text}")
    # Evaluate on all sequences
    logger.info(f"Running segmentation evaluation with {args.model_name} on {len(segmentation_data)} sequences...")
    
    results = []
    
    for sequence in tqdm.tqdm(segmentation_data, desc="Evaluating segmentation"):
        # Create true segments list for evaluation
        true_segments = []
        for i in range(len(sequence['segment_boundaries']) - 1):
            start_idx = sequence['segment_boundaries'][i]
            end_idx = sequence['segment_boundaries'][i+1] - 1
            
            true_segments.append({
                'start_index': start_idx,
                'end_index': end_idx
            })
        
        # Get model prediction
        pred_segments = segmenter.segment(
            events_description=sequence['sequence'],
            process_definitions_text=process_definitions_text
        )
        
        pred_boundaries = [seg['start_index'] for seg in pred_segments if seg['start_index'] > 0]
        
        # Evaluate boundary detection
        boundary_metrics = evaluate_boundary_detection(
            true_boundaries=sequence['segment_boundaries'][1:-1],  # Skip the first and last boundary (0) and (last)
            pred_boundaries=pred_boundaries,
            sequence_length=len(sequence['sequence']),
            tolerance=args.tolerance
        )
        
        # Calculate edit distance
        edit_distance = calculate_segmentation_edit_distance(
            true_boundaries=sequence['segment_boundaries'][1:-1],
            pred_boundaries=pred_boundaries
        )
        
        # Create visualization
        visualize_segmentation(
            true_segments=true_segments,
            pred_segments=pred_segments,
            sequence_length=len(sequence['sequence']),
            output_path=os.path.join(output_dir, f"visualization_{sequence['sequence_id']}.png")
        )
        
        # Store results
        results.append({
            'sequence_id': sequence['sequence_id'],
            'num_workflows': sequence['num_workflows'],
            'sequence_length': len(sequence['sequence']),
            'boundary_precision': boundary_metrics['precision'],
            'boundary_recall': boundary_metrics['recall'],
            'boundary_f1': boundary_metrics['f1'],
            'edit_distance': edit_distance,
            'true_segments': true_segments,
            'pred_segments': pred_segments
        })
        
        # Clear GPU memory if using PyTorch
        torch.cuda.empty_cache()
        
        # Wait between requests if specified to avoid rate limiting
        time.sleep(args.wait_time)
    
    # Calculate aggregate metrics
    avg_precision = np.mean([r['boundary_precision'] for r in results])
    avg_recall = np.mean([r['boundary_recall'] for r in results])
    avg_f1 = np.mean([r['boundary_f1'] for r in results])
    avg_edit_distance = np.mean([r['edit_distance'] for r in results])
    
    # Log results
    logger.info(f"Segmentation Boundary Detection - Precision: {avg_precision:.4f}, Recall: {avg_recall:.4f}, F1: {avg_f1:.4f}")
    logger.info(f"Average Segmentation Edit Distance: {avg_edit_distance:.2f}")
    
    # Save detailed results
    results_path = os.path.join(output_dir, "segmentation_results.json")
    with open(results_path, "w") as f:
        # Create a simplified version for JSON serialization
        serializable_results = []
        for r in results:
            r_copy = r.copy()
            # Convert numpy values to Python native types
            for k, v in r_copy.items():
                if isinstance(v, np.floating):
                    r_copy[k] = float(v)
                elif isinstance(v, np.integer):
                    r_copy[k] = int(v)
            serializable_results.append(r_copy)
            
        json.dump({
            'individual_results': serializable_results,
            'aggregate_metrics': {
                'boundary_precision': float(avg_precision),
                'boundary_recall': float(avg_recall),
                'boundary_f1': float(avg_f1),
                'edit_distance': float(avg_edit_distance)
            }
        }, f, indent=2)
    
    logger.info(f"Detailed results saved to {results_path}")
    
    # Create summary visualizations
    create_summary_plots(results, output_dir)

def create_summary_plots(results, output_dir):
    """
    Create summary plots of the segmentation results.
    
    Args:
        results: List of result dictionaries
        output_dir: Directory to save the plots
    """
    # Plot performance vs. number of workflows
    plt.figure(figsize=(10, 6))
    
    # Group results by number of workflows
    workflow_counts = {}
    for r in results:
        num = r['num_workflows']
        if num not in workflow_counts:
            workflow_counts[num] = []
        workflow_counts[num].append(r)
    
    # Calculate average metrics for each workflow count
    x = sorted(workflow_counts.keys())
    y_f1 = [np.mean([r['boundary_f1'] for r in workflow_counts[num]]) for num in x]
    
    plt.plot(x, y_f1, 'o-', label='Boundary F1')
    plt.xlabel('Number of Workflows')
    plt.ylabel('F1 Score')
    plt.title('Segmentation Performance vs. Number of Workflows')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, "performance_vs_workflows.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot performance vs. sequence length
    plt.figure(figsize=(10, 6))
    
    # Bin results by sequence length
    lengths = [r['sequence_length'] for r in results]
    min_len, max_len = min(lengths), max(lengths)
    num_bins = 5
    bin_width = (max_len - min_len) / num_bins if max_len > min_len else 1
    
    length_bins = {}
    for r in results:
        bin_idx = min(num_bins - 1, int((r['sequence_length'] - min_len) / bin_width)) if bin_width > 0 else 0
        bin_label = f"{int(min_len + bin_idx * bin_width)}-{int(min_len + (bin_idx + 1) * bin_width)}"
        
        if bin_label not in length_bins:
            length_bins[bin_label] = []
        length_bins[bin_label].append(r)
    
    # Calculate average metrics for each length bin
    x_labels = sorted(length_bins.keys(), key=lambda x: int(x.split('-')[0]))
    y_f1 = [np.mean([r['boundary_f1'] for r in length_bins[label]]) for label in x_labels]
    
    x = range(len(x_labels))
    plt.bar(x, y_f1, width=0.6, alpha=0.7)
    plt.xlabel('Sequence Length')
    plt.ylabel('F1 Score')
    plt.title('Segmentation Performance vs. Sequence Length')
    plt.xticks(x, x_labels, rotation=45)
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "performance_vs_length.png"), dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    # Parse arguments
    args = parse_args()
    
    # Set GPU environment variable if specified
    if args.gpus:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
        logger.info(f"Setting CUDA_VISIBLE_DEVICES={args.gpus}")
    
    import torch
    # Set random seed
    set_seed(args.seed)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize data processor and load data
    processor = WorkflowDataProcessor(data_path=args.data_path)
    processor.load_data()
    
    # Run segmentation evaluation
    evaluation(processor, args)
