# ============================================================================
# ENTROPHY © 2025 by Workfabric
# Licensed under Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International
# https://creativecommons.org/licenses/by-nc-sa/4.0/
# ============================================================================

import argparse
import datetime
import json
import logging
import os
import re
import time
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tqdm
import yaml
from sklearn.metrics import classification_report, confusion_matrix

from data_processor import WorkflowDataProcessor

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("workflow_classification")

# ================================
# PROMPTS
# ================================

PROMPT = """\
Given the following user interaction sequence, classify it into one of the following workflow types: {classes}.
        
User interaction sequence:
{sequence}

Provide your answer enclosed in \\answer{{}}."""


SYSTEM_PROMPT = "You are a workflow classification assistant that analyzes user interactions and determines the workflow type."


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
    with open(config_path, "r") as f:
        if config_path.endswith(".yaml") or config_path.endswith(".yml"):
            config = yaml.safe_load(f)
        else:
            # Fallback to JSON for backward compatibility
            config = json.load(f)
    logger.info(f"Loaded configuration from {config_path}")
    return config


def parse_args():
    """
    Parse command line arguments.
    """
    parser = argparse.ArgumentParser(description="Workflow Classification")

    # Config file argument
    parser.add_argument("--config", type=str, help="Path to YAML configuration file")

    # Data arguments
    parser.add_argument("--data_path", type=str, help="Path to input JSON data file")
    parser.add_argument("--output_dir", type=str, help="Directory for saving outputs")

    # Model arguments
    parser.add_argument("--model_name", type=str, help="Foundation model name")
    parser.add_argument(
        "--provider", type=str, help="Provider for workflow classification"
    )
    parser.add_argument(
        "--api_key", type=str, help="API key for workflow classification"
    )
    parser.add_argument(
        "--api_base", type=str, help="API base for workflow classification"
    )
    parser.add_argument(
        "--max_tokens", type=int, help="Maximum tokens for workflow classification"
    )
    parser.add_argument(
        "--temperature", type=float, help="Temperature for workflow classification"
    )
    parser.add_argument("--seed", type=int, help="Random seed")
    parser.add_argument(
        "--gpus",
        type=str,
        help="Comma-separated list of GPU IDs to use (e.g., '0,1,2')",
    )
    parser.add_argument("--batch_size", type=int, help="Batch size for classification")
    parser.add_argument(
        "--wait_time",
        type=float,
        help="Wait time between classifications to avoid rate limiting",
    )
    args = parser.parse_args()

    # If a config file is provided, load it and update args
    if args.config:
        config = load_config(args.config)
        # Update args with values from config file
        for key, value in config.items():
            if key != "config" and not getattr(
                args, key, None
            ):  # Don't override explicitly passed args
                setattr(args, key, value)

    # Check required arguments after merging config
    required_args = ["data_path", "model_name", "provider"]
    missing_args = [arg for arg in required_args if not getattr(args, arg, None)]
    if missing_args:
        parser.error(
            f"The following required arguments are missing: {', '.join(missing_args)}"
        )

    # Set defaults for optional arguments if not provided
    if not args.output_dir:
        args.output_dir = "./outputs"
    if not args.max_tokens:
        args.max_tokens = 500
    if not args.temperature:
        args.temperature = 0.8
    if not args.wait_time:
        args.wait_time = 0.0
    if not args.seed:
        args.seed = 2404

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
# CLASSIFICATION
# ================================


class WorkflowClassifier:
    """
    Workflow classifier using SOTA foundation models.
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
        max_tokens: int = 500,
        temperature: float = 0.8,
        gpu_ids: Optional[str] = None,
        batch_size: int = 8,
        output_dir: Optional[str] = None,
    ):
        """
        Initialize the workflow classifier.

        Args:
            model_name: Name of the model to use
            provider: One of "openai", "anthropic", "google", "huggingface", "local"
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

                self.client = (
                    openai.OpenAI(api_key=api_key, base_url=api_base)
                    if api_key
                    else openai.OpenAI()
                )
            except ImportError:
                raise ImportError(
                    "OpenAI Python package not installed. Run `pip install openai`."
                )

        elif self.provider == "anthropic":
            try:
                import anthropic

                self.client = anthropic.Anthropic(api_key=api_key)
            except ImportError:
                raise ImportError(
                    "Anthropic Python package not installed. Run `pip install anthropic`."
                )

        elif self.provider == "google":
            try:
                from google import genai

                self.client = genai.Client(api_key=api_key)
            except ImportError:
                raise ImportError(
                    "Google genai package not installed. Run `pip install google-genai`."
                )

        elif self.provider == "huggingface":
            try:
                from transformers import (AutoModelForCausalLM, AutoTokenizer,
                                          pipeline)
            except ImportError:
                raise ImportError(
                    "Transformers package not installed. Run `pip install transformers`."
                )

            self.tokenizer = AutoTokenizer.from_pretrained(model_name, token=api_key)

            model_load_kwargs = {"torch_dtype": "auto"}
            # Base arguments for pipeline; model will be added after loading.
            pipeline_init_kwargs = {
                "tokenizer": self.tokenizer,
            }

            can_use_cuda = torch.cuda.is_available() and torch.cuda.device_count() > 0

            if can_use_cuda:
                print(
                    f"CUDA is available with {torch.cuda.device_count()} visible devices"
                )
                cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "")
                if cuda_visible_devices:
                    print(
                        f"Using GPUs specified by CUDA_VISIBLE_DEVICES: {cuda_visible_devices}"
                    )

                if gpu_ids and "," in gpu_ids:
                    print(
                        "Multi-GPU: Model will be loaded with device_map='auto'. Pipeline will use model's device map."
                    )
                    model_load_kwargs["device_map"] = "auto"
                elif gpu_ids:
                    print(
                        "Single-GPU: Pipeline will target device 0 (relative to visible devices)."
                    )
                    pipeline_init_kwargs["device"] = 0
                else:
                    print(
                        "No specific GPUs (user) & CUDA available: Model will be loaded with device_map='auto'. Pipeline will use model's device map."
                    )
                    model_load_kwargs["device_map"] = "auto"
            else:
                # No CUDA available
                print("CUDA not available, using CPU")
                pipeline_init_kwargs["device"] = "cpu"
                if "device_map" in model_load_kwargs:
                    del model_load_kwargs["device_map"]

            self.model = AutoModelForCausalLM.from_pretrained(
                model_name, token=api_key, **model_load_kwargs
            )
            pipeline_init_kwargs["model"] = self.model
            self.client = pipeline("text-generation", **pipeline_init_kwargs)
        else:
            raise ValueError(
                f"Unsupported provider: {provider}. Use 'openai', 'anthropic', 'google', or 'huggingface'."
            )

    def classify(
        self, events_description: str, possible_classes: List[str]
    ) -> Tuple[str, float]:
        """
        Classify a workflow based on its events description.

        Args:
            events_description: Textual description of workflow events
            possible_classes: List of possible class labels

        Returns:
            predicted_class
        """
        # Construct prompt for the LLM
        prompt = PROMPT.format(
            classes=", ".join(possible_classes), sequence=events_description
        )

        # Handle different providers
        if self.provider == "openai":
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=self.max_tokens,
                temperature=self.temperature,
            )
            generated_text = response.choices[0].message.content.strip()

        elif self.provider == "anthropic":
            response = self.client.messages.create(
                model=self.model_name,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=self.max_tokens,
                temperature=self.temperature,
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
            generated_text = generated_text[1]["content"].strip()

        # save prompt and generated text to a file
        if self.output_dir:
            # Create a record for this classification
            record = {
                "prompt": prompt,
                "generated_text": generated_text,
                "timestamp": datetime.datetime.now().isoformat(),
            }

            # Path for the JSON file
            json_path = os.path.join(self.output_dir, "prompt_and_generated_text.json")

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

        # Look for the answer pattern in the generated text
        answer_pattern = r"\\answer\{(.*?)\}"
        match = re.search(answer_pattern, generated_text)

        if match:
            matched_class = match.group(1).strip()
        else:
            # If no answer pattern is found, return None
            matched_class = None

        return matched_class

    def classify_batch(
        self, event_descriptions: List[str], possible_classes: List[str]
    ) -> List[str]:
        """
        Classify multiple workflows in batch for efficient GPU processing.

        Args:
            event_descriptions: List of textual descriptions of workflow events
            possible_classes: List of possible class labels

        Returns:
            List of predicted classes
        """
        if self.provider != "huggingface":
            raise ValueError(
                "Batch processing is only supported for HuggingFace provider."
            )

        # Batch processing for HuggingFace provider
        prompts = []
        class_str = ", ".join(possible_classes)

        for desc in event_descriptions:
            prompt = PROMPT.format(classes=class_str, sequence=desc)
            prompts.append([{"role": "user", "content": prompt}])

        # Process in batches for efficient GPU utilization
        all_outputs = []
        for i in range(0, len(prompts), self.batch_size):
            batch = prompts[i : i + self.batch_size]
            outputs = self.client(
                batch,
                max_new_tokens=self.max_tokens,
                temperature=self.temperature,
                do_sample=self.temperature > 0,
            )
            all_outputs.extend(outputs)

        # Parse all responses
        predicted_classes = []
        answer_pattern = r"\\answer\{(.*?)\}"

        for output in all_outputs:
            generated_text = output[0]["generated_text"]
            try:
                # Access content from the response format
                if isinstance(generated_text, list) and len(generated_text) > 1:
                    generated_text = generated_text[1]["content"].strip()

                match = re.search(answer_pattern, generated_text)
                if match:
                    matched_class = match.group(1).strip()
                else:
                    matched_class = None

                predicted_classes.append(matched_class)
            except (IndexError, AttributeError, TypeError) as e:
                # Handle unexpected response formats
                logger.error(f"Error parsing model output: {e}")
                predicted_classes.append(None)

        return predicted_classes


def create_heatmap(true_labels, pred_labels, figsize=(10, 8), output_path=None):
    label_list = list(set(true_labels))
    report = pd.DataFrame(
        classification_report(
            true_labels,
            pred_labels,
            labels=label_list,
            output_dict=True,
            zero_division=0,
        )
    )
    report.drop(columns=["weighted avg"], inplace=True)
    try:
        report.drop(columns=["micro avg"], inplace=True)
    except:  # noqa: E722
        logger.debug('Encountered an exception dropping column `micro avg`.')
    report.rename(columns={"macro avg": "Average"}, inplace=True)

    plt.figure(figsize=figsize, dpi=300)
    sns.set(font_scale=1.2)
    sns.set_style("whitegrid")

    # Create a more visually appealing heatmap
    ax = sns.heatmap(
        report.iloc[:-1, :].T,
        annot=True,
        fmt=".3f",
        cmap="viridis",
        linewidths=0.5,
        cbar_kws={"label": "Score"},
        annot_kws={"size": 10},
    )

    # Enhance the plot with better styling
    plt.title(
        "Classification Performance Metrics", fontsize=16, fontweight="bold", pad=20
    )
    plt.ylabel("Classes", fontsize=14, labelpad=10)
    plt.xlabel("Metrics", fontsize=14, labelpad=10)

    # Improve layout
    plt.tight_layout()

    # Save the figure if output path is provided
    if output_path:
        plt.savefig(output_path, bbox_inches="tight", dpi=300)
        print(f"Heatmap saved to {output_path}")

    return report


def create_confusion_matrix(
    true_labels, pred_labels, figsize=(10, 8), output_path=None
):
    """
    Create and optionally save a confusion matrix visualization.

    Args:
        true_labels: List of ground truth labels
        pred_labels: List of predicted labels
        figsize: Figure size as tuple (width, height)
        output_path: Path to save the figure (optional)

    Returns:
        DataFrame containing the confusion matrix
    """
    label_list = list(set(true_labels + pred_labels))
    cm = confusion_matrix(true_labels, pred_labels, labels=label_list)
    cm_df = pd.DataFrame(cm, index=label_list, columns=label_list)

    plt.figure(figsize=figsize, dpi=300)
    sns.set(font_scale=1.2)
    sns.set_style("whitegrid")

    # Create a more visually appealing heatmap
    ax = sns.heatmap(
        cm_df,
        annot=True,
        fmt="d",
        cmap="Blues",
        linewidths=0.5,
        square=True,
        cbar_kws={"label": "Count"},
        annot_kws={"size": 10},
    )

    # Enhance the plot with better styling
    plt.ylabel("Actual Label", fontsize=14, labelpad=10)
    plt.xlabel("Predicted Label", fontsize=14, labelpad=10)
    plt.title("Confusion Matrix", fontsize=16, fontweight="bold", pad=20)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)

    # Improve layout
    plt.tight_layout()

    # Save the figure if output path is provided
    if output_path:
        plt.savefig(output_path, bbox_inches="tight", dpi=300)
        print(f"Confusion matrix saved to {output_path}")

    return cm_df


def evaluation(processor, args):
    """
    Evaluate workflow classification.

    Args:
        processor: WorkflowDataProcessor instance
        args: Command line arguments
    """
    output_dir = os.path.join(args.output_dir, args.model_name.split("/")[-1])

    os.makedirs(output_dir, exist_ok=True)

    # Prepare classification data
    classification_data = processor.prepare_workflow_classification_data()
    class_names = processor.label_encoder.classes_

    # Log GPU configuration
    if args.gpus:
        logger.info(
            f"Evaluation using GPUs: {args.gpus} (set via CUDA_VISIBLE_DEVICES)"
        )
    else:
        logger.info("Evaluation using default GPU configuration")

    # Initialize classifier with batch size
    classifier = WorkflowClassifier(
        model_name=args.model_name,
        provider=args.provider,
        api_key=args.api_key,
        api_base=args.api_base,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        gpu_ids=args.gpus,
        batch_size=args.batch_size,
        output_dir=output_dir,
    )

    # Evaluate on test data
    logger.info(f"Running evaluation with {args.model_name}...")
    all_preds = []
    all_labels = []

    if args.provider.lower() == "huggingface" and args.batch_size > 1:
        # Batch processing for HuggingFace provider
        logger.info(
            f"Using batch processing (batch_size={args.batch_size}) for HuggingFace provider"
        )

        # Prepare batch data
        batch_descriptions = []
        batch_indices = []

        for i, item in enumerate(classification_data):
            # Create description for the workflow
            description = "\n".join(item["event_descriptions"])
            batch_descriptions.append(description)
            batch_indices.append(i)
            all_labels.append(item["process_name"])

        # Process batches
        for i in tqdm.tqdm(
            range(0, len(batch_descriptions), args.batch_size),
            desc="Processing batches",
        ):
            batch_desc = batch_descriptions[i : i + args.batch_size]
            batch_pred_classes = classifier.classify_batch(
                event_descriptions=batch_desc, possible_classes=list(class_names)
            )

            # Process predictions
            for j, pred_class in enumerate(batch_pred_classes):
                if pred_class not in class_names:
                    logger.info(
                        f"Predicted class {pred_class} not found in class names {class_names}"
                    )
                    if pred_class is None:
                        pred_class = "Format Error"
                    else:
                        pred_class = "Other"

                all_preds.append(pred_class)
    else:
        # Single-item processing for other providers
        for i, item in tqdm.tqdm(
            enumerate(classification_data), total=len(classification_data)
        ):
            # Create description for the workflow
            description = "\n".join(item["event_descriptions"])

            # Get prediction
            pred_class = classifier.classify(
                events_description=description, possible_classes=list(class_names)
            )
            if pred_class not in class_names:
                logger.info(
                    f"Predicted class {pred_class} not found in class names {class_names}"
                )
                if pred_class is None:
                    pred_class = "Format Error"
                else:
                    pred_class = "Other"

            # wait for args.wait_time seconds
            time.sleep(args.wait_time)
            torch.cuda.empty_cache()
            # Store results
            all_preds.append(pred_class)
            all_labels.append(item["process_name"])

    # Calculate metrics
    accuracy = np.mean(np.array(all_preds) == np.array(all_labels))

    # Generate classification report with only the labels that are present
    report = create_heatmap(
        all_labels, all_preds, output_path=os.path.join(output_dir, "heatmap.png")
    )

    # Print and save report
    logger.info(f"Classification accuracy: {accuracy:.4f}")
    logger.info(f"Classification report:\n{report}")

    # Save classification report as JSON
    report_json_path = os.path.join(output_dir, "report.json")
    report.to_json(report_json_path, orient="columns", indent=4)
    logger.info(f"Classification report saved to {report_json_path}")

    # Generate confusion matrix
    cm = create_confusion_matrix(
        all_labels,
        all_preds,
        output_path=os.path.join(output_dir, "confusion_matrix.png"),
    )

    # Save confusion matrix as JSON
    cm_json_path = os.path.join(output_dir, "confusion_matrix.json")
    cm.to_json(cm_json_path, orient="columns", indent=4)
    logger.info(f"Confusion matrix saved to {cm_json_path}")

    # Save detailed predictions
    detailed_results = []
    for i, item in enumerate(classification_data):
        true_class = item["process_name"]
        pred_class = all_preds[i]

        detailed_results.append(
            {
                "workflow_id": item["workflow_id"],
                "true_class": true_class,
                "predicted_class": pred_class,
                "correct": true_class == pred_class,
            }
        )

    detailed_path = os.path.join(output_dir, "detailed_results.json")
    with open(detailed_path, "w") as f:
        json.dump(detailed_results, f, indent=2)


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

    # If no GPUs are specified, use the a single GPU if available, otherwise use CPU
    if not args.gpus:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {device}")

    # Initialize data processor and load data
    processor = WorkflowDataProcessor(data_path=args.data_path)
    processor.load_data()

    # Evaluation
    evaluation(processor, args)
