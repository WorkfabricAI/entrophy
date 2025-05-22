# ============================================================================
# ENTROPHY © 2025 by Workfabric
# Licensed under Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International
# https://creativecommons.org/licenses/by-nc-sa/4.0/
# ============================================================================

import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap


class PlotGenerator:
    def __init__(self, input_root, output_dir):
        self.input_root = Path(input_root)
        self.tasks = self._get_tasks()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        # Set enhanced publication-quality plot settings
        plt.style.use("seaborn-v0_8-whitegrid")
        plt.rcParams.update(
            {
                "font.family": "sans-serif",
                "font.sans-serif": [
                    "Arial",
                    "DejaVu Sans",
                    "Liberation Sans",
                    "Bitstream Vera Sans",
                    "sans-serif",
                ],
                "font.size": 12,
                "font.weight": "medium",
                "axes.linewidth": 1.5,
                "axes.labelsize": 14,
                "axes.labelweight": "bold",
                "axes.titlesize": 16,
                "axes.titleweight": "bold",
                "axes.spines.top": False,
                "axes.spines.right": False,
                "axes.grid": True,
                "grid.alpha": 0.3,
                "xtick.major.width": 1.5,
                "ytick.major.width": 1.5,
                "xtick.labelsize": 12,
                "ytick.labelsize": 12,
                "legend.fontsize": 12,
                "legend.frameon": True,
                "legend.framealpha": 0.8,
                "legend.edgecolor": "lightgray",
                "figure.dpi": 300,
                "savefig.dpi": 300,
                "savefig.bbox": "tight",
                "savefig.facecolor": "white",
                "figure.facecolor": "white",
                "figure.autolayout": True,
            }
        )

        # Custom enhanced color palette for models
        self.model_colors = {
            "gpt-4.1": "#1f77b4",
            "claude-3.5-haiku": "#ff7f0e",
            "deepseek-r1": "#d62728",
            "Qwen3-32B": "#9467bd",
            "gemini-2.5-flash": "#8c564b",
        }

        # Create custom colormaps
        self.cmap = LinearSegmentedColormap.from_list(
            "custom_cmap", ["#f7fbff", "#4292c6", "#08306b"], N=256
        )
        self.diverging_cmap = LinearSegmentedColormap.from_list(
            "custom_diverging", ["#d73027", "#f7f7f7", "#1a9850"], N=256
        )

        # Map for custom display names
        self.display_name_map = {
            # Model names
            "gpt-4.1": "GPT-4.1",
            "claude-3.5-haiku": "Claude-3.5-Haiku",
            "deepseek-r1": "DeepSeek-R1",
            "Qwen3-32B": "Qwen3-32B",
            "gemini-2.5-flash": "Gemini-2.5-Flash",
            "hr": "HR",
        }

    def _get_display_name(self, original_name):
        """Get the display name for a given original name (model, domain, task)."""
        # First, try direct lookup
        if original_name in self.display_name_map:
            return self.display_name_map[original_name]

        # If not found, apply some default formatting
        # For task types, often they are like 'domain_task' or 'task_subtype'
        if "_" in original_name:
            parts = original_name.split("_")
            # Check if the first part is a known domain, and if so, format the rest
            if parts[0] in self.tasks or parts[0] in self.display_name_map:
                domain_display = self._get_display_name(
                    parts[0]
                )  # Recursive call for domain part
                task_part_display = " ".join(p.capitalize() for p in parts[1:])
                return f"{domain_display} - {task_part_display}"
            else:  # Default for other underscore-separated names
                return " ".join(p.capitalize() for p in parts)

        # For simple names (likely domains or unmapped models)
        return original_name.capitalize()

    def _get_tasks(self):
        """Get all task domains and their subtypes from the input directory."""
        tasks = {}
        for domain in os.listdir(self.input_root):
            domain_path = self.input_root / domain
            if domain_path.is_dir():
                tasks[domain] = []
                for task_type in os.listdir(domain_path):
                    task_path = domain_path / task_type
                    if task_path.is_dir():
                        tasks[domain].append(task_type)
        return tasks

    def load_classification_data(self, domain, task_type):
        """Load classification metrics for all models for a specific domain and task."""
        task_path = self.input_root / domain / task_type
        results = {}

        for model_name in os.listdir(task_path):
            model_path = task_path / model_name
            if not model_path.is_dir():
                continue

            report_path = model_path / "report.json"
            confusion_matrix_path = model_path / "confusion_matrix.json"
            detailed_results_path = model_path / "detailed_results.json"

            if report_path.exists():
                with open(report_path, "r") as f:
                    report = json.load(f)

                # Extract metrics
                if "Average" in report:
                    results[model_name] = {
                        "precision": report["Average"]["precision"],
                        "recall": report["Average"]["recall"],
                        "f1": report["Average"]["f1-score"],
                        "classes": {
                            k: report[k]
                            for k in report
                            if k not in ["Average", "accuracy"]
                        },
                        "support": report["Average"]["support"],
                    }

                    # Get accuracy from report.json if available
                    if "accuracy" in report:
                        results[model_name]["accuracy"] = report["accuracy"][
                            "precision"
                        ]

            # If detailed_results.json exists and accuracy wasn't found in report.json
            if detailed_results_path.exists() and (
                model_name not in results or "accuracy" not in results[model_name]
            ):
                with open(detailed_results_path, "r") as f:
                    detailed_results = json.load(f)

                # Calculate accuracy from detailed_results.json
                correct_count = sum(
                    1 for result in detailed_results if result.get("correct", False)
                )
                total_count = len(detailed_results)
                accuracy = correct_count / total_count if total_count > 0 else 0

                if model_name not in results:
                    results[model_name] = {"accuracy": accuracy}
                else:
                    results[model_name]["accuracy"] = accuracy

            if confusion_matrix_path.exists():
                with open(confusion_matrix_path, "r") as f:
                    confusion_matrix = json.load(f)
                    if model_name in results:
                        results[model_name]["confusion_matrix"] = confusion_matrix

        return results

    def load_segmentation_data(self, domain, task_type):
        """Load segmentation metrics for all models for a specific domain and task."""
        task_path = self.input_root / domain / task_type
        results = {}

        for model_name in os.listdir(task_path):
            model_path = task_path / model_name
            if not model_path.is_dir():
                continue

            results_path = model_path / "segmentation_results.json"

            if results_path.exists():
                with open(results_path, "r") as f:
                    data = json.load(f)

                # Extract aggregate metrics
                if "aggregate_metrics" in data:
                    results[model_name] = data["aggregate_metrics"]
                    # Add individual results if available
                    if "individual_results" in data:
                        results[model_name]["individual_results"] = data[
                            "individual_results"
                        ]
                else:
                    # Calculate aggregates from individual results
                    individual_results = data.get("individual_results", [])
                    if individual_results:
                        precision = np.mean(
                            [r["boundary_precision"] for r in individual_results]
                        )
                        recall = np.mean(
                            [r["boundary_recall"] for r in individual_results]
                        )
                        f1 = np.mean([r["boundary_f1"] for r in individual_results])
                        edit_distance = np.mean(
                            [r["edit_distance"] for r in individual_results]
                        )

                        results[model_name] = {
                            "boundary_precision": precision,
                            "boundary_recall": recall,
                            "boundary_f1": f1,
                            "edit_distance": edit_distance,
                            "individual_results": individual_results,
                        }

        return results

    def plot_classification_metrics(self, domain, task_type):
        """Create bar plots for precision, recall, F1, and accuracy across models."""
        data = self.load_classification_data(domain, task_type)
        if not data:
            print(f"No classification data found for {domain}/{task_type}")
            return

        models = list(data.keys())
        display_models = [self._get_display_name(m) for m in models]
        precision = [data[m]["precision"] for m in models]
        recall = [data[m]["recall"] for m in models]
        f1 = [data[m]["f1"] for m in models]
        accuracy = [data[m]["accuracy"] for m in models]

        # Create figure with constrained layout for better automatic spacing
        fig, ax = plt.subplots(figsize=(15, 10), constrained_layout=True)

        x = np.arange(len(models))
        width = 0.2  # Width of a single bar in a group

        model_bar_colors = [self.model_colors.get(m, "#333333") for m in models]

        # Plot bars with enhanced styling
        bars1 = ax.bar(
            x - 1.5 * width,
            precision,
            width,
            label="Precision",
            color=model_bar_colors,
            alpha=0.85,
            edgecolor="white",
            linewidth=1.0,
        )
        bars2 = ax.bar(
            x - 0.5 * width,
            recall,
            width,
            label="Recall",
            color=model_bar_colors,
            alpha=0.70,
            edgecolor="white",
            linewidth=1.0,
        )
        bars3 = ax.bar(
            x + 0.5 * width,
            f1,
            width,
            label="F1-Score",
            color=model_bar_colors,
            alpha=0.55,
            edgecolor="white",
            linewidth=1.0,
        )
        bars4 = ax.bar(
            x + 1.5 * width,
            accuracy,
            width,
            label="Accuracy",
            color=model_bar_colors,
            alpha=0.40,
            edgecolor="white",
            linewidth=1.0,
        )

        # Add data labels on top of bars with new styling
        def add_labels(bars_group):
            for bar in bars_group:
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height + 0.015,  # Adjusted offset
                    f"{height:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=14,
                    fontweight="bold",  # Updated font
                    bbox=dict(
                        facecolor="white",
                        alpha=0.7,
                        boxstyle="round,pad=0.2",
                        edgecolor="lightgray",
                    ),
                )

        add_labels(bars1)
        add_labels(bars2)
        add_labels(bars3)
        add_labels(bars4)

        # Add styled labels and title
        ax.set_ylabel("Score", fontweight="bold", fontsize=18)  # Increased font size

        display_domain = self._get_display_name(domain)
        display_task_type = self._get_display_name(task_type)
        title = f"Classification Performance Metrics ({display_domain})"
        ax.set_title(
            title, fontweight="bold", fontsize=24, pad=20
        )  # Increased font size

        ax.set_xticks([])  # Remove x-axis ticks
        ax.set_xticklabels([])  # Remove x-axis labels (model names)

        ax.set_ylim(0, 1.)  # Give more room for the data labels
        plt.setp(ax.get_yticklabels(), fontsize=18)  # Increased y-tick label size

        # Add subtle grid lines only on the y-axis, with color
        ax.grid(axis="y", linestyle="--", alpha=0.3, color="gray")

        # Add a light background color to emphasize the chart area
        ax.set_facecolor("#f9f9f9")  # Matched color

        # Style spines
        for spine_val in ax.spines.values():
            if spine_val.spine_type in [
                "left",
                "bottom",
            ]:  # Only style visible default spines
                spine_val.set_edgecolor("#dddddd")
                spine_val.set_linewidth(1.0)  # Ensure consistent linewidth
            else:
                spine_val.set_visible(False)  # Ensure others are off

        # Create figure-level legend for metrics (P, R, F1, Acc)
        metric_handles = [
            bars1[0],
            bars2[0],
            bars3[0],
            bars4[0],
        ]  # Representative bar for each metric
        metric_labels = ["Precision", "Recall", "F1-Score", "Accuracy"]
        fig.legend(
            metric_handles,
            metric_labels,
            loc="lower center",
            ncol=4,
            bbox_to_anchor=(0.5, -0.08),  # Adjusted position
            fontsize=18,
            frameon=True,
            framealpha=0.8,
            edgecolor="lightgray",
        )

        # Create figure-level legend for models
        model_legend_handles = [
            plt.Rectangle(
                (0, 0), 1, 1, color=self.model_colors.get(m, "#333333"), alpha=0.85
            )
            for m in models
        ]  # Use alpha from first bar group for consistency
        model_legend_labels = display_models
        fig.legend(
            model_legend_handles,
            model_legend_labels,
            loc="lower center",
            ncol=min(len(models), 5),
            bbox_to_anchor=(0.5, -0.16),  # Adjusted position below metric legend
            fontsize=18,
            frameon=True,
            framealpha=0.8,
            edgecolor="lightgray",
        )

        # Adjust layout to make space for legends at the bottom
        fig.subplots_adjust(bottom=0.25)  # Increased bottom margin

        # Save figure
        # plt.tight_layout() # Remove as constrained_layout is used
        fig_path = self.output_dir / f"{domain}_{task_type}_metrics.png"
        plt.savefig(
            fig_path, dpi=300, bbox_inches="tight"
        )  # Removed pad_inches, rely on tight_layout and subplots_adjust
        plt.close(fig)

        # Create heatmap for top confusion pairs
        for model_idx, model_name in enumerate(models):  # Use model_name from the loop
            if "confusion_matrix" in data[model_name]:
                self.plot_confusion_heatmap(
                    data[model_name], model_name, domain, task_type
                )

    def plot_confusion_heatmap(self, model_data, model_name, domain, task_type):
        """Create a heatmap of the confusion matrix for a specific model."""
        confusion_matrix = model_data["confusion_matrix"]

        # Convert confusion matrix to dataframe
        classes = list(confusion_matrix.keys())
        df = pd.DataFrame(0, index=classes, columns=classes)

        for true_class, pred_dict in confusion_matrix.items():
            for pred_class, count in pred_dict.items():
                df.at[true_class, pred_class] = count

        # Calculate row sums for percentages
        row_sums = df.sum(axis=1)
        df_percentages = df.div(row_sums, axis=0).fillna(0) * 100

        # Compute metrics for coloring
        total = df.values.sum()
        correct = df.values.diagonal().sum()
        accuracy = correct / total if total > 0 else 0

        # Create modern single-hue colormap optimized for confusion matrices
        colors = ['#f8f9fa', '#e9ecef', '#dee2e6', '#ced4da', '#adb5bd', 
                 '#6c757d', '#495057', '#343a40', '#212529', '#000000']
        modern_cmap = LinearSegmentedColormap.from_list("modern_confusion", colors, N=256)

        # Enhanced figure sizing based on number of classes
        base_size = max(10, len(classes) * 1.2)
        fig_width = min(base_size, 20)  # Cap at 20 inches
        fig_height = min(base_size * 0.9, 18)  # Cap at 18 inches
        
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        fig.patch.set_facecolor('white')

        # Create custom annotation matrix with both counts and percentages
        annotation_matrix = np.empty_like(df.values, dtype=object)
        for i in range(df.shape[0]):
            for j in range(df.shape[1]):
                count = df.values[i, j]
                percentage = df_percentages.values[i, j]
                if count == 0:
                    annotation_matrix[i, j] = ""
                else:
                    annotation_matrix[i, j] = f"{count:,}"

        # Plot the heatmap with enhanced styling
        im = ax.imshow(df_percentages.values, cmap=modern_cmap, aspect='equal', 
                      vmin=0, vmax=100, interpolation='nearest')

        # Add custom annotations with better formatting
        for i in range(df.shape[0]):
            for j in range(df.shape[1]):
                count = df.values[i, j]
                percentage = df_percentages.values[i, j]
                
                if count > 0:
                    # Determine text color based on background intensity
                    bg_intensity = percentage / 100.0
                    text_color = 'white' if bg_intensity > 0.6 else 'black'
                    
                    # Different font weights based on value significance
                    font_weight = 'bold' if count >= df.values.max() * 0.1 else 'normal'
                    font_size = 14 if count >= df.values.max() * 0.05 else 12
                    
                    # Format numbers with commas for readability
                    if count >= 1000:
                        count_text = f"{count:,}"
                    else:
                        count_text = str(count)
                    
                    text = f"{count_text}"
                    
                    ax.text(j, i, text, ha='center', va='center',
                           color=text_color, fontsize=font_size, fontweight=font_weight)

        # Enhanced colorbar with better styling
        cbar = plt.colorbar(im, ax=ax, shrink=0.8, aspect=30, pad=0.02)
        cbar.set_label('Percentage of True Class (%)', fontsize=16, fontweight='bold', 
                      labelpad=20, rotation=270, va='bottom')
        cbar.ax.tick_params(labelsize=14, width=1.5, length=5)
        
        # Style the colorbar frame
        cbar.outline.set_edgecolor('#cccccc')
        cbar.outline.set_linewidth(1.5)

        # Enhanced axis styling
        ax.set_xlim(-0.5, len(classes) - 0.5)
        ax.set_ylim(len(classes) - 0.5, -0.5)
        
        # Set ticks and labels with better formatting
        ax.set_xticks(range(len(classes)))
        ax.set_yticks(range(len(classes)))
        
        # Get display names for classes
        display_classes = [self._get_display_name(c) for c in classes]
        
        ax.set_xticklabels(display_classes, rotation=45, ha='right', fontsize=16, 
                          fontweight='medium')
        ax.set_yticklabels(display_classes, rotation=0, fontsize=16, 
                          fontweight='medium')

        # Add subtle grid lines
        for i in range(len(classes) + 1):
            ax.axhline(i - 0.5, color='white', linewidth=2)
            ax.axvline(i - 0.5, color='white', linewidth=2)

        # Enhanced labels and title
        display_model_name = self._get_display_name(model_name)
        display_domain = self._get_display_name(domain)
        display_task_type = self._get_display_name(task_type)
        
        ax.set_xlabel('Predicted Class', fontsize=18, fontweight='bold', 
                     labelpad=20, color='#2c3e50')
        ax.set_ylabel('True Class', fontsize=18, fontweight='bold', 
                     labelpad=20, color='#2c3e50')

        # Multi-line title with better typography
        title_line2 = f"{display_model_name} • {display_domain}"
        
        ax.text(0.5, 1.08, title_line2, transform=ax.transAxes, 
               fontsize=20, fontweight='medium', ha='center', 
               color='#7f8c8d', style='italic')

        # Enhanced border around the entire heatmap
        border_rect = plt.Rectangle((-0.5, -0.5), len(classes), len(classes), 
                                  fill=False, color='#34495e', linewidth=3)
        ax.add_patch(border_rect)

        # Remove default spines and ticks
        for spine in ax.spines.values():
            spine.set_visible(False)
        
        ax.tick_params(bottom=False, top=False, left=False, right=False)

        # Enhanced layout and save
        plt.tight_layout()
        fig_path = self.output_dir / f"{domain}_{task_type}_{model_name}_confusion.png"
        plt.savefig(fig_path, dpi=300, bbox_inches='tight', pad_inches=0.5, 
                   facecolor='white', edgecolor='none')
        plt.close(fig)

    def plot_segmentation_metrics(self, domain, task_type):
        """Create plots for segmentation metrics across models."""
        data = self.load_segmentation_data(domain, task_type)
        if not data:
            print(f"No segmentation data found for {domain}/{task_type}")
            return

        # Prepare data for plotting
        models = list(data.keys())
        display_models = [self._get_display_name(m) for m in models]
        precision = [data[m].get("boundary_precision", 0) for m in models]
        recall = [data[m].get("boundary_recall", 0) for m in models]
        f1 = [data[m].get("boundary_f1", 0) for m in models]

        # Create figure with constrained layout
        fig, ax = plt.subplots(figsize=(15, 10), constrained_layout=True)

        x = np.arange(len(models))
        width = 0.2  # Width of a single bar in a group

        model_bar_colors = [self.model_colors.get(m, "#333333") for m in models]

        # Plot bars with enhanced styling
        bars1 = ax.bar(
            x - 1.5 * width,
            precision,
            width,
            label="Precision",
            color=model_bar_colors,
            alpha=0.85,
            edgecolor="white",
            linewidth=1.0,
        )
        bars2 = ax.bar(
            x - 0.5 * width,
            recall,
            width,
            label="Recall",
            color=model_bar_colors,
            alpha=0.70,
            edgecolor="white",
            linewidth=1.0,
        )
        bars3 = ax.bar(
            x + 0.5 * width,
            f1,
            width,
            label="F1-Score",
            color=model_bar_colors,
            alpha=0.55,
            edgecolor="white",
            linewidth=1.0,
        )

        # Add data labels on top of bars
        def add_labels(bars_group):
            for bar in bars_group:
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height + 0.015,
                    f"{height:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=14,
                    fontweight="bold",
                    bbox=dict(
                        facecolor="white",
                        alpha=0.7,
                        boxstyle="round,pad=0.2",
                        edgecolor="lightgray",
                    ),
                )

        add_labels(bars1)
        add_labels(bars2)
        add_labels(bars3)

        # Add styled labels and title
        ax.set_ylabel("Score", fontweight="bold", fontsize=18)

        display_domain = self._get_display_name(domain)
        display_task_type = self._get_display_name(task_type)
        title = f"Segmentation Performance Metrics ({display_domain})"
        ax.set_title(title, fontweight="bold", fontsize=22, pad=20)

        ax.set_xticks([])  # Remove x-axis ticks
        ax.set_xticklabels([])  # Remove x-axis labels (model names)

        ax.set_ylim(0, 1.)  # Give more room for the data labels
        plt.setp(ax.get_yticklabels(), fontsize=18)

        # Add subtle grid lines only on the y-axis
        ax.grid(axis="y", linestyle="--", alpha=0.3, color="gray")

        # Add a light background color
        ax.set_facecolor("#f9f9f9")

        # Style spines
        for spine_val in ax.spines.values():
            if spine_val.spine_type in ["left", "bottom"]:
                spine_val.set_edgecolor("#dddddd")
                spine_val.set_linewidth(1.0)
            else:
                spine_val.set_visible(False)

        # Create figure-level legend for metrics
        metric_handles = [bars1[0], bars2[0], bars3[0]]
        metric_labels = ["Precision", "Recall", "F1-Score"]
        fig.legend(
            metric_handles,
            metric_labels,
            loc="lower center",
            ncol=3,
            bbox_to_anchor=(0.5, -0.08),
            fontsize=18,
            frameon=True,
            framealpha=0.8,
            edgecolor="lightgray",
        )

        # Create figure-level legend for models
        model_legend_handles = [
            plt.Rectangle(
                (0, 0), 1, 1, color=self.model_colors.get(m, "#333333"), alpha=0.85
            )
            for m in models
        ]
        model_legend_labels = display_models
        fig.legend(
            model_legend_handles,
            model_legend_labels,
            loc="lower center",
            ncol=min(len(models), 5),
            bbox_to_anchor=(0.5, -0.16),
            fontsize=18,
            frameon=True,
            framealpha=0.8,
            edgecolor="lightgray",
        )

        # Adjust layout to make space for legends at the bottom
        fig.subplots_adjust(bottom=0.25)

        # Save figure
        fig_path = self.output_dir / f"{domain}_{task_type}_boundary_metrics.png"
        plt.savefig(fig_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

    def plot_detailed_accuracy(self, domain, task_type):
        """Create plots showing accuracy information from detailed_results.json."""
        task_path = self.input_root / domain / task_type
        model_results = {}

        for model_name in os.listdir(task_path):
            model_path = task_path / model_name
            if not model_path.is_dir():
                continue

            detailed_results_path = model_path / "detailed_results.json"

            if detailed_results_path.exists():
                with open(detailed_results_path, "r") as f:
                    detailed_results = json.load(f)

                # Count correct/incorrect predictions
                correct_count = sum(
                    1 for result in detailed_results if result.get("correct", False)
                )
                incorrect_count = len(detailed_results) - correct_count

                # Save results
                model_results[model_name] = {
                    "correct": correct_count,
                    "incorrect": incorrect_count,
                    "total": len(detailed_results),
                    "accuracy": (
                        correct_count / len(detailed_results)
                        if len(detailed_results) > 0
                        else 0
                    ),
                }

                # Count predictions per class
                class_counts = {}
                for result in detailed_results:
                    true_class = result.get("true_class", "Unknown")
                    correct = result.get("correct", False)

                    if true_class not in class_counts:
                        class_counts[true_class] = {
                            "correct": 0,
                            "incorrect": 0,
                            "total": 0,
                        }

                    class_counts[true_class]["total"] += 1
                    if correct:
                        class_counts[true_class]["correct"] += 1
                    else:
                        class_counts[true_class]["incorrect"] += 1

                model_results[model_name]["class_counts"] = class_counts

        if not model_results:
            print(f"No detailed results found for {domain}/{task_type}")
            return

        # Create per-class accuracy plots for each model
        for model_name, data in model_results.items():
            if "class_counts" in data:
                self._plot_class_accuracy(
                    data["class_counts"], model_name, domain, task_type
                )

    def _plot_class_accuracy(self, class_counts, model_name, domain, task_type):
        """Create a modern, professional plot showing per-class accuracy for a model."""
        classes = list(class_counts.keys())
        correct = [class_counts[c]["correct"] for c in classes]
        incorrect = [class_counts[c]["incorrect"] for c in classes]
        accuracy = [
            (
                class_counts[c]["correct"] / class_counts[c]["total"]
                if class_counts[c]["total"] > 0
                else 0
            )
            for c in classes
        ]

        # Get display names
        display_model_name = self._get_display_name(model_name)
        display_domain = self._get_display_name(domain)
        display_task_type = self._get_display_name(task_type)
        display_classes = [self._get_display_name(c) for c in classes]

        # Sort classes by accuracy for better visualization
        sorted_indices = np.argsort(accuracy)
        classes = [classes[i] for i in sorted_indices]
        correct = [correct[i] for i in sorted_indices]
        incorrect = [incorrect[i] for i in sorted_indices]
        accuracy = [accuracy[i] for i in sorted_indices]
        sorted_display_classes = [display_classes[i] for i in sorted_indices]

        # Enhanced figure sizing based on number of classes
        base_height = max(8, len(classes) * 0.7)
        fig_height = min(base_height, 16)  # Cap at 16 inches
        fig, ax = plt.subplots(figsize=(14, fig_height), constrained_layout=True)
        fig.patch.set_facecolor('white')

        # Create modern color scheme with gradient based on accuracy
        base_color = self.model_colors.get(model_name, "#333333")
        # Convert hex to RGB for gradient calculations
        import matplotlib.colors as mcolors
        base_rgb = mcolors.hex2color(base_color)
        
        # Create colors with varying intensity based on accuracy
        bar_colors = []
        for acc in accuracy:
            # Higher accuracy = more saturated color
            intensity = 0.3 + (acc * 0.7)  # Range from 0.3 to 1.0
            color_rgb = tuple(c * intensity + (1 - intensity) * 0.95 for c in base_rgb)
            bar_colors.append(color_rgb)

        # Plot enhanced horizontal bars
        y = np.arange(len(classes))
        bars = ax.barh(
            y, 
            accuracy, 
            height=0.7, 
            color=bar_colors,
            alpha=0.9,
            edgecolor='white',
            linewidth=1.5
        )

        # Add gradient effect by overlaying semi-transparent bars
        for i, (bar, acc) in enumerate(zip(bars, accuracy)):
            # Add a subtle gradient overlay
            gradient_alpha = 0.2 + (acc * 0.3)
            ax.barh(
                y[i], 
                acc, 
                height=0.7, 
                color=base_color,
                alpha=gradient_alpha,
                edgecolor='none'
            )

        # Enhanced data labels with styled boxes
        for i, (acc, c, ic) in enumerate(zip(accuracy, correct, incorrect)):
            total = c + ic
            shift = 0.02 if acc > 0.15 else 0.1
            # Accuracy percentage on the right
            ax.text(
                acc + shift, 
                i, 
                f"{acc:.1%}", 
                va="center", 
                ha="left",
                fontsize=12,
                fontweight="bold",
                bbox=dict(
                    facecolor="white",
                    alpha=0.9,
                    boxstyle="round,pad=0.3",
                    edgecolor="lightgray",
                    linewidth=0.5
                )
            )
            
            # Count information on the left (inside bar if space allows)
            count_text = f"{c}/{total}"
            if acc > 0.15:  # If bar is wide enough, put text inside
                text_x = acc * 0.05
                text_color = "white"
                bbox_props = dict(
                    facecolor=base_color,
                    alpha=0.8,
                    boxstyle="round,pad=0.2",
                    edgecolor="none"
                )
            else:  # Otherwise put it outside
                text_x = 0.02
                text_color = "#2c3e50"
                bbox_props = dict(
                    facecolor="white",
                    alpha=0.9,
                    boxstyle="round,pad=0.2",
                    edgecolor="lightgray",
                    linewidth=0.5
                )
                
            ax.text(
                text_x,
                i,
                count_text,
                va="center",
                ha="left",
                fontsize=10,
                fontweight="bold",
                color=text_color,
                bbox=bbox_props
            )

        # Enhanced styling
        ax.set_facecolor("#f8f9fa")
        
        # Enhanced grid
        ax.grid(axis="x", linestyle="--", alpha=0.4, color="#6c757d", linewidth=0.8)
        ax.set_axisbelow(True)

        # Enhanced axis styling
        ax.set_xlabel("Accuracy", fontsize=16, fontweight="bold", color="#2c3e50", labelpad=15)
        ax.set_ylabel("Class", fontsize=16, fontweight="bold", color="#2c3e50", labelpad=15)

        # Enhanced title with better typography
        title_line1 = f"Per-Class Accuracy Analysis"
        title_line2 = f"{display_model_name} • {display_domain}"
        # ax.set_title(title_line1, fontsize=22, fontweight="bold", pad=15, color="#2c3e50")
        ax.text(0.5, 1.08, title_line2, transform=ax.transAxes, 
               fontsize=20, fontweight="medium", ha="center", 
               color="#7f8c8d", style="italic")

        # Enhanced y-axis
        ax.set_yticks(y)
        ax.set_yticklabels(sorted_display_classes, fontsize=14, fontweight="medium")
        plt.setp(ax.get_yticklabels(), color="#2c3e50")
        
        # Enhanced x-axis
        ax.set_xlim(0, 1.15)  # Extra space for labels
        ax.set_xticks(np.arange(0, 1.1, 0.2))
        ax.set_xticklabels([f"{x:.0%}" for x in np.arange(0, 1.1, 0.2)], 
                          fontsize=12, fontweight="medium")
        plt.setp(ax.get_xticklabels(), color="#2c3e50")

        # Enhanced spines
        for spine_name, spine in ax.spines.items():
            if spine_name in ['left', 'bottom']:
                spine.set_edgecolor("#dee2e6")
                spine.set_linewidth(1.5)
            else:
                spine.set_visible(False)

        # Add subtle border around the plot area
        border_rect = plt.Rectangle((0, -0.5), 1.0, len(classes), 
                                  fill=False, color="#dee2e6", linewidth=2, alpha=0.7)
        ax.add_patch(border_rect)

        # Add performance summary in a text box
        avg_accuracy = np.mean(accuracy)
        min_accuracy = np.min(accuracy)
        max_accuracy = np.max(accuracy)
        
        summary_text = f"Average: {avg_accuracy:.1%}"
        ax.text(0.98, 0.98, summary_text, transform=ax.transAxes,
               fontsize=11, fontweight="medium",
               verticalalignment="top", horizontalalignment="right",
               bbox=dict(facecolor="white", alpha=0.9, boxstyle="round,pad=0.5",
                        edgecolor="#dee2e6", linewidth=1))

        # Enhanced save options
        fig_path = (
            self.output_dir / f"{domain}_{task_type}_{model_name}_class_accuracy.png"
        )
        plt.savefig(fig_path, dpi=300, bbox_inches='tight', pad_inches=0.3,
                   facecolor='white', edgecolor='none')
        plt.close(fig)

    def plot_all_domains_accuracy_comparison(self):
        """Create a single figure with subplots comparing accuracy across models for each domain."""

        # Filter domains to include only those with classification tasks
        classification_domains = {}
        for domain, task_types in self.tasks.items():
            classification_task_types_in_domain = [
                tt for tt in task_types if "classification" in tt
            ]
            if classification_task_types_in_domain:
                classification_domains[domain] = classification_task_types_in_domain

        if not classification_domains:
            print(
                "No domains with classification tasks found to generate accuracy comparison plot."
            )
            return

        num_domains = len(classification_domains)

        # Determine subplot layout (aim for 2 columns, adjust rows)
        ncols = 3
        nrows = 1

        fig, axes = plt.subplots(
            nrows=nrows,
            ncols=ncols,
            figsize=(7 * ncols, 6 * nrows),
            squeeze=False,
            constrained_layout=True,
        )
        axes_flat = axes.flatten()  # Flatten for easy iteration
        all_models_in_figure_accuracy = (
            set()
        )  # To collect unique models for the figure legend

        domain_idx = 0
        for domain, classification_task_types in classification_domains.items():
            ax = axes_flat[domain_idx]
            domain_model_results = {}

            for task_type in classification_task_types:
                task_path = self.input_root / domain / task_type
                if not task_path.exists():
                    continue

                for model_name_dir in os.listdir(task_path):
                    model_path = task_path / model_name_dir
                    if not model_path.is_dir():
                        continue

                    model_name = (
                        model_name_dir  # model_name_dir is already just the model name
                    )

                    detailed_results_path = model_path / "detailed_results.json"
                    if detailed_results_path.exists():
                        with open(detailed_results_path, "r") as f:
                            detailed_results = json.load(f)

                        correct_count = sum(
                            1
                            for result in detailed_results
                            if result.get("correct", False)
                        )
                        total_count = len(detailed_results)
                        incorrect_count = total_count - correct_count

                        if model_name not in domain_model_results:
                            domain_model_results[model_name] = {
                                "correct": 0,
                                "incorrect": 0,
                                "total": 0,
                            }

                        domain_model_results[model_name]["correct"] += correct_count
                        domain_model_results[model_name]["incorrect"] += incorrect_count
                        domain_model_results[model_name]["total"] += total_count

            if not domain_model_results:
                ax.set_title(
                    f"{self._get_display_name(domain)}\\n(No data)",
                    fontweight="bold",
                    fontsize=14,
                )
                ax.text(
                    0.5,
                    0.5,
                    "No detailed results found for this domain.",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                    fontsize=10,
                )
                ax.set_xticks([])
                ax.set_yticks([])
                domain_idx += 1
                continue

            models = list(domain_model_results.keys())
            correct_counts = [domain_model_results[m]["correct"] for m in models]
            incorrect_counts = [domain_model_results[m]["incorrect"] for m in models]

            accuracy_values = []
            for m in models:
                all_models_in_figure_accuracy.add(m)  # Collect model name
                total = domain_model_results[m]["total"]
                accuracy_values.append(
                    domain_model_results[m]["correct"] / total if total > 0 else 0
                )

            width = 0.6
            x = np.arange(len(models))

            # Plot stacked bars
            ax.bar(
                x,
                correct_counts,
                width,
                label="Correct",
                color=[self.model_colors.get(m, "#333333") for m in models],
                alpha=0.85,
                edgecolor="white",
                linewidth=0.5,
            )
            ax.bar(
                x,
                incorrect_counts,
                width,
                bottom=correct_counts,
                label="Incorrect",
                color=[self.model_colors.get(m, "#333333") for m in models],
                alpha=0.4,
                edgecolor="white",
                linewidth=0.5,
                hatch="///",
            )

            # Add accuracy percentage labels
            for i, (c, ic) in enumerate(zip(correct_counts, incorrect_counts)):
                total_preds = c + ic
                acc = accuracy_values[i]
                ax.text(
                    i,
                    total_preds + (total_preds * 0.03),
                    f"{acc:.1%}",
                    ha="center",
                    va="bottom",
                    fontsize=20,
                    fontweight="bold",
                    bbox=dict(
                        facecolor="white",
                        alpha=0.7,
                        boxstyle="round,pad=0.2",
                        edgecolor="lightgray",
                    ),
                )

            if domain == "finance":
                ax.set_ylabel("Number of Predictions", fontweight="bold", fontsize=24)
            else:
                ax.set_ylabel("", fontweight="bold", fontsize=24)

            if domain == "hr":
                ax.set_title("HR", fontweight="bold", fontsize=26, pad=15)
            else:
                ax.set_title(
                    f"{self._get_display_name(domain)}",
                    fontweight="bold",
                    fontsize=26,
                    pad=15,
                )
            ax.set_xticks(x)
            # ax.set_xticklabels([self._get_display_name(m) for m in models], rotation=45, ha='right', fontsize=24) # Remove this
            ax.set_xticklabels([])  # Remove x-tick labels
            ax.set_xticks([])  # Remove x-ticks

            # Increase font size of y-axis tick labels
            plt.setp(ax.get_yticklabels(), fontsize=24)

            ax.grid(axis="y", linestyle="--", alpha=0.3, color="gray")
            ax.set_facecolor("#f9f9f9")

            y_max_subplot = (
                max([c + ic for c, ic in zip(correct_counts, incorrect_counts)])
                if correct_counts
                else 1
            )
            ax.set_ylim(0, y_max_subplot * 1.20)

            for spine_val in ax.spines.values():
                spine_val.set_edgecolor("#dddddd")

            domain_idx += 1

        # Hide any unused subplots
        for i in range(domain_idx, nrows * ncols):
            fig.delaxes(axes_flat[i])

        # Create a single legend for the whole figure if there are plotted results
        if any(
            domain_model_results
            for _, domain_model_results in locals()
            .get("classification_domains", {})
            .items()
            if domain_model_results
        ):
            # Custom handles and labels for fill types (Correct/Incorrect)
            labels_fill_type = ["Correct", "Incorrect"]
            handles_fill_type = [
                plt.Rectangle(
                    (0, 0),
                    1,
                    1,
                    color="gray",
                    alpha=0.85,
                    edgecolor="white",
                    linewidth=0.5,
                ),
                plt.Rectangle(
                    (0, 0),
                    1,
                    1,
                    facecolor="gray",
                    alpha=0.6,
                    hatch="///",
                    edgecolor="white",
                    linewidth=0.5,
                ),
            ]
            fig.legend(
                handles_fill_type,
                labels_fill_type,
                loc="lower center",
                ncol=2,
                bbox_to_anchor=(0.5, -0.25),
                frameon=True,
                fontsize=22,
            )

        if all_models_in_figure_accuracy:
            sorted_models_for_legend = sorted(list(all_models_in_figure_accuracy))
            legend_labels = [
                self._get_display_name(m) for m in sorted_models_for_legend
            ]
            # Create handles for the accuracy plot legend (using model colors)
            legend_handles = [
                plt.Rectangle(
                    (0, 0), 1, 1, color=self.model_colors.get(m, "#333333"), alpha=0.85
                )
                for m in sorted_models_for_legend
            ]

            # Add Correct/Incorrect to the legend manually if needed, or rely on subplot legend if kept
            # For now, just model names as requested.

            fig.legend(
                legend_handles,
                legend_labels,
                loc="lower center",
                ncol=min(5, len(sorted_models_for_legend)),  # Match previous adjustment
                bbox_to_anchor=(0.5, -0.12),  # Match previous adjustment
                fontsize=22,
                frameon=True,
                framealpha=0.8,
                edgecolor="lightgray",
            )

            fig.subplots_adjust(bottom=0.2)  # Make more space if ncol is larger

        # plt.tight_layout(rect=[0, 0.05, 1, 0.97]) # constrained_layout should handle this better

        fig_path = self.output_dir / "domain_detailed_accuracy_subplots.pdf"
        plt.savefig(fig_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

    def plot_all_domains_segmentation_f1_comparison(self):
        """Create a single figure with subplots comparing Boundary F1 scores across models for each domain."""

        segmentation_domains = {}
        for domain, task_types in self.tasks.items():
            segmentation_task_types_in_domain = [
                tt for tt in task_types if "segmentation" in tt
            ]
            if segmentation_task_types_in_domain:
                segmentation_domains[domain] = segmentation_task_types_in_domain

        if not segmentation_domains:
            print(
                "No domains with segmentation tasks found to generate F1 comparison plot."
            )
            return

        num_domains = len(segmentation_domains)

        ncols = 3
        nrows = (num_domains + ncols - 1) // ncols

        fig, axes = plt.subplots(
            nrows=nrows,
            ncols=ncols,
            figsize=(7 * ncols, 6 * nrows),
            squeeze=False,
            constrained_layout=True,
        )
        axes_flat = axes.flatten()

        domain_idx = 0
        all_models_in_figure = (
            set()
        )  # To collect all unique models for the figure legend

        for domain, segmentation_task_types in segmentation_domains.items():
            ax = axes_flat[domain_idx]

            # Stores final {model_name: avg_f1} for the current domain
            domain_model_avg_f1s = {}
            # Helper dict to collect all F1s for a model before averaging
            # {model_name: [f1_score_task1, f1_score_task2, ...]}
            temp_model_f1s_for_domain = {}

            for task_type in segmentation_task_types:
                segmentation_data_for_task = self.load_segmentation_data(
                    domain, task_type
                )

                for model_name, metrics in segmentation_data_for_task.items():
                    if model_name not in temp_model_f1s_for_domain:
                        temp_model_f1s_for_domain[model_name] = []

                    f1_score = metrics.get("boundary_f1")
                    if f1_score is not None:
                        temp_model_f1s_for_domain[model_name].append(f1_score)

            # Calculate average F1 for each model in the domain
            for model_name, f1_list in temp_model_f1s_for_domain.items():
                if f1_list:
                    domain_model_avg_f1s[model_name] = np.mean(f1_list)
                    all_models_in_figure.add(
                        model_name
                    )  # Add model to the set for figure legend

            if not domain_model_avg_f1s:
                ax.set_title(
                    f"{self._get_display_name(domain)}\\n(No data)",
                    fontweight="bold",
                    fontsize=14,
                )
                ax.text(
                    0.5,
                    0.5,
                    "No segmentation F1 data found for this domain.",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                    fontsize=10,
                )
                ax.set_xticks([])
                ax.set_yticks([])
                domain_idx += 1
                continue

            models = sorted(list(domain_model_avg_f1s.keys()))
            f1_values = [domain_model_avg_f1s[m] for m in models]

            width = 0.6
            x = np.arange(len(models))

            ax.bar(
                x,
                f1_values,
                width,
                color=[self.model_colors.get(m, "#333333") for m in models],
                alpha=0.85,
                edgecolor="white",
                linewidth=0.5,
            )

            for i, f1_val in enumerate(f1_values):
                ax.text(
                    i,
                    f1_val + (0.02 * 1.0),
                    f"{f1_val:.2f}",  # Use 1.0 as reference for offset
                    ha="center",
                    va="bottom",
                    fontsize=20,
                    fontweight="bold",
                    bbox=dict(
                        facecolor="white",
                        alpha=0.7,
                        boxstyle="round,pad=0.2",
                        edgecolor="lightgray",
                    ),
                )

            if domain == "finance":
                ax.set_ylabel("Boundary F1 Score", fontweight="bold", fontsize=24)
            else:
                ax.set_ylabel("", fontweight="bold", fontsize=24)

            if domain == "hr":
                ax.set_title("HR", fontweight="bold", fontsize=26, pad=15)
            else:
                ax.set_title(
                    f"{self._get_display_name(domain)}",
                    fontweight="bold",
                    fontsize=26,
                    pad=15,
                )
            ax.set_xticks(x)
            # Remove x-axis tick labels
            ax.set_xticklabels([])
            ax.set_xticks([])

            # Add a legend instead
            # REMOVE INDIVIDUAL SUBPLOT LEGEND
            # legend_labels = [self._get_display_name(m) for m in models]
            # legend_handles = [plt.Rectangle((0,0), 1, 1, color=self.model_colors.get(m, '#333333'), alpha=0.85) for m in models]
            # ax.legend(legend_handles, legend_labels, loc='upper center', bbox_to_anchor=(0.5, -0.15),
            #          ncol=min(3, len(models)), fontsize=20, frameon=True, framealpha=0.8,
            #          edgecolor='lightgray')
            # Increase font size of y-axis tick labels
            plt.setp(ax.get_yticklabels(), fontsize=24)

            ax.grid(axis="y", linestyle="--", alpha=0.3, color="gray")
            ax.set_facecolor("#f9f9f9")

            ax.set_ylim(0, 1.05)  # F1 scores are 0-1, allow space for text

            for spine_val in ax.spines.values():
                spine_val.set_edgecolor("#dddddd")

            domain_idx += 1

        for i in range(domain_idx, nrows * ncols):
            fig.delaxes(axes_flat[i])

        # Create a single legend for the whole figure
        if all_models_in_figure:
            sorted_models_for_legend = sorted(list(all_models_in_figure))
            legend_labels = [
                self._get_display_name(m) for m in sorted_models_for_legend
            ]
            legend_handles = [
                plt.Rectangle(
                    (0, 0), 1, 1, color=self.model_colors.get(m, "#333333"), alpha=0.85
                )
                for m in sorted_models_for_legend
            ]
            fig.legend(
                legend_handles,
                legend_labels,
                loc="lower center",
                ncol=min(4, len(sorted_models_for_legend)),
                bbox_to_anchor=(0.5, -0.15),  # Adjust y-offset to place below subplots
                fontsize=22,
                frameon=True,
                framealpha=0.8,
                edgecolor="lightgray",
            )

            fig.subplots_adjust(
                bottom=0.15
            )  # Adjust bottom to make space for figure legend

        fig_path = self.output_dir / "domain_segmentation_f1_subplots.pdf"
        plt.savefig(fig_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

    def generate_all_plots(self):
        """Generate all plots for all domains and tasks."""
        for domain in self.tasks:
            for task_type in self.tasks[domain]:
                if "classification" in task_type:
                    self.plot_classification_metrics(domain, task_type)
                    self.plot_detailed_accuracy(domain, task_type)
                elif "segmentation" in task_type:
                    self.plot_segmentation_metrics(domain, task_type)

        # Generate overall accuracy comparison plot
        self.plot_all_domains_accuracy_comparison()

        # Generate overall segmentation F1 comparison plot
        self.plot_all_domains_segmentation_f1_comparison()

        print(f"All plots saved to {self.output_dir}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate plots from model evaluation results"
    )
    parser.add_argument(
        "--input_root",
        type=str,
        default="outputs",
        help="Root directory containing the evaluation results",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="figures",
        help="Output directory for the plots",
    )
    args = parser.parse_args()

    generator = PlotGenerator(args.input_root, args.output_dir)
    generator.generate_all_plots()
