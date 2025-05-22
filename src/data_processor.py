# ============================================================================
# ENTROPHY © 2025 by Workfabric
# Licensed under Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International
# https://creativecommons.org/licenses/by-nc-sa/4.0/
# ============================================================================

import warnings
import pandas as pd
import numpy as np
from typing import Dict, List, Any
from sklearn.preprocessing import LabelEncoder

class WorkflowDataProcessor:
    """
    Main data processing class for workflow intelligence tasks.
    Handles loading, preprocessing, and tokenization of workflow interaction data.
    """
    
    def __init__(self, data_path: str):
        """
        Initialize the data processor.
        
        Args:
            data_path: Path to the JSON data file
        """
        self.data_path = data_path
        
        # This will be populated once load_data is called
        self.data_df = None
        self.process_names = None
        self.label_encoder = LabelEncoder()
        
    def load_data(self) -> None:
        """
        Load data from the specified JSON file into a pandas DataFrame and preprocess.
        """
        print(f"Loading data from {self.data_path}")
        
        # Load data into a pandas DataFrame
        self.data_df = pd.read_json(self.data_path)

        # Convert time_stamp to datetime objects if not already, handling potential errors
        self.data_df["time_stamp"] = pd.to_datetime(self.data_df["time_stamp"], errors='coerce')
        
        # Sort events by process_instance_uuid and then by timestamp
        self.data_df.sort_values(by=["process_instance_uuid", "time_stamp"], inplace=True)
        
        # Extract unique process names and fit the label encoder
        self.process_names = self.data_df["process_name"].unique().tolist()
        self.label_encoder.fit(self.process_names)

        if self.data_df is not None:
            print(f"Loaded {self.data_df['process_instance_uuid'].nunique()} process instances with {len(self.process_names)} distinct process types")
        else:
            print("Failed to load data into DataFrame.")
    
    def _create_event_description(self, event: Dict[str, Any]) -> str:
        """
        Create a textual description of a single event.
        
        Args:
            event: Single interactionevent dictionary from the JSON data
            
        Returns:
            String description of the event
        """
        application = event.get("application_name", "")
        description = event.get("description")
        screen_name = event.get("screen_name", "")
        return f"{description} in the screen {screen_name} of the application {application}"        
    
    def prepare_workflow_classification_data(self) -> List[Dict]:
        """
        Prepare data for workflow classification task.

        Returns:
            List of dictionaries with workflow_id, event_descriptions, process_name, label_id
        """
        if self.data_df is None:
            self.load_data()
            
        if self.data_df is None or self.data_df.empty:
            print("Error: Data not loaded or empty. Cannot prepare classification data.")
            return []

        # Create a dataset of (workflow_sequence, label) pairs
        dataset = []
        # Group by 'process_instance_uuid' which represents a workflow
        for workflow_id, events_df in self.data_df.groupby("process_instance_uuid"):
            if events_df.empty:
                warnings.warn(f"Warning: Workflow {workflow_id} is empty. Skipping.")
                continue
                
            # Create sequence of event descriptions from the DataFrame rows
            event_descriptions = [self._create_event_description(row.to_dict()) for _, row in events_df.iterrows()]
            
            # Get the process name (label) from the first event of the workflow
            process_name = events_df["process_name"].iloc[0]
                
            # Encode the label
            label_id = self.label_encoder.transform([process_name])[0]

            dataset.append({
                "workflow_id": str(workflow_id),
                "event_descriptions": event_descriptions,
                "process_name": process_name,
                "label_id": label_id
            })
            
        return dataset
    
    def prepare_workflow_segmentation_data(self, min_workflows=2, max_workflows=4, random_state=2404, num_samples=100):
        """
        Create segmentation datasets by concatenating multiple workflows into longer sequences.
        
        Args:
            min_workflows: Minimum number of workflows to concatenate
            max_workflows: Maximum number of workflows to concatenate
            random_state: Random seed for reproducibility
            num_samples: Number of concatenated workflows to create
        Returns:
            List of concatenated workflows
        """
        if self.data_df is None:
            self.load_data()
            
        np.random.seed(random_state)
        
        # Group events by workflow ID
        workflow_events = {}
        for workflow_id, events_df in self.data_df.groupby("process_instance_uuid"):
            event_descriptions = [self._create_event_description(row.to_dict()) for _, row in events_df.iterrows()]
            process_name = events_df["process_name"].iloc[0]
            
            workflow_events[workflow_id] = {
                "events": event_descriptions,
                "process_name": process_name
            }
        
        # Convert to list for easier sampling
        workflow_list = list(workflow_events.items())
        concatenated_workflows = []
        
        for i in range(num_samples):
            # Randomly choose how many workflows to concatenate
            num_workflows = np.random.randint(min_workflows, max_workflows + 1)
            
            # Randomly select workflows (without replacement)
            selected_indices = np.random.choice(
                len(workflow_list), size=min(num_workflows, len(workflow_list)), replace=False
            )
            selected_workflows = [workflow_list[idx] for idx in selected_indices]
            
            # Construct the challenge
            concatenated_events = []
            segment_boundaries = []
            workflow_types = []
            workflow_ids = []
            
            current_position = 0
            for workflow_id, workflow_data in selected_workflows:
                events = workflow_data["events"]
                concatenated_events.extend(events)
                
                # Record segment boundary (start position of this workflow)
                segment_boundaries.append(current_position)
                current_position += len(events)
                
                workflow_types.append(workflow_data["process_name"])
                workflow_ids.append(workflow_id)
            
            # Add end boundary
            segment_boundaries.append(current_position)
            
            concatenated_workflows.append({
                "sequence_id": f"sequence_{i}",
                "sequence": concatenated_events,
                "segment_boundaries": segment_boundaries,
                "workflow_types": workflow_types,
                "workflow_ids": workflow_ids,
                "num_workflows": len(selected_workflows)
            })
        
        return concatenated_workflows