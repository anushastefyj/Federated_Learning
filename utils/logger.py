import os
import csv
from datetime import datetime

class CSVLogger:
    """
    Simple CSV logger for tracking experiment results.
    """
    def __init__(self, output_dir: str, filename: str = "results.csv"):
        self.filepath = os.path.join(output_dir, filename)
        self.file_exists = os.path.exists(self.filepath)

    def log(self, metrics_dict: dict):
        """
        Appends a dictionary of metrics as a row in the CSV.
        """
        # Add timestamp
        metrics_dict["timestamp"] = datetime.now().isoformat()
        
        with open(self.filepath, mode='a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=metrics_dict.keys())
            if not self.file_exists:
                writer.writeheader()
                self.file_exists = True
            writer.writerow(metrics_dict)
