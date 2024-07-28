import pandas as pd
import numpy as np
from typing import Dict
from datetime import datetime, time

class DataMerger:
    def __init__(self, datasets: Dict[str, str]):
        """
        Initialize the DataMerger with a dictionary of dataset paths.

        Args:
            datasets (Dict[str, str]): A dictionary where keys are dataset names
                                       and values are paths to the dataset files.
        """
        self.datasets = datasets
        self.merged_data = None
        self.load_and_merge_datasets()

    def load_dataset(self, name: str, path: str) -> pd.DataFrame:
        """
        Load a dataset from a file and perform initial preprocessing.

        Args:
            name (str): The name of the dataset.
            path (str): The path to the dataset file.

        Returns:
            pd.DataFrame: The loaded and preprocessed dataset.
        """
        try:
            df = pd.read_csv(path)
        except FileNotFoundError:
            raise ValueError(f"Dataset file not found: {path}")
        except pd.errors.EmptyDataError:
            raise ValueError(f"Dataset file is empty: {path}")

        if 'date' not in df.columns:
            raise ValueError(f"Dataset {name} is missing 'date' column")

        df['date'] = self.normalize_date(df['date'])
        df.columns = [col.strip().lower().replace(" ", "_").replace("®", "").replace('"', '') for col in df.columns]

        for col in df.columns:
            if col != 'date':
                df[col] = self.convert_to_numeric(df[col])

        return df

    def normalize_date(self, date_series: pd.Series) -> pd.Series:
        """
        Normalize date/time data to start of day.

        Args:
            date_series (pd.Series): Series containing date/time data.

        Returns:
            pd.Series: Normalized date series.
        """
        try:
            date_series = pd.to_datetime(date_series, errors='coerce')
            return date_series.dt.normalize()
        except Exception as e:
            raise ValueError(f"Unable to parse dates: {str(e)}")

    def convert_to_numeric(self, series: pd.Series) -> pd.Series:
        """
        Convert a series to numeric, handling percentage values.

        Args:
            series (pd.Series): The series to convert.

        Returns:
            pd.Series: The converted numeric series.
        """
        series = pd.to_numeric(series.astype(str).str.replace(',', '').str.rstrip('%'), errors='coerce')
        if series.dtype == 'object':
            series = series.astype(float) / 100
        return series

    def load_and_merge_datasets(self):
        """
        Load all datasets, merge them, and perform final preprocessing.
        """
        merged_df = pd.DataFrame()  # Initialize with an empty DataFrame instead of None
        for name, path in self.datasets.items():
            df = self.load_dataset(name, path)
            if merged_df.empty:
                merged_df = df
            else:
                merged_df = pd.merge(merged_df, df, on='date', how='outer', suffixes=('', f'_{name}'))

        # Ensure OHLC data is present
        required_columns = ['open', 'high', 'low', 'close']
        missing_columns = [col for col in required_columns if col not in merged_df.columns]
        if missing_columns:
            raise ValueError(f"Missing required OHLC columns: {', '.join(missing_columns)}")

        # Sort the merged DataFrame by date, interpolate missing values, and drop rows with any remaining NaN values
        merged_df = merged_df.sort_values('date').interpolate(method='linear').dropna()

        self.merged_data = merged_df

    def get_merged_data(self) -> pd.DataFrame:
        """
        Get the merged and processed dataset.

        Returns:
            pd.DataFrame: The final merged and processed dataset.
        """
        if self.merged_data is None:
            raise ValueError("Datasets have not been merged yet.")
        return self.merged_data

    def get_row_count(self) -> int:
        """
        Get the total number of rows in the complete merged dataset.

        Returns:
            int: The number of rows in the merged dataset.
        """
        if self.merged_data is None:
            raise ValueError("Datasets have not been merged yet.")
        return len(self.merged_data)

    def export_to_csv(self, filepath: str):
        """
        Export the merged dataset to a CSV file.

        Args:
            filepath (str): The path where the CSV file should be saved.
        """
        if self.merged_data is None:
            raise ValueError("Datasets have not been merged yet.")
        self.merged_data.to_csv(filepath, index=False)
        print(f"Merged dataset exported to {filepath}")
