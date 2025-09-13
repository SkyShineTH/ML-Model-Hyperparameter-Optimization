import pandas as pd
import numpy as np
from typing import Tuple, Optional, Dict, Any, List
from sklearn.datasets import (
    load_iris, load_wine, load_breast_cancer, load_digits,
    load_diabetes, make_classification, make_regression
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import logging

logger = logging.getLogger(__name__)


class DataLoader:
    CLASSIFICATION_DATASETS = {
        'iris': load_iris,
        'wine': load_wine,
        'breast_cancer': load_breast_cancer,
        'digits': load_digits
    }
    
    REGRESSION_DATASETS = {
        'diabetes': load_diabetes
    }
    
    def __init__(self, random_state: int = 42):
        """
        Initialize DataLoader.
        
        Args:
            random_state: Random state for reproducibility
        """
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        
    def load_builtin_dataset(
        self,
        dataset_name: str,
        test_size: float = 0.2,
        stratify: bool = True,
        scale_features: bool = True,
        return_dataframe: bool = False
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Load built-in sklearn dataset.
        
        Args:
            dataset_name: Name of the dataset ('iris', 'wine', etc.)
            test_size: Proportion of test set
            stratify: Whether to stratify split for classification
            scale_features: Whether to standardize features
            return_dataframe: Whether to return pandas DataFrames instead of arrays
            
        Returns:
            X_train, X_test, y_train, y_test
        """
        logger.info(f"Loading dataset: {dataset_name}")
        
        # Check if dataset exists
        all_datasets = {**self.CLASSIFICATION_DATASETS, **self.REGRESSION_DATASETS}
        if dataset_name not in all_datasets:
            available = list(all_datasets.keys())
            raise ValueError(f"Dataset '{dataset_name}' not available. "
                           f"Available datasets: {available}")
        
        # Load dataset
        dataset_func = all_datasets[dataset_name]
        
        try:
            data = dataset_func()
        except ImportError as e:
            if dataset_name == 'boston':
                logger.warning("Boston housing dataset deprecated. Using make_regression instead.")
                return self._create_synthetic_regression(
                    n_samples=506, n_features=13, test_size=test_size,
                    scale_features=scale_features, return_dataframe=return_dataframe
                )
            raise e
        
        X, y = data.data, data.target
        feature_names = getattr(data, 'feature_names', None)
        target_names = getattr(data, 'target_names', None)
        
        # Determine if it's classification or regression
        is_classification = dataset_name in self.CLASSIFICATION_DATASETS
        
        # Split data
        stratify_param = y if (stratify and is_classification) else None
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state,
            stratify=stratify_param
        )
        
        # Scale features if requested
        if scale_features:
            X_train = self.scaler.fit_transform(X_train)
            X_test = self.scaler.transform(X_test)
        
        # Convert to DataFrame if requested
        if return_dataframe:
            feature_names = feature_names if feature_names is not None else [f"feature_{i}" for i in range(X.shape[1])]
            X_train = pd.DataFrame(X_train, columns=feature_names)
            X_test = pd.DataFrame(X_test, columns=feature_names)
            y_train = pd.Series(y_train, name='target')
            y_test = pd.Series(y_test, name='target')
        
        logger.info(f"Dataset loaded successfully: "
                   f"X_train: {X_train.shape}, X_test: {X_test.shape}")
        
        return X_train, X_test, y_train, y_test
    
    def load_csv_dataset(
        self,
        file_path: str,
        target_column: str,
        test_size: float = 0.2,
        stratify: bool = None,
        scale_features: bool = True,
        drop_columns: Optional[List[str]] = None,
        handle_categorical: str = 'encode'  # 'encode', 'drop', or 'keep'
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Load dataset from CSV file.
        
        Args:
            file_path: Path to CSV file
            target_column: Name of target column
            test_size: Proportion of test set
            stratify: Whether to stratify split (auto-detect if None)
            scale_features: Whether to standardize numerical features
            drop_columns: Columns to drop before processing
            handle_categorical: How to handle categorical variables
            
        Returns:
            X_train, X_test, y_train, y_test as DataFrames/Series
        """
        logger.info(f"Loading CSV dataset from: {file_path}")
        
        # Load data
        try:
            df = pd.read_csv(file_path)
        except Exception as e:
            logger.error(f"Error loading CSV file: {e}")
            raise
        
        logger.info(f"Dataset shape: {df.shape}")
        
        # Drop specified columns
        if drop_columns:
            df = df.drop(columns=drop_columns, errors='ignore')
            logger.info(f"Dropped columns: {drop_columns}")
        
        # Separate features and target
        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found in dataset")
        
        X = df.drop(columns=[target_column])
        y = df[target_column]
        
        # Handle categorical variables
        X = self._handle_categorical_features(X, method=handle_categorical)
        
        # Auto-detect classification vs regression
        if stratify is None:
            stratify = self._is_classification_target(y)
        
        # Split data
        stratify_param = y if stratify else None
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state,
            stratify=stratify_param
        )
        
        # Scale numerical features
        if scale_features:
            numerical_cols = X_train.select_dtypes(include=[np.number]).columns
            if len(numerical_cols) > 0:
                X_train[numerical_cols] = self.scaler.fit_transform(X_train[numerical_cols])
                X_test[numerical_cols] = self.scaler.transform(X_test[numerical_cols])
                logger.info(f"Scaled {len(numerical_cols)} numerical features")
        
        logger.info(f"Dataset processed successfully: "
                   f"X_train: {X_train.shape}, X_test: {X_test.shape}")
        
        return X_train, X_test, y_train, y_test
    
    def create_synthetic_classification(
        self,
        n_samples: int = 1000,
        n_features: int = 10,
        n_classes: int = 3,
        n_informative: int = None,
        test_size: float = 0.2,
        scale_features: bool = True,
        return_dataframe: bool = False
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Create synthetic classification dataset.
        
        Args:
            n_samples: Number of samples
            n_features: Number of features
            n_classes: Number of classes
            n_informative: Number of informative features (default: n_features)
            test_size: Proportion of test set
            scale_features: Whether to standardize features
            return_dataframe: Whether to return pandas DataFrames
            
        Returns:
            X_train, X_test, y_train, y_test
        """
        logger.info(f"Creating synthetic classification dataset: "
                   f"{n_samples} samples, {n_features} features, {n_classes} classes")
        
        n_informative = n_informative or n_features
        
        X, y = make_classification(
            n_samples=n_samples,
            n_features=n_features,
            n_classes=n_classes,
            n_informative=n_informative,
            n_redundant=max(0, n_features - n_informative),
            random_state=self.random_state
        )
        
        return self._process_synthetic_data(
            X, y, test_size, scale_features, return_dataframe, is_classification=True
        )
    
    def _create_synthetic_regression(
        self,
        n_samples: int = 1000,
        n_features: int = 10,
        noise: float = 0.1,
        test_size: float = 0.2,
        scale_features: bool = True,
        return_dataframe: bool = False
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Create synthetic regression dataset."""
        logger.info(f"Creating synthetic regression dataset: "
                   f"{n_samples} samples, {n_features} features")
        
        X, y = make_regression(
            n_samples=n_samples,
            n_features=n_features,
            noise=noise,
            random_state=self.random_state
        )
        
        return self._process_synthetic_data(
            X, y, test_size, scale_features, return_dataframe, is_classification=False
        )
    
    def _process_synthetic_data(
        self, X, y, test_size, scale_features, return_dataframe, is_classification
    ):
        """Process synthetic data (common logic for classification and regression)."""
        # Split data
        stratify_param = y if is_classification else None
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state,
            stratify=stratify_param
        )
        
        # Scale features
        if scale_features:
            X_train = self.scaler.fit_transform(X_train)
            X_test = self.scaler.transform(X_test)
        
        # Convert to DataFrame if requested
        if return_dataframe:
            feature_names = [f"feature_{i}" for i in range(X.shape[1])]
            X_train = pd.DataFrame(X_train, columns=feature_names)
            X_test = pd.DataFrame(X_test, columns=feature_names)
            y_train = pd.Series(y_train, name='target')
            y_test = pd.Series(y_test, name='target')
        
        return X_train, X_test, y_train, y_test
    
    def _handle_categorical_features(
        self, X: pd.DataFrame, method: str = 'encode'
    ) -> pd.DataFrame:
        """Handle categorical features in the dataset."""
        categorical_cols = X.select_dtypes(include=['object', 'category']).columns
        
        if len(categorical_cols) == 0:
            return X
        
        X = X.copy()
        
        if method == 'encode':
            # One-hot encode categorical variables
            X = pd.get_dummies(X, columns=categorical_cols, prefix=categorical_cols)
            logger.info(f"One-hot encoded {len(categorical_cols)} categorical features")
        
        elif method == 'drop':
            # Drop categorical columns
            X = X.drop(columns=categorical_cols)
            logger.info(f"Dropped {len(categorical_cols)} categorical features")
        
        elif method == 'keep':
            # Keep as is (may cause issues with some models)
            logger.warning(f"Keeping {len(categorical_cols)} categorical features as-is")
        
        return X
    
    def _is_classification_target(self, y: pd.Series) -> bool:
        """Determine if target is classification or regression."""
        # Check if target is categorical or has few unique values
        if y.dtype == 'object' or y.dtype.name == 'category':
            return True
        
        # If numeric, check if it's integer-like (could be classification)
        if np.issubdtype(y.dtype, np.floating):
            # For float data, check if all values are actually integers
            if not np.allclose(y, y.astype(int), equal_nan=True):
                # Has decimal values, likely regression
                return False
        
        # If numeric, check number of unique values
        unique_values = y.nunique()
        total_values = len(y)
        
        # More conservative heuristic: 
        # - Less than 20 unique values AND less than 20% unique ratio
        # - OR less than 10 unique values regardless of ratio
        if (unique_values < 20 and (unique_values / total_values) < 0.2) or unique_values < 10:
            return True
        
        return False
    
    def get_dataset_info(self, dataset_name: str) -> Dict[str, Any]:
        """Get information about a built-in dataset."""
        all_datasets = {**self.CLASSIFICATION_DATASETS, **self.REGRESSION_DATASETS}
        
        if dataset_name not in all_datasets:
            raise ValueError(f"Dataset '{dataset_name}' not available")
        
        try:
            data = all_datasets[dataset_name]()
        except ImportError:
            if dataset_name == 'boston':
                return {
                    'name': dataset_name,
                    'n_samples': 506,
                    'n_features': 13,
                    'task_type': 'regression',
                    'description': 'Boston housing (deprecated, using synthetic alternative)'
                }
            raise
        
        info = {
            'name': dataset_name,
            'n_samples': data.data.shape[0],
            'n_features': data.data.shape[1],
            'task_type': 'classification' if dataset_name in self.CLASSIFICATION_DATASETS else 'regression',
            'description': getattr(data, 'DESCR', 'No description available')[:200] + '...'
        }
        
        if hasattr(data, 'target_names'):
            info['n_classes'] = len(data.target_names)
            info['classes'] = list(data.target_names)
        
        return info
    
    def list_available_datasets(self) -> Dict[str, List[str]]:
        """List all available built-in datasets."""
        return {
            'classification': list(self.CLASSIFICATION_DATASETS.keys()),
            'regression': list(self.REGRESSION_DATASETS.keys())
        }
    
# Convenience functions for quick data loading
def load_iris_data(**kwargs):
    """Quick loader for iris dataset."""
    loader = DataLoader()
    return loader.load_builtin_dataset('iris', **kwargs)


def load_wine_data(**kwargs):
    """Quick loader for wine dataset."""
    loader = DataLoader()
    return loader.load_builtin_dataset('wine', **kwargs)


def load_breast_cancer_data(**kwargs):
    """Quick loader for breast cancer dataset."""
    loader = DataLoader()
    return loader.load_builtin_dataset('breast_cancer', **kwargs)