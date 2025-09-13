import pytest
import pandas as pd
import numpy as np
import tempfile
import os
from unittest.mock import patch, MagicMock
from sklearn.datasets import load_iris, load_wine

# Assuming the project structure, adjust import path as needed
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.data.loader import DataLoader, load_iris_data, load_wine_data

class TestDataLoader:
    """Test cases for DataLoader class."""
    
    @pytest.fixture
    def data_loader(self):
        """Create DataLoader instance for testing."""
        return DataLoader(random_state=42)
    
    @pytest.fixture
    def sample_csv_data(self):
        """Create sample CSV data for testing."""
        data = {
            'feature_1': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            'feature_2': [2.0, 4.0, 6.0, 8.0, 10.0, 12.0],
            'category': ['A', 'B', 'A', 'B', 'A', 'B'],
            'target': [0, 1, 0, 1, 0, 1]
        }
        return pd.DataFrame(data)
    
    def test_init(self):
        """Test DataLoader initialization."""
        loader = DataLoader(random_state=123)
        assert loader.random_state == 123
        assert loader.scaler is not None
        assert loader.label_encoder is not None
    
    def test_load_iris_dataset(self, data_loader):
        """Test loading iris dataset."""
        X_train, X_test, y_train, y_test = data_loader.load_builtin_dataset('iris')
        
        # Check shapes
        assert X_train.shape[0] + X_test.shape[0] == 150  # Iris has 150 samples
        assert X_train.shape[1] == 4  # Iris has 4 features
        assert X_test.shape[1] == 4
        assert len(y_train) == X_train.shape[0]
        assert len(y_test) == X_test.shape[0]
        
        # Check data types
        assert isinstance(X_train, np.ndarray)
        assert isinstance(y_train, np.ndarray)
        
        # Check target classes
        unique_classes = np.unique(np.concatenate([y_train, y_test]))
        assert len(unique_classes) == 3  # Iris has 3 classes
    
    def test_load_wine_dataset(self, data_loader):
        """Test loading wine dataset."""
        X_train, X_test, y_train, y_test = data_loader.load_builtin_dataset('wine')
        
        # Check shapes
        assert X_train.shape[0] + X_test.shape[0] == 178  # Wine has 178 samples
        assert X_train.shape[1] == 13  # Wine has 13 features
        
        # Check target classes
        unique_classes = np.unique(np.concatenate([y_train, y_test]))
        assert len(unique_classes) == 3  # Wine has 3 classes
    
    def test_load_dataset_with_dataframe_return(self, data_loader):
        """Test loading dataset with DataFrame return."""
        X_train, X_test, y_train, y_test = data_loader.load_builtin_dataset(
            'iris', return_dataframe=True
        )
        
        # Check types
        assert isinstance(X_train, pd.DataFrame)
        assert isinstance(X_test, pd.DataFrame)
        assert isinstance(y_train, pd.Series)
        assert isinstance(y_test, pd.Series)
        
        # Check column names
        assert len(X_train.columns) == 4
        assert y_train.name == 'target'
        assert y_test.name == 'target'
    
    def test_load_dataset_without_scaling(self, data_loader):
        """Test loading dataset without feature scaling."""
        X_train_scaled, _, _, _ = data_loader.load_builtin_dataset('iris', scale_features=True)
        X_train_unscaled, _, _, _ = data_loader.load_builtin_dataset('iris', scale_features=False)
        
        # Scaled data should have different values than unscaled
        assert not np.allclose(X_train_scaled, X_train_unscaled)
        
        # Scaled data should have approximately zero mean and unit variance
        assert np.allclose(np.mean(X_train_scaled, axis=0), 0, atol=1e-10)
        assert np.allclose(np.std(X_train_scaled, axis=0), 1, atol=1e-10)
    
    def test_load_nonexistent_dataset(self, data_loader):
        """Test loading non-existent dataset raises error."""
        with pytest.raises(ValueError, match="Dataset 'nonexistent' not available"):
            data_loader.load_builtin_dataset('nonexistent')
    
    def test_load_csv_dataset(self, data_loader, sample_csv_data):
        """Test loading dataset from CSV file."""
        # Create temporary CSV file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            sample_csv_data.to_csv(f.name, index=False)
            temp_file = f.name
        
        try:
            X_train, X_test, y_train, y_test = data_loader.load_csv_dataset(
                temp_file, target_column='target'
            )
            
            # Check types
            assert isinstance(X_train, pd.DataFrame)
            assert isinstance(y_train, pd.Series)
            
            # Check shapes
            assert X_train.shape[0] + X_test.shape[0] == 6  # Original data has 6 samples
            assert len(y_train) == X_train.shape[0]
            assert len(y_test) == X_test.shape[0]
            
            # Check that categorical column was handled
            assert 'category_A' in X_train.columns or 'category_B' in X_train.columns
            
        finally:
            os.unlink(temp_file)
    
    def test_load_csv_with_nonexistent_target(self, data_loader, sample_csv_data):
        """Test loading CSV with non-existent target column."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            sample_csv_data.to_csv(f.name, index=False)
            temp_file = f.name
        
        try:
            with pytest.raises(ValueError, match="Target column 'nonexistent' not found"):
                data_loader.load_csv_dataset(temp_file, target_column='nonexistent')
        finally:
            os.unlink(temp_file)
    
    def test_load_csv_with_drop_columns(self, data_loader, sample_csv_data):
        """Test loading CSV with columns to drop."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            sample_csv_data.to_csv(f.name, index=False)
            temp_file = f.name
        
        try:
            X_train, X_test, y_train, y_test = data_loader.load_csv_dataset(
                temp_file, target_column='target', drop_columns=['feature_2']
            )
            
            # Check that feature_2 was dropped
            assert 'feature_2' not in X_train.columns
            assert 'feature_1' in X_train.columns
            
        finally:
            os.unlink(temp_file)
    
    def test_create_synthetic_classification(self, data_loader):
        """Test creating synthetic classification dataset."""
        X_train, X_test, y_train, y_test = data_loader.create_synthetic_classification(
            n_samples=100, n_features=5, n_classes=3
        )
        
        # Check shapes
        assert X_train.shape[0] + X_test.shape[0] == 100
        assert X_train.shape[1] == 5
        assert X_test.shape[1] == 5
        
        # Check target classes
        unique_classes = np.unique(np.concatenate([y_train, y_test]))
        assert len(unique_classes) == 3
        
        # Check data types
        assert isinstance(X_train, np.ndarray)
        assert isinstance(y_train, np.ndarray)
    
    def test_create_synthetic_classification_dataframe(self, data_loader):
        """Test creating synthetic classification dataset as DataFrame."""
        X_train, X_test, y_train, y_test = data_loader.create_synthetic_classification(
            n_samples=50, n_features=3, n_classes=2, return_dataframe=True
        )
        
        # Check types
        assert isinstance(X_train, pd.DataFrame)
        assert isinstance(y_train, pd.Series)
        
        # Check column names
        expected_columns = ['feature_0', 'feature_1', 'feature_2']
        assert list(X_train.columns) == expected_columns
    
    def test_handle_categorical_features_encode(self, data_loader):
        """Test categorical feature encoding."""
        df = pd.DataFrame({
            'num_feature': [1, 2, 3, 4],
            'cat_feature': ['A', 'B', 'A', 'C']
        })
        
        result = data_loader._handle_categorical_features(df, method='encode')
        
        # Check that categorical column was one-hot encoded
        assert 'cat_feature_A' in result.columns
        assert 'cat_feature_B' in result.columns
        assert 'cat_feature_C' in result.columns
        assert 'cat_feature' not in result.columns
        assert 'num_feature' in result.columns
    
    def test_handle_categorical_features_drop(self, data_loader):
        """Test dropping categorical features."""
        df = pd.DataFrame({
            'num_feature': [1, 2, 3, 4],
            'cat_feature': ['A', 'B', 'A', 'C']
        })
        
        result = data_loader._handle_categorical_features(df, method='drop')
        
        # Check that categorical column was dropped
        assert 'cat_feature' not in result.columns
        assert 'num_feature' in result.columns
        assert result.shape[1] == 1
    
    def test_is_classification_target(self, data_loader):
        """Test classification target detection."""
        # Test categorical target
        cat_target = pd.Series(['A', 'B', 'A', 'B', 'C'])
        assert data_loader._is_classification_target(cat_target) == True
        
        # Test numeric target with few unique values
        numeric_cat_target = pd.Series([0, 1, 0, 1, 2])
        assert data_loader._is_classification_target(numeric_cat_target) == True
        
        # Test regression target
        regression_target = pd.Series([1.1, 2.3, 4.5, 6.7, 8.9, 10.1, 12.3])
        assert data_loader._is_classification_target(regression_target) == False
    
    def test_get_dataset_info(self, data_loader):
        """Test getting dataset information."""
        info = data_loader.get_dataset_info('iris')
        
        assert info['name'] == 'iris'
        assert info['n_samples'] == 150
        assert info['n_features'] == 4
        assert info['task_type'] == 'classification'
        assert info['n_classes'] == 3
        assert 'description' in info
    
    def test_get_dataset_info_nonexistent(self, data_loader):
        """Test getting info for non-existent dataset."""
        with pytest.raises(ValueError, match="Dataset 'nonexistent' not available"):
            data_loader.get_dataset_info('nonexistent')
    
    def test_list_available_datasets(self, data_loader):
        """Test listing available datasets."""
        datasets = data_loader.list_available_datasets()
        
        assert 'classification' in datasets
        assert 'regression' in datasets
        assert 'iris' in datasets['classification']
        assert 'wine' in datasets['classification']
        assert isinstance(datasets['classification'], list)
        assert isinstance(datasets['regression'], list)
    
    @patch('src.data.loader.load_boston')
    def test_boston_dataset_deprecation_warning(self, mock_load_boston, data_loader):
        """Test handling of deprecated Boston housing dataset."""
        # Mock the ImportError that occurs with deprecated Boston dataset
        mock_load_boston.side_effect = ImportError("Boston dataset deprecated")
        
        # Should fall back to synthetic regression data
        X_train, X_test, y_train, y_test = data_loader.load_builtin_dataset('boston')
        
        # Check that we got data (from synthetic generation)
        assert X_train is not None
        assert X_test is not None
        assert y_train is not None
        assert y_test is not None
        assert X_train.shape[0] + X_test.shape[0] == 506  # Boston dataset size
        assert X_train.shape[1] == 13  # Boston feature count
    
    def test_reproducibility(self, data_loader):
        """Test that results are reproducible with same random state."""
        # Load same dataset twice
        X_train1, X_test1, y_train1, y_test1 = data_loader.load_builtin_dataset('iris')
        
        # Create new loader with same random state
        loader2 = DataLoader(random_state=42)
        X_train2, X_test2, y_train2, y_test2 = loader2.load_builtin_dataset('iris')
        
        # Results should be identical
        np.testing.assert_array_equal(X_train1, X_train2)
        np.testing.assert_array_equal(y_train1, y_train2)
        np.testing.assert_array_equal(X_test1, X_test2)
        np.testing.assert_array_equal(y_test1, y_test2)


class TestConvenienceFunctions:
    """Test convenience functions."""
    
    def test_load_iris_data(self):
        """Test iris convenience function."""
        X_train, X_test, y_train, y_test = load_iris_data()
        
        assert X_train.shape[0] + X_test.shape[0] == 150
        assert X_train.shape[1] == 4
        assert len(np.unique(np.concatenate([y_train, y_test]))) == 3
    
    def test_load_wine_data(self):
        """Test wine convenience function."""
        X_train, X_test, y_train, y_test = load_wine_data(test_size=0.3)
        
        assert X_train.shape[0] + X_test.shape[0] == 178
        assert X_train.shape[1] == 13
        
        # Check test_size parameter worked
        total_samples = X_train.shape[0] + X_test.shape[0]
        expected_test_samples = int(0.3 * 178)
        assert abs(X_test.shape[0] - expected_test_samples) <= 1  # Allow for rounding
    
    def test_convenience_function_parameters(self):
        """Test that convenience functions pass parameters correctly."""
        X_train, X_test, y_train, y_test = load_iris_data(
            test_size=0.3,
            scale_features=False,
            return_dataframe=True
        )
        
        # Check DataFrame return
        assert isinstance(X_train, pd.DataFrame)
        assert isinstance(y_train, pd.Series)
        
        # Check test size (approximately)
        total_samples = X_train.shape[0] + X_test.shape[0]
        expected_test_samples = int(0.3 * 150)
        assert abs(X_test.shape[0] - expected_test_samples) <= 1


class TestEdgeCases:
    """Test edge cases and error conditions."""
    
    @pytest.fixture
    def data_loader(self):
        return DataLoader(random_state=42)
    
    def test_empty_csv_file(self, data_loader):
        """Test handling of empty CSV file."""
        # Create empty CSV file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write('')  # Empty file
            temp_file = f.name
        
        try:
            with pytest.raises(Exception):  # Should raise some exception
                data_loader.load_csv_dataset(temp_file, target_column='target')
        finally:
            os.unlink(temp_file)
    
    def test_csv_with_missing_values(self, data_loader):
        """Test CSV with missing values."""
        df = pd.DataFrame({
            'feature_1': [1.0, 2.0, np.nan, 4.0],
            'feature_2': [2.0, np.nan, 6.0, 8.0],
            'target': [0, 1, 0, 1]
        })
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            df.to_csv(f.name, index=False)
            temp_file = f.name
        
        try:
            # Should handle missing values gracefully (or raise appropriate error)
            X_train, X_test, y_train, y_test = data_loader.load_csv_dataset(
                temp_file, target_column='target'
            )
            # If it doesn't raise error, check that we got some data
            assert X_train is not None
        except Exception as e:
            # If it raises an error, it should be a meaningful one
            assert isinstance(e, (ValueError, TypeError))
        finally:
            os.unlink(temp_file)
    
    def test_very_small_dataset(self, data_loader):
        """Test handling very small datasets."""
        X_train, X_test, y_train, y_test = data_loader.create_synthetic_classification(
            n_samples=4, n_features=2, n_classes=2, test_size=0.5
        )
        
        # Should still work with small datasets
        assert X_train.shape[0] == 2
        assert X_test.shape[0] == 2
        assert len(y_train) == 2
        assert len(y_test) == 2


if __name__ == "__main__":
    # Run tests if script is executed directly
    pytest.main([__file__, "-v"])