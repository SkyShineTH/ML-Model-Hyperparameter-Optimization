import pytest
import numpy as np
from unittest.mock import patch, MagicMock
from typing import Dict, Any

# Sklearn imports
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, SVR

# Assuming the project structure, adjust import path as needed
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.model.model_factory import (
    ModelFactory, 
    RandomForestConfig, 
    LogisticRegressionConfig,
    SVMConfig,
    BaseModelConfig,
    create_random_forest,
    create_xgboost,
    create_lightgbm,
    XGBOOST_AVAILABLE,
    LIGHTGBM_AVAILABLE,
    CATBOOST_AVAILABLE
)


class TestBaseModelConfig:
    """Test BaseModelConfig abstract class."""
    
    def test_abstract_methods(self):
        """Test that BaseModelConfig cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BaseModelConfig()


class TestRandomForestConfig:
    """Test RandomForest model configuration."""
    
    @pytest.fixture
    def rf_config_clf(self):
        """Random Forest classifier config."""
        return RandomForestConfig(task_type='classification', random_state=42)
    
    @pytest.fixture
    def rf_config_reg(self):
        """Random Forest regressor config."""
        return RandomForestConfig(task_type='regression', random_state=42)
    
    def test_init(self, rf_config_clf):
        """Test RandomForest config initialization."""
        assert rf_config_clf.task_type == 'classification'
        assert rf_config_clf.random_state == 42
    
    def test_create_classifier(self, rf_config_clf):
        """Test creating Random Forest classifier."""
        model = rf_config_clf.create_model()
        
        assert isinstance(model, RandomForestClassifier)
        assert model.random_state == 42
        assert model.n_estimators == 100  # default value
    
    def test_create_regressor(self, rf_config_reg):
        """Test creating Random Forest regressor."""
        model = rf_config_reg.create_model()
        
        assert isinstance(model, RandomForestRegressor)
        assert model.random_state == 42
    
    def test_create_model_with_params(self, rf_config_clf):
        """Test creating model with custom parameters."""
        model = rf_config_clf.create_model(n_estimators=200, max_depth=10)
        
        assert model.n_estimators == 200
        assert model.max_depth == 10
        assert model.random_state == 42  # Should preserve random_state
    
    def test_get_default_params(self, rf_config_clf):
        """Test getting default parameters."""
        params = rf_config_clf.get_default_params()
        
        assert isinstance(params, dict)
        assert 'n_estimators' in params
        assert 'max_depth' in params
        assert 'n_jobs' in params
        assert params['n_estimators'] == 100
    
    def test_get_hyperparameter_space_optuna(self, rf_config_clf):
        """Test getting hyperparameter space for Optuna."""
        space = rf_config_clf.get_hyperparameter_space('optuna')
        
        assert isinstance(space, dict)
        assert 'n_estimators' in space
        assert 'max_depth' in space
        assert 'min_samples_split' in space
        
        # Check format for Optuna
        assert space['n_estimators'][0] == 'int'
        assert space['n_estimators'][1] == 50  # min value
        assert space['n_estimators'][2] == 500  # max value
    
    def test_get_hyperparameter_space_hyperopt(self, rf_config_clf):
        """Test getting hyperparameter space for Hyperopt."""
        space = rf_config_clf.get_hyperparameter_space('hyperopt')
        
        assert isinstance(space, dict)
        assert 'n_estimators' in space
        assert 'max_depth' in space
        
        # Should contain hyperopt objects (can't test exact type without hyperopt)
        assert space['n_estimators'] is not None
    
    def test_unsupported_optimization_framework(self, rf_config_clf):
        """Test error for unsupported optimization framework."""
        with pytest.raises(ValueError, match="Unsupported optimization framework"):
            rf_config_clf.get_hyperparameter_space('unknown_framework')


class TestLogisticRegressionConfig:
    """Test LogisticRegression model configuration."""
    
    @pytest.fixture
    def lr_config(self):
        """Logistic Regression config."""
        return LogisticRegressionConfig(random_state=42)
    
    def test_create_model(self, lr_config):
        """Test creating Logistic Regression model."""
        model = lr_config.create_model()
        
        assert isinstance(model, LogisticRegression)
        assert model.random_state == 42
        assert model.max_iter == 1000
    
    def test_get_hyperparameter_space(self, lr_config):
        """Test hyperparameter space for Logistic Regression."""
        space = lr_config.get_hyperparameter_space('optuna')
        
        assert 'C' in space
        assert 'penalty' in space
        assert 'solver' in space
        
        # Check log-uniform for C parameter
        assert space['C'][0] == 'float_log'
        
        # Check categorical choices
        assert 'l1' in space['penalty'][1]
        assert 'l2' in space['penalty'][1]


class TestSVMConfig:
    """Test SVM model configuration."""
    
    @pytest.fixture
    def svm_config_clf(self):
        """SVM classifier config."""
        return SVMConfig(task_type='classification', random_state=42)
    
    @pytest.fixture
    def svm_config_reg(self):
        """SVM regressor config."""
        return SVMConfig(task_type='regression', random_state=42)
    
    def test_create_classifier(self, svm_config_clf):
        """Test creating SVM classifier."""
        model = svm_config_clf.create_model()
        
        assert isinstance(model, SVC)
        assert model.random_state == 42
    
    def test_create_regressor(self, svm_config_reg):
        """Test creating SVM regressor."""
        model = svm_config_reg.create_model()
        
        assert isinstance(model, SVR)
        # SVR doesn't have random_state, but should work
    
    def test_hyperparameter_space(self, svm_config_clf):
        """Test SVM hyperparameter space."""
        space = svm_config_clf.get_hyperparameter_space('optuna')
        
        assert 'C' in space
        assert 'kernel' in space
        assert 'gamma' in space
        
        # Check kernel choices
        kernels = space['kernel'][1]
        assert 'rbf' in kernels
        assert 'linear' in kernels
        assert 'poly' in kernels


@pytest.mark.skipif(not XGBOOST_AVAILABLE, reason="XGBoost not installed")
class TestXGBoostConfig:
    """Test XGBoost configuration (only if available)."""
    
    def test_xgboost_import(self):
        """Test XGBoost import and config creation."""
        from src.model.model_factory import XGBoostConfig
        import xgboost as xgb
        
        config = XGBoostConfig(task_type='classification', random_state=42)
        model = config.create_model()
        
        assert isinstance(model, xgb.XGBClassifier)
        assert model.random_state == 42
    
    def test_xgboost_hyperparameter_space(self):
        """Test XGBoost hyperparameter space."""
        from src.model.model_factory import XGBoostConfig
        
        config = XGBoostConfig(task_type='classification', random_state=42)
        space = config.get_hyperparameter_space('optuna')
        
        assert 'n_estimators' in space
        assert 'max_depth' in space
        assert 'learning_rate' in space
        assert 'subsample' in space


@pytest.mark.skipif(not LIGHTGBM_AVAILABLE, reason="LightGBM not installed")
class TestLightGBMConfig:
    """Test LightGBM configuration (only if available)."""
    
    def test_lightgbm_import(self):
        """Test LightGBM import and config creation."""
        from src.model.model_factory import LightGBMConfig
        import lightgbm as lgb
        
        config = LightGBMConfig(task_type='classification', random_state=42)
        model = config.create_model()
        
        assert isinstance(model, lgb.LGBMClassifier)
        assert model.random_state == 42


class TestModelFactory:
    """Test ModelFactory class."""
    
    @pytest.fixture
    def factory(self):
        """Model factory instance."""
        return ModelFactory(random_state=42)
    
    def test_init(self, factory):
        """Test factory initialization."""
        assert factory.random_state == 42
    
    def test_list_available_models(self, factory):
        """Test listing available models."""
        models = factory.list_available_models()
        
        assert isinstance(models, list)
        assert 'random_forest' in models
        assert 'logistic_regression' in models
        assert 'svm' in models
        
        # XGBoost should be in list even if not available (will fail at creation)
        assert 'xgboost' in models
    
    def test_create_model_config_random_forest(self, factory):
        """Test creating Random Forest config."""
        config = factory.create_model_config('random_forest', 'classification')
        
        assert isinstance(config, RandomForestConfig)
        assert config.task_type == 'classification'
        assert config.random_state == 42
    
    def test_create_model_config_invalid(self, factory):
        """Test creating config for invalid model."""
        with pytest.raises(ValueError, match="Model 'invalid_model' not available"):
            factory.create_model_config('invalid_model')
    
    def test_create_model_config_task_type_error(self, factory):
        """Test error for invalid task type combination."""
        with pytest.raises(ValueError, match="Logistic regression only supports classification"):
            factory.create_model_config('logistic_regression', 'regression')
    
    def test_create_model_random_forest(self, factory):
        """Test creating Random Forest model."""
        model = factory.create_model('random_forest', 'classification')
        
        assert isinstance(model, RandomForestClassifier)
        assert model.random_state == 42
    
    def test_create_model_with_params(self, factory):
        """Test creating model with custom parameters."""
        params = {'n_estimators': 200, 'max_depth': 10}
        model = factory.create_model('random_forest', 'classification', params=params)
        
        assert model.n_estimators == 200
        assert model.max_depth == 10
    
    def test_create_model_regressor(self, factory):
        """Test creating regressor model."""
        model = factory.create_model('random_forest', 'regression')
        
        assert isinstance(model, RandomForestRegressor)
    
    def test_get_hyperparameter_space(self, factory):
        """Test getting hyperparameter space through factory."""
        space = factory.get_hyperparameter_space('random_forest', 'classification', 'optuna')
        
        assert isinstance(space, dict)
        assert 'n_estimators' in space
        assert 'max_depth' in space
    
    def test_get_default_params(self, factory):
        """Test getting default parameters through factory."""
        params = factory.get_default_params('random_forest', 'classification')
        
        assert isinstance(params, dict)
        assert 'n_estimators' in params
        assert params['n_estimators'] == 100
    
    def test_get_model_info(self, factory):
        """Test getting model information."""
        info = factory.get_model_info('random_forest')
        
        assert isinstance(info, dict)
        assert info['name'] == 'random_forest'
        assert info['supports_classification'] == True
        assert info['supports_regression'] == True
        assert info['requires_external_lib'] == False
    
    def test_get_model_info_logistic_regression(self, factory):
        """Test model info for logistic regression."""
        info = factory.get_model_info('logistic_regression')
        
        assert info['supports_regression'] == False
        assert info['supports_classification'] == True
    
    def test_get_model_info_xgboost(self, factory):
        """Test model info for XGBoost."""
        info = factory.get_model_info('xgboost')
        
        assert info['requires_external_lib'] == True
    
    def test_get_model_info_invalid(self, factory):
        """Test error for invalid model info request."""
        with pytest.raises(ValueError, match="Model 'invalid' not available"):
            factory.get_model_info('invalid')


class TestConvenienceFunctions:
    """Test convenience functions."""
    
    def test_create_random_forest_default(self):
        """Test creating Random Forest with defaults."""
        model = create_random_forest()
        
        assert isinstance(model, RandomForestClassifier)
        assert model.n_estimators == 100
    
    def test_create_random_forest_with_params(self):
        """Test creating Random Forest with parameters."""
        model = create_random_forest('classification', n_estimators=200, max_depth=10)
        
        assert model.n_estimators == 200
        assert model.max_depth == 10
    
    def test_create_random_forest_regressor(self):
        """Test creating Random Forest regressor."""
        model = create_random_forest('regression')
        
        assert isinstance(model, RandomForestRegressor)
    
    @pytest.mark.skipif(not XGBOOST_AVAILABLE, reason="XGBoost not installed")
    def test_create_xgboost(self):
        """Test XGBoost convenience function."""
        import xgboost as xgb
        
        model = create_xgboost('classification')
        assert isinstance(model, xgb.XGBClassifier)
    
    @pytest.mark.skipif(not LIGHTGBM_AVAILABLE, reason="LightGBM not installed")
    def test_create_lightgbm(self):
        """Test LightGBM convenience function."""
        import lightgbm as lgb
        
        model = create_lightgbm('classification')
        assert isinstance(model, lgb.LGBMClassifier)


class TestErrorHandling:
    """Test error handling and edge cases."""
    
    @pytest.fixture
    def factory(self):
        return ModelFactory(random_state=42)
    
    def test_missing_external_library_error(self):
        """Test error when external library is missing."""
        # Mock missing XGBoost
        with patch('src.model.model_factory.XGBOOST_AVAILABLE', False):
            from src.model.model_factory import XGBoostConfig
            
            with pytest.raises(ImportError, match="XGBoost not installed"):
                XGBoostConfig(task_type='classification')
    
    def test_hyperparameter_space_with_missing_library(self):
        """Test hyperparameter space when hyperopt is missing."""
        factory = ModelFactory()
        
        # This should work for optuna
        space = factory.get_hyperparameter_space('random_forest', 'classification', 'optuna')
        assert isinstance(space, dict)
        
        # If hyperopt is not available, this might raise ImportError in create_model_config
        # That's expected behavior
    
    def test_invalid_task_type_combinations(self, factory):
        """Test various invalid task type combinations."""
        # Logistic regression with regression
        with pytest.raises(ValueError):
            factory.create_model_config('logistic_regression', 'regression')
        
        # Invalid task type
        with pytest.raises(Exception):  # Could be ValueError or other
            factory.create_model_config('random_forest', 'invalid_task')


class TestIntegration:
    """Integration tests combining multiple components."""
    
    def test_full_workflow_classification(self):
        """Test complete workflow for classification."""
        factory = ModelFactory(random_state=42)
        
        # Create model
        model = factory.create_model('random_forest', 'classification', 
                                   params={'n_estimators': 50})
        
        # Get hyperparameter space
        space = factory.get_hyperparameter_space('random_forest', 'classification')
        
        # Get default params
        defaults = factory.get_default_params('random_forest', 'classification')
        
        # Test with dummy data
        X = np.random.rand(100, 4)
        y = np.random.randint(0, 3, 100)
        
        # Should be able to fit
        model.fit(X, y)
        predictions = model.predict(X)
        
        assert len(predictions) == 100
        assert isinstance(space, dict)
        assert isinstance(defaults, dict)
    
    def test_full_workflow_regression(self):
        """Test complete workflow for regression."""
        factory = ModelFactory(random_state=42)
        
        # Create regressor
        model = factory.create_model('random_forest', 'regression')
        
        # Test with dummy data
        X = np.random.rand(100, 4)
        y = np.random.rand(100)
        
        # Should be able to fit
        model.fit(X, y)
        predictions = model.predict(X)
        
        assert len(predictions) == 100
        assert isinstance(model, RandomForestRegressor)
    
    def test_reproducibility(self):
        """Test that models are reproducible with same random state."""
        factory1 = ModelFactory(random_state=42)
        factory2 = ModelFactory(random_state=42)
        
        model1 = factory1.create_model('random_forest', 'classification')
        model2 = factory2.create_model('random_forest', 'classification')
        
        # Test with same dummy data
        X = np.random.RandomState(42).rand(50, 4)
        y = np.random.RandomState(42).randint(0, 3, 50)
        
        model1.fit(X, y)
        model2.fit(X, y)
        
        pred1 = model1.predict(X)
        pred2 = model2.predict(X)
        
        np.testing.assert_array_equal(pred1, pred2)


if __name__ == "__main__":
    # Run tests if script is executed directly
    pytest.main([__file__, "-v"])