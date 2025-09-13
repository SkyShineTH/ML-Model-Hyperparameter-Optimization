"""
Model factory for creating and configuring ML models with hyperparameter spaces.
Supports various algorithms including tree-based models, linear models, and ensemble methods.
"""

import numpy as np
from typing import Dict, Any, Optional, List, Union, Tuple
from abc import ABC, abstractmethod
import logging

# Scikit-learn models
from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor,
    GradientBoostingClassifier, GradientBoostingRegressor,
    ExtraTreesClassifier, ExtraTreesRegressor,
    AdaBoostClassifier, AdaBoostRegressor
)
from sklearn.svm import SVC, SVR
from sklearn.linear_model import (
    LogisticRegression, LinearRegression, Ridge, Lasso,
    ElasticNet, SGDClassifier, SGDRegressor
)
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

# Gradient boosting libraries
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    xgb = None

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    lgb = None

try:
    import catboost as cb
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False
    cb = None

logger = logging.getLogger(__name__)


class BaseModelConfig(ABC):
    """Base class for model configurations with hyperparameter spaces."""
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
    
    @abstractmethod
    def create_model(self, **kwargs) -> Any:
        """Create model instance with given parameters."""
        pass
    
    @abstractmethod
    def get_hyperparameter_space(self, optimization_framework: str = 'optuna') -> Dict[str, Any]:
        """Get hyperparameter search space for optimization framework."""
        pass
    
    @abstractmethod
    def get_default_params(self) -> Dict[str, Any]:
        """Get default hyperparameters for the model."""
        pass


class RandomForestConfig(BaseModelConfig):
    """Configuration for Random Forest models."""
    
    def __init__(self, task_type: str = 'classification', random_state: int = 42):
        super().__init__(random_state)
        self.task_type = task_type
    
    def create_model(self, **kwargs) -> Union[RandomForestClassifier, RandomForestRegressor]:
        """Create Random Forest model."""
        params = self.get_default_params()
        params.update(kwargs)
        params['random_state'] = self.random_state
        
        if self.task_type == 'classification':
            return RandomForestClassifier(**params)
        else:
            return RandomForestRegressor(**params)
    
    def get_hyperparameter_space(self, optimization_framework: str = 'optuna') -> Dict[str, Any]:
        """Get hyperparameter space for Random Forest."""
        if optimization_framework == 'optuna':
            return {
                'n_estimators': ('int', 50, 500),
                'max_depth': ('int_nullable', 3, 20),
                'min_samples_split': ('int', 2, 20),
                'min_samples_leaf': ('int', 1, 10),
                'max_features': ('categorical', ['sqrt', 'log2', None]),
                'bootstrap': ('categorical', [True, False])
            }
        elif optimization_framework == 'hyperopt':
            from hyperopt import hp
            return {
                'n_estimators': hp.randint('n_estimators', 50, 500),
                'max_depth': hp.choice('max_depth', [None] + list(range(3, 21))),
                'min_samples_split': hp.randint('min_samples_split', 2, 20),
                'min_samples_leaf': hp.randint('min_samples_leaf', 1, 10),
                'max_features': hp.choice('max_features', ['sqrt', 'log2', None]),
                'bootstrap': hp.choice('bootstrap', [True, False])
            }
        else:
            raise ValueError(f"Unsupported optimization framework: {optimization_framework}")
    
    def get_default_params(self) -> Dict[str, Any]:
        """Get default parameters for Random Forest."""
        return {
            'n_estimators': 100,
            'max_depth': None,
            'min_samples_split': 2,
            'min_samples_leaf': 1,
            'bootstrap': True,
            'n_jobs': -1
        }


class XGBoostConfig(BaseModelConfig):
    """Configuration for XGBoost models."""
    
    def __init__(self, task_type: str = 'classification', random_state: int = 42):
        super().__init__(random_state)
        self.task_type = task_type
        if not XGBOOST_AVAILABLE:
            raise ImportError("XGBoost not installed. Install with: pip install xgboost")
    
    def create_model(self, **kwargs) -> Union[xgb.XGBClassifier, xgb.XGBRegressor]:
        """Create XGBoost model."""
        params = self.get_default_params()
        params.update(kwargs)
        params['random_state'] = self.random_state
        
        if self.task_type == 'classification':
            return xgb.XGBClassifier(**params)
        else:
            return xgb.XGBRegressor(**params)
    
    def get_hyperparameter_space(self, optimization_framework: str = 'optuna') -> Dict[str, Any]:
        """Get hyperparameter space for XGBoost."""
        if optimization_framework == 'optuna':
            return {
                'n_estimators': ('int', 50, 1000),
                'max_depth': ('int', 3, 10),
                'learning_rate': ('float', 0.01, 0.3),
                'subsample': ('float', 0.6, 1.0),
                'colsample_bytree': ('float', 0.6, 1.0),
                'min_child_weight': ('int', 1, 10),
                'reg_alpha': ('float', 0.0, 1.0),
                'reg_lambda': ('float', 0.0, 1.0)
            }
        elif optimization_framework == 'hyperopt':
            from hyperopt import hp
            return {
                'n_estimators': hp.randint('n_estimators', 50, 1000),
                'max_depth': hp.randint('max_depth', 3, 10),
                'learning_rate': hp.uniform('learning_rate', 0.01, 0.3),
                'subsample': hp.uniform('subsample', 0.6, 1.0),
                'colsample_bytree': hp.uniform('colsample_bytree', 0.6, 1.0),
                'min_child_weight': hp.randint('min_child_weight', 1, 10),
                'reg_alpha': hp.uniform('reg_alpha', 0.0, 1.0),
                'reg_lambda': hp.uniform('reg_lambda', 0.0, 1.0)
            }
        else:
            raise ValueError(f"Unsupported optimization framework: {optimization_framework}")
    
    def get_default_params(self) -> Dict[str, Any]:
        """Get default parameters for XGBoost."""
        base_params = {
            'n_estimators': 100,
            'max_depth': 6,
            'learning_rate': 0.1,
            'subsample': 1.0,
            'colsample_bytree': 1.0,
            'n_jobs': -1,
            'verbosity': 0
        }
        
        if self.task_type == 'classification':
            base_params['eval_metric'] = 'logloss'
        else:
            base_params['eval_metric'] = 'rmse'
            
        return base_params


class LightGBMConfig(BaseModelConfig):
    """Configuration for LightGBM models."""
    
    def __init__(self, task_type: str = 'classification', random_state: int = 42):
        super().__init__(random_state)
        self.task_type = task_type
        if not LIGHTGBM_AVAILABLE:
            raise ImportError("LightGBM not installed. Install with: pip install lightgbm")
    
    def create_model(self, **kwargs) -> Union[lgb.LGBMClassifier, lgb.LGBMRegressor]:
        """Create LightGBM model."""
        params = self.get_default_params()
        params.update(kwargs)
        params['random_state'] = self.random_state
        
        if self.task_type == 'classification':
            return lgb.LGBMClassifier(**params)
        else:
            return lgb.LGBMRegressor(**params)
    
    def get_hyperparameter_space(self, optimization_framework: str = 'optuna') -> Dict[str, Any]:
        """Get hyperparameter space for LightGBM."""
        if optimization_framework == 'optuna':
            return {
                'n_estimators': ('int', 50, 1000),
                'max_depth': ('int', 3, 15),
                'learning_rate': ('float', 0.01, 0.3),
                'num_leaves': ('int', 20, 300),
                'subsample': ('float', 0.6, 1.0),
                'colsample_bytree': ('float', 0.6, 1.0),
                'min_child_samples': ('int', 5, 100),
                'reg_alpha': ('float', 0.0, 1.0),
                'reg_lambda': ('float', 0.0, 1.0)
            }
        elif optimization_framework == 'hyperopt':
            from hyperopt import hp
            return {
                'n_estimators': hp.randint('n_estimators', 50, 1000),
                'max_depth': hp.randint('max_depth', 3, 15),
                'learning_rate': hp.uniform('learning_rate', 0.01, 0.3),
                'num_leaves': hp.randint('num_leaves', 20, 300),
                'subsample': hp.uniform('subsample', 0.6, 1.0),
                'colsample_bytree': hp.uniform('colsample_bytree', 0.6, 1.0),
                'min_child_samples': hp.randint('min_child_samples', 5, 100),
                'reg_alpha': hp.uniform('reg_alpha', 0.0, 1.0),
                'reg_lambda': hp.uniform('reg_lambda', 0.0, 1.0)
            }
        else:
            raise ValueError(f"Unsupported optimization framework: {optimization_framework}")
    
    def get_default_params(self) -> Dict[str, Any]:
        """Get default parameters for LightGBM."""
        return {
            'n_estimators': 100,
            'max_depth': -1,
            'learning_rate': 0.1,
            'num_leaves': 31,
            'subsample': 1.0,
            'colsample_bytree': 1.0,
            'n_jobs': -1,
            'verbosity': -1
        }


class CatBoostConfig(BaseModelConfig):
    """Configuration for CatBoost models."""
    
    def __init__(self, task_type: str = 'classification', random_state: int = 42):
        super().__init__(random_state)
        self.task_type = task_type
        if not CATBOOST_AVAILABLE:
            raise ImportError("CatBoost not installed. Install with: pip install catboost")
    
    def create_model(self, **kwargs) -> Union[cb.CatBoostClassifier, cb.CatBoostRegressor]:
        """Create CatBoost model."""
        params = self.get_default_params()
        params.update(kwargs)
        params['random_state'] = self.random_state
        
        if self.task_type == 'classification':
            return cb.CatBoostClassifier(**params)
        else:
            return cb.CatBoostRegressor(**params)
    
    def get_hyperparameter_space(self, optimization_framework: str = 'optuna') -> Dict[str, Any]:
        """Get hyperparameter space for CatBoost."""
        if optimization_framework == 'optuna':
            return {
                'iterations': ('int', 50, 1000),
                'depth': ('int', 4, 10),
                'learning_rate': ('float', 0.01, 0.3),
                'l2_leaf_reg': ('float', 1.0, 10.0),
                'border_count': ('int', 32, 255),
                'bagging_temperature': ('float', 0.0, 1.0)
            }
        elif optimization_framework == 'hyperopt':
            from hyperopt import hp
            return {
                'iterations': hp.randint('iterations', 50, 1000),
                'depth': hp.randint('depth', 4, 10),
                'learning_rate': hp.uniform('learning_rate', 0.01, 0.3),
                'l2_leaf_reg': hp.uniform('l2_leaf_reg', 1.0, 10.0),
                'border_count': hp.randint('border_count', 32, 255),
                'bagging_temperature': hp.uniform('bagging_temperature', 0.0, 1.0)
            }
        else:
            raise ValueError(f"Unsupported optimization framework: {optimization_framework}")
    
    def get_default_params(self) -> Dict[str, Any]:
        """Get default parameters for CatBoost."""
        return {
            'iterations': 100,
            'depth': 6,
            'learning_rate': 0.1,
            'verbose': False,
            'thread_count': -1
        }


class LogisticRegressionConfig(BaseModelConfig):
    """Configuration for Logistic Regression."""
    
    def __init__(self, random_state: int = 42):
        super().__init__(random_state)
    
    def create_model(self, **kwargs) -> LogisticRegression:
        """Create Logistic Regression model."""
        params = self.get_default_params()
        params.update(kwargs)
        params['random_state'] = self.random_state
        return LogisticRegression(**params)
    
    def get_hyperparameter_space(self, optimization_framework: str = 'optuna') -> Dict[str, Any]:
        """Get hyperparameter space for Logistic Regression."""
        if optimization_framework == 'optuna':
            return {
                'C': ('float_log', 0.001, 100.0),
                'penalty': ('categorical', ['l1', 'l2', 'elasticnet']),
                'solver': ('categorical', ['liblinear', 'saga']),
                'l1_ratio': ('float', 0.0, 1.0)  # Only used when penalty='elasticnet'
            }
        elif optimization_framework == 'hyperopt':
            from hyperopt import hp
            return {
                'C': hp.loguniform('C', np.log(0.001), np.log(100.0)),
                'penalty': hp.choice('penalty', ['l1', 'l2', 'elasticnet']),
                'solver': hp.choice('solver', ['liblinear', 'saga']),
                'l1_ratio': hp.uniform('l1_ratio', 0.0, 1.0)
            }
        else:
            raise ValueError(f"Unsupported optimization framework: {optimization_framework}")
    
    def get_default_params(self) -> Dict[str, Any]:
        """Get default parameters for Logistic Regression."""
        return {
            'C': 1.0,
            'penalty': 'l2',
            'solver': 'liblinear',
            'max_iter': 1000
        }


class SVMConfig(BaseModelConfig):
    """Configuration for Support Vector Machine."""
    
    def __init__(self, task_type: str = 'classification', random_state: int = 42):
        super().__init__(random_state)
        self.task_type = task_type
    
    def create_model(self, **kwargs) -> Union[SVC, SVR]:
        """Create SVM model."""
        params = self.get_default_params()
        params.update(kwargs)
        params['random_state'] = self.random_state
        
        if self.task_type == 'classification':
            return SVC(**params)
        else:
            # del parameter random_state because SVR don't want
            if 'random_state' in params:
                del params['random_state']
            return SVR(**params)
    
    def get_hyperparameter_space(self, optimization_framework: str = 'optuna') -> Dict[str, Any]:
        """Get hyperparameter space for SVM."""
        if optimization_framework == 'optuna':
            return {
                'C': ('float_log', 0.001, 100.0),
                'kernel': ('categorical', ['linear', 'poly', 'rbf', 'sigmoid']),
                'gamma': ('categorical', ['scale', 'auto'] + [0.001, 0.01, 0.1, 1.0]),
                'degree': ('int', 2, 5)  # Only used for poly kernel
            }
        elif optimization_framework == 'hyperopt':
            from hyperopt import hp
            return {
                'C': hp.loguniform('C', np.log(0.001), np.log(100.0)),
                'kernel': hp.choice('kernel', ['linear', 'poly', 'rbf', 'sigmoid']),
                'gamma': hp.choice('gamma', ['scale', 'auto', 0.001, 0.01, 0.1, 1.0]),
                'degree': hp.randint('degree', 2, 5)
            }
        else:
            raise ValueError(f"Unsupported optimization framework: {optimization_framework}")
    
    def get_default_params(self) -> Dict[str, Any]:
        """Get default parameters for SVM."""
        return {
            'C': 1.0,
            'kernel': 'rbf',
            'gamma': 'scale'
        }


class ModelFactory:
    """Factory class for creating ML models and their configurations."""
    
    # Available model configurations
    MODEL_CONFIGS = {
        'random_forest': RandomForestConfig,
        'xgboost': XGBoostConfig,
        'lightgbm': LightGBMConfig,
        'catboost': CatBoostConfig,
        'logistic_regression': LogisticRegressionConfig,
        'svm': SVMConfig
    }
    
    def __init__(self, random_state: int = 42):
        """Initialize ModelFactory."""
        self.random_state = random_state
        
    def create_model_config(
        self, 
        model_name: str, 
        task_type: str = 'classification',
        **kwargs
    ) -> BaseModelConfig:
        """
        Create model configuration.
        
        Args:
            model_name: Name of the model
            task_type: 'classification' or 'regression'
            **kwargs: Additional arguments for model config
            
        Returns:
            Model configuration instance
        """
        if model_name not in self.MODEL_CONFIGS:
            available_models = list(self.MODEL_CONFIGS.keys())
            raise ValueError(f"Model '{model_name}' not available. "
                           f"Available models: {available_models}")
        
        # Validate task_type
        if task_type not in ['classification', 'regression']:
            raise ValueError(f"Invalid task_type '{task_type}'. "
                           f"Must be 'classification' or 'regression'")
        
        config_class = self.MODEL_CONFIGS[model_name]
        
        # Check if model supports the task type
        if model_name == 'logistic_regression' and task_type == 'regression':
            raise ValueError("Logistic regression only supports classification tasks")
        
        # Create config with appropriate parameters
        if model_name in ['random_forest', 'xgboost', 'lightgbm', 'catboost', 'svm']:
            return config_class(task_type=task_type, random_state=self.random_state, **kwargs)
        else:
            return config_class(random_state=self.random_state, **kwargs)
    
    def create_model(
        self, 
        model_name: str, 
        task_type: str = 'classification',
        params: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Any:
        """
        Create model instance.
        
        Args:
            model_name: Name of the model
            task_type: 'classification' or 'regression'
            params: Hyperparameters for the model
            **kwargs: Additional arguments
            
        Returns:
            Model instance
        """
        config = self.create_model_config(model_name, task_type, **kwargs)
        
        if params is None:
            params = {}
            
        return config.create_model(**params)
    
    def get_hyperparameter_space(
        self,
        model_name: str,
        task_type: str = 'classification',
        optimization_framework: str = 'optuna'
    ) -> Dict[str, Any]:
        """
        Get hyperparameter search space for a model.
        
        Args:
            model_name: Name of the model
            task_type: 'classification' or 'regression'
            optimization_framework: 'optuna' or 'hyperopt'
            
        Returns:
            Hyperparameter space definition
        """
        config = self.create_model_config(model_name, task_type)
        return config.get_hyperparameter_space(optimization_framework)
    
    def get_default_params(
        self,
        model_name: str,
        task_type: str = 'classification'
    ) -> Dict[str, Any]:
        """Get default parameters for a model."""
        config = self.create_model_config(model_name, task_type)
        return config.get_default_params()
    
    def list_available_models(self) -> List[str]:
        """List all available models."""
        return list(self.MODEL_CONFIGS.keys())
    
    def get_model_info(self, model_name: str) -> Dict[str, Any]:
        """Get information about a model."""
        if model_name not in self.MODEL_CONFIGS:
            raise ValueError(f"Model '{model_name}' not available")
        
        info = {
            'name': model_name,
            'supports_classification': True,
            'supports_regression': True,
            'requires_external_lib': False
        }
        
        # Update specific model information
        if model_name == 'logistic_regression':
            info['supports_regression'] = False
        elif model_name in ['xgboost', 'lightgbm', 'catboost']:
            info['requires_external_lib'] = True
            
        return info


# Convenience functions
def create_random_forest(task_type: str = 'classification', **params):
    """Create Random Forest model quickly."""
    factory = ModelFactory()
    return factory.create_model('random_forest', task_type, params)


def create_xgboost(task_type: str = 'classification', **params):
    """Create XGBoost model quickly."""
    factory = ModelFactory()
    return factory.create_model('xgboost', task_type, params)


def create_lightgbm(task_type: str = 'classification', **params):
    """Create LightGBM model quickly."""
    factory = ModelFactory()
    return factory.create_model('lightgbm', task_type, params)


if __name__ == "__main__":
    # Example usage
    import logging
    
    logging.basicConfig(level=logging.INFO)
    
    factory = ModelFactory(random_state=42)
    
    # List available models
    print("Available models:", factory.list_available_models())
    
    # Create models
    print("\n" + "="*50)
    
    # Random Forest
    rf_model = factory.create_model('random_forest', 'classification')
    print(f"Random Forest: {type(rf_model)}")
    
    # XGBoost (if available)
    try:
        xgb_model = factory.create_model('xgboost', 'classification')
        print(f"XGBoost: {type(xgb_model)}")
    except ImportError as e:
        print(f"XGBoost not available: {e}")
    
    # Get hyperparameter spaces
    print("\n" + "="*50)
    rf_space = factory.get_hyperparameter_space('random_forest', optimization_framework='optuna')
    print(f"Random Forest hyperparameter space: {list(rf_space.keys())}")
    
    # Get default parameters
    rf_defaults = factory.get_default_params('random_forest')
    print(f"Random Forest defaults: {rf_defaults}")
    
    # Model info
    print("\n" + "="*50)
    for model_name in factory.list_available_models():
        try:
            info = factory.get_model_info(model_name)
            print(f"{model_name}: {info}")
        except Exception as e:
            print(f"{model_name}: Error - {e}")