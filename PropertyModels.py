from sklearn.linear_model import LinearRegression,Ridge,RidgeCV,ElasticNet,SGDRegressor
from sklearn.model_selection import KFold,GroupShuffleSplit
import numpy as np
from sklearn.svm import LinearSVR
from sklearn.model_selection import GridSearchCV


class SimpleLinearRegressor:
    def __init__(self):
        """
        Initialize the SimpleLinearRegressor with an option to fit the intercept.
        
        :param fit_intercept: bool, default False
            Whether to calculate the intercept for this model. If set to False,
            no intercept will be used in calculations (i.e., data is expected to be centered).
        """
        self.model = LinearRegression(fit_intercept=False,)
        self.params = None
    def fit(self, features, target_values, *args):
        """
        Fit the linear model.
        
        :param features: array-like, shape (n_samples, n_features)
            Training data.
        :param target_values: array-like, shape (n_samples,) or (n_samples, n_targets)
            Target values.
        :return: None : None
        """
        self.model.fit(features, target_values)
        self.params = list(self.model.coef_)


    def predict(self, features):
        """
        Predict using the linear model.
        
        :param features: array-like, shape (n_samples, n_features)
            Samples.
        :return: C : array, shape (n_samples,)
            Returns predicted values.
        """
        return self.model.predict(features)

    def get_params(self):
        """
        Get the parameters of the fitted linear model.
        
        :return: list
            Returns a list of fitted coefficients.
        """
        return self.params
    
class LinearRegressionCVShuffleSplit:
    """
    This class implements linear regression with cross-validation to provide 
    a robust set of parameters that generalize well to unseen data.
    
    Attributes:
        n_splits: The number of random splits for cross-validation.
        fit_intercept: Whether to fit the intercept or not.
        models: A list to store the models trained during cross-validation.
        weighted_params: The weighted average of model parameters based on cross-validation scores.
    """
    def __init__(self, groups, n_splits=500, fit_intercept=False):
        """
        Initialize the LinearRegressionCV with the number of splits for cross-validation 
        and whether to fit the intercept.
        : param groups: array-like, shape (n_samples,)
            Group labels for the samples used while splitting the dataset into train/test set.
        :param n_splits: int, default 1000
            The number of splits for cross-validation.
        :param fit_intercept: bool, default False
            Whether to calculate the intercept for this model.
        """
        self.GroupIndexs=groups
        self.n_splits = n_splits
        self.fit_intercept = fit_intercept
        self.models = []
        self.weighted_params = None

    def fit(self, X, y):
        """
        Fit the linear model using cross-validation.
        
        :param X: array-like, shape (n_samples, n_features)
            Training data.
        :param y: array-like, shape (n_samples,) or (n_samples, n_targets)
            Target values.
        :return: None
        """
        kf = GroupShuffleSplit(n_splits=self.n_splits,test_size=.01)
        scores = []
        param_list = []

        for train_index, test_index in kf.split(X, y, groups=self.GroupIndexs):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            
            model = LinearRegression(fit_intercept=self.fit_intercept)
            model.fit(X_train, y_train)
            score = model.score(X_test, y_test)
            scores.append(score)
            param_list.append(model.coef_)

        # Weighted averaging of parameters
        scores = 1/ np.array(scores)
        normalized_scores = scores / scores.sum()
        self.weighted_params = np.average(param_list, axis=0, weights=normalized_scores)

    def predict(self, X):
        """
        Predict using the linear model with weighted parameters.
        
        :param X: array-like, shape (n_samples, n_features)
            Samples for making predictions.
        :return: array, shape (n_samples,)
            Returns predicted values.
        """
        if self.weighted_params is None:
            raise Exception("The model has not been fitted yet.")
        return X @ self.weighted_params

    def get_params(self):
        """
        Get the weighted parameters of the fitted linear model.
        
        :return: array
            Returns the weighted average of model coefficients.
        """
        if self.weighted_params is None:
            raise Exception("The model has not been fitted yet.")
        return self.weighted_params

class ElasticNetRegressor:
    """
    ElasticNet regression is a linear regression model trained with both l1 and l2 -norm regularization of the coefficients.
    
    Parameters:
    -----------
    alpha : float, default=1.0
        Constant that multiplies the penalty terms. Defaults to 1.0.
        `alpha = 0` is equivalent to an ordinary least square, solved by the LinearRegression object.
        
    l1_ratio : float, default=0.5
        The ElasticNet mixing parameter, with `0 <= l1_ratio <= 1`.
        `l1_ratio=0` corresponds to L2 penalty, `l1_ratio=1` to L1.
        
    Attributes:
    -----------
    model : ElasticNet object
        The underlying ElasticNet model.
    
    coef_ : ndarray of shape (n_features,) or (n_targets, n_features)
        Parameter vector (w in the cost function formula).
    
    intercept_ : float or ndarray of shape (n_targets,)
        Independent term in the linear model.
    """
    
    def __init__(self, alpha=1e-6, l1_ratio=0.5):
        self.model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio)
        self.coef_ = None
        self.intercept_ = None

    def fit(self, X, y):
        """
        Fit ElasticNet model.
        
        Parameters:
        -----------
        X : array-like or sparse matrix, shape (n_samples, n_features)
            Training data.
            
        y : array-like, shape (n_samples,) or (n_samples, n_targets)
            Target values.
        
        Returns:
        --------
        self : object
            Returns self.
        """
        self.model.fit(X, y)
        self.coef_ = self.model.coef_
        self.intercept_ = self.model.intercept_
        return self

    def predict(self, X):
        """
        Predict using the ElasticNet model.
        
        Parameters:
        -----------
        X : array-like or sparse matrix, shape (n_samples, n_features)
            Samples.
        
        Returns:
        --------
        C : ndarray of shape (n_samples,)
            Returns predicted values.
        """
        return self.model.predict(X)

    def get_params(self):
        """
        Get parameters for this estimator.
        
        Returns:
        --------
        params : dict
            Parameter names mapped to their values.
        """
        return self.model.get_params()

class LinearSVRRegressor:
    """
    This class implements a Linear Support Vector Regression model.
    
    Attributes:
        model: The LinearSVR model from scikit-learn.
    """
    def __init__(self, **kwargs):
        """
        Initialize the LinearSVRRegressor with any keyword arguments accepted by sklearn's LinearSVR.
        
        :param kwargs: Keyword arguments for LinearSVR's constructor (e.g., C, tol, max_iter).
        """
        self.model = LinearSVR(**kwargs)

    def fit(self, X, y):
        """
        Fit the LinearSVR model.
        
        :param X: array-like, shape (n_samples, n_features)
            Training data.
        :param y: array-like, shape (n_samples,) or (n_samples, n_targets)
            Target values.
        """
        self.model.fit(X, y)

    def predict(self, X):
        """
        Predict using the LinearSVR model.
        
        :param X: array-like, shape (n_samples, n_features)
            Samples.
        :return: array, shape (n_samples,)
            Returns predicted values.
        """
        return self.model.predict(X)

    def get_params(self):
        """
        Get the parameters of the fitted LinearSVR model.
        
        :return: dict
            Returns the parameters of the model.
        """
        return self.model.get_params()
    
class RidgeRegressor:
    """
    This class implements a Ridge Regression model.
    
    Attributes:
        model: The Ridge regression model from scikit-learn.
    """
    def __init__(self, alpha=1.0, **kwargs):
        """
        Initialize the RidgeRegressor with the regularization strength 'alpha' and any other keyword arguments accepted by sklearn's Ridge.
        
        :param alpha: float, default=1.0
            Regularization strength; must be a positive float. Larger values specify stronger regularization.
        :param kwargs: Keyword arguments for Ridge's constructor (e.g., fit_intercept, normalize, max_iter).
        """
        self.model = Ridge(alpha=alpha, **kwargs)

    def fit(self, X, y):
        """
        Fit the Ridge regression model.
        
        :param X: array-like, shape (n_samples, n_features)
            Training data.
        :param y: array-like, shape (n_samples,) or (n_samples, n_targets)
            Target values.
        """
        self.model.fit(X, y)

    def predict(self, X):
        """
        Predict using the Ridge regression model.
        
        :param X: array-like, shape (n_samples, n_features)
            Samples.
        :return: array, shape (n_samples,)
            Returns predicted values.
        """
        return self.model.predict(X)

    def get_params(self):
        """
        Get the parameters of the fitted Ridge regression model.
        
        :return: dict
            Returns the parameters of the model, including 'alpha'.
        """
        return self.model.get_params()

    def set_params(self, **params):
        """
        Set the parameters of the Ridge regression model.
        
        :param params: dict
            Model parameters to be set.
        """
        self.model.set_params(**params)

class RidgeCVRegressor:
    """
    This class implements Ridge Regression with built-in cross-validation of the alpha parameter.
    
    Attributes:
        model: The RidgeCV regression model from scikit-learn.
        alphas: Array of alpha values to try.
    """
    def __init__(self, alphas=(1e-5,1e-4,1e-3,1e-2,1e-1,1), **kwargs):
        """
        Initialize the RidgeCVRegressor with an array of alphas for cross-validation 
        and any other keyword arguments accepted by sklearn's RidgeCV.
        
        :param alphas: array-like, default=(0.1, 1.0, 10.0)
            Array of alpha values to determine the strength of regularization.
        :param kwargs: Keyword arguments for RidgeCV's constructor (e.g., fit_intercept, normalize, cv).
        """
        self.model = RidgeCV(alphas=alphas, **kwargs)

    def fit(self, X, y):
        """
        Fit the RidgeCV regression model.
        
        :param X: array-like, shape (n_samples, n_features)
            Training data.
        :param y: array-like, shape (n_samples,) or (n_samples, n_targets)
            Target values.
        """
        self.model.fit(X, y)

    def predict(self, X):
        """
        Predict using the RidgeCV regression model.
        
        :param X: array-like, shape (n_samples, n_features)
            Samples.
        :return: array, shape (n_samples,)
            Returns predicted values.
        """
        return self.model.predict(X)

    def get_params(self):
        """
        Get the parameters of the fitted RidgeCV regression model.
        
        :return: dict
            Returns the parameters of the model, including the best 'alpha' found during CV.
        """
        return self.model.get_params()

    def get_best_alpha(self):
        """
        Get the best alpha value found during cross-validation.
        
        :return: float
            Returns the best alpha value found during CV.
        """
        return self.model.alpha_
    
class LinearSGDRegressor:
    def __init__(self):
        """
        Initialize the SGDRegressor with any keyword arguments accepted by sklearn's SGDRegressor.
        
        :param kwargs: Keyword arguments for SGDRegressor's constructor (e.g., loss, penalty, alpha, l1_ratio, max_iter, tol).
        """
        self.model = SGDRegressor(max_iter=10000, tol=0.000001)
    def fit(self, X, y):
        """
        Fit the SGDRegressor model.
        
        :param X: array-like, shape (n_samples, n_features)
            Training data.
        :param y: array-like, shape (n_samples,) or (n_samples, n_targets)
            Target values.
        """
        self.model.fit(X, y)
    def predict(self, X):
        """
        Predict using the SGDRegressor model.
        
        :param X: array-like, shape (n_samples, n_features)
            Samples.
        :return: array, shape (n_samples,)
            Returns predicted values.
        """
        return self.model.predict(X)
    def get_params(self):
        """
        Get the parameters of the fitted SGDRegressor model.
        
        :return: dict
            Returns the parameters of the model.
        """
        return self.model.get_params()
    

