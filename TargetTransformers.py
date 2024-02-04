import numpy as np
from scipy.stats import boxcox
from scipy.special import inv_boxcox
class NatLog():
    def __init__(self):
        pass
    def Transform(self,y):
        return np.log(y)
    def InverseTransform(self,y):
        return np.exp(y)
    
class NatLogRecip():
    def __init__(self):
        pass
    def Transform(self,y):
        return np.log(1/y)
    def InverseTransform(self,y):
        return 1/np.exp(y)
    
class NoChange():
    def __init__(self):
        pass
    def Transform(self,y):
        return y
    def InverseTransform(self,y):
        return y


class Exp():
    def __init__(self):
        pass
    def Transform(self,y):
        return np.exp(y)
    def InverseTransform(self,y):
        return np.log(y)



class BoxCox():

    def __init__(self):
        self.lambda_ = None

    def Transform(self, y):
        # Apply Box-Cox transformation
        # The function also finds the best lambda for the transformation
        y_transformed, self.lambda_ = boxcox(y)
        return y_transformed

    def InverseTransform(self, y):
        # Check if lambda was set
        if self.lambda_ is None:
            raise ValueError("The data has not been transformed yet. Cannot perform inverse transform.")
        
        # Apply the inverse Box-Cox transformation
        return inv_boxcox(y, self.lambda_)
    
class NatLoNoiseg:
    def __init__(self):
        pass

    def Transform(self, y, noise_level=.3):
        """
        Applies natural logarithm transformation to y and adds Gaussian noise.

        :param y: Input array-like data.
        :param noise_level: Standard deviation of Gaussian noise to be added. Default is 0.0 (no noise).
        :return: Transformed data with optional noise.
        """
        transformed_y = np.log(y)
        if noise_level > 0:
            noise = np.random.normal(0, noise_level, y.shape)
            transformed_y += noise
        return transformed_y

    def InverseTransform(self, y):
        """
        Reverses the natural logarithm transformation.

        :param y: Transformed array-like data.
        :return: Original data before transformation.
        """
        return np.exp(y)