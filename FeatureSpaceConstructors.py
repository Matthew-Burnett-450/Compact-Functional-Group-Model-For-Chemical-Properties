import numpy as np
from sklearn.preprocessing import PolynomialFeatures,StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectFpr, r_regression,f_regression
from sklearn.feature_selection import mutual_info_regression


class MainFeatures():

    def __init__(self):
        pass
    def Transform(self,X): 
        X
        return X
    

############################################################################################################## VISCOUS PROPERTIES##############################################################################################################
class Yaws():
    """

    Transforms the Feature Matrix to a pseudo Yaws equation representation
    For Viscoscity Liquid-Gas Equilibrium
    >> A + B/T + C*T + D*T^2 <<
    
    Assumes the first column is temperature (T) and the rest are functional groups (n)
    A = a dot n
    B = b dot n
    C = c dot n
    D = d dot n
    
    """
    def __init__(self):
        pass
    def Transform(self,X): 
        T=X[:,0][:,np.newaxis]

        Intercept=np.ones((len(T),1))

        A=X[:,1:]

        B=A/T

        C=A*T

        D=A*T**2

        return np.hstack((Intercept,A,B,C,D))
    
class YawsPower():
    """

    Transforms the Feature Matrix to a pseudo Yaws equation representation
    For Viscoscity Liquid-Gas Equilibrium
    >> A + B/T + C*T + D*T^2 <<
    
    Assumes the first column is temperature (T) and the rest are functional groups (n)
    A = a dot n
    B = b dot n
    C = c dot n
    D = d dot n
    
    """
    def __init__(self):
        pass
    def Transform(self,X): 
        T=X[:,0][:,np.newaxis]

        Intercept=np.ones((len(T),1))

        A=X[:,1:]

        B=A/T

        C=A*T

        D=A*T**2

        E=A*T**3

        A2=A**2

        B2=A2/T

        C2=A2*T

        D2=A2*T**2

        E2=A2*T**3
        
        return np.hstack((Intercept,A,B,C,D,E,A2,B2,C2,D2,E2))


class OrrickErbar():
    """

    Transforms the Feature Matrix to a pseudo OrrickErbar equation representation
    For Viscoscity Liquid-Gas Equilibrium
    >> A + B/T + C/T^2 + D/T^3 <<
    
    Assumes the first column is temperature (T) and the rest are functional groups (n)0
    A = a dot n
    B = b dot n
    C = c dot n
    D = d dot n
    
    """
    def __init__(self):
        pass
    def Transform(self,X): 
        T=X[:,0][:,np.newaxis]

        Intercept=np.ones((len(T),1))

        A=X[:,1:]

        B=A/T
        
        C=A/T**2

        D=A/T**3

        return np.hstack((Intercept,A,B,C,D))


class OrrickErbarPower():
    """

    Transforms the Feature Matrix to a pseudo OrrickErbar equation representation with additional power terms
    For Viscoscity Liquid-Gas Equilibrium
    >> A + B/T + C/T^2 + D/T^3 + A^2 + B^2/T + C^3/T^2 + D^3/T^3 <<
    
    Assumes the first column is temperature (T) and the rest are functional groups (n)
    A = a dot n
    B = b dot n
    C = c dot n
    D = d dot n
    
    """
    def __init__(self):
        pass
    def Transform(self,X): 
        T=X[:,0][:,np.newaxis]

        Intercept=np.ones((len(T),1))

        invT=1/T

        invT2=1/T**2

        invT3=1/T**3

        A=X[:,1:]

        B=A/T

        C=A/T**2

        D=A/T**3

        A2=A**2

        B2=B**2/T

        C2=C**2/T**2

        D2=D**2/T**3

        A3=A**3

        B3=B**3/T**2

        C3=C**3/T**2

        D3=D**3/T**3

        return np.hstack((Intercept,invT,invT2,invT3,A,B,C,D,A2,B2,C2,D2,A3,B3,C3,D3))

def transform_features2(X):
    T = X[:, 0].reshape(-1, 1)
    
    # Perform the operations
    B = X / T
    C = X / np.square(T)
    D = X / np.power(T, 3)
    
    # Horizontally stack the original X with the transformed data
    X_transformed = np.hstack((X, B, C,D))
    
    # Grab every 8th column
    Intercepts = X_transformed[:, 7::8]
    
    # Set the first column of every_eighth_col to all ones
    Intercepts[:, 0] = 1
    Intercepts=Intercepts[:,0][:,np.newaxis]
    # Now remove every 8th column from the original transformed X
    X_nointercepts = np.delete(X_transformed, np.s_[7::8], axis=1)
    
    # Finally, horizontally stack the modified X with the extracted columns at the end
    # Perform the operations
    A2= np.square(X[:,1:]) 
    B2 = np.square(X[:,1:]) / T
    C2 = np.square(X[:,1:]) / np.square(T)

    A3= np.power(X[:,1:],3) 
    B3 = np.power(X[:,1:],3) / T
    C3 = np.power(X[:,1:],3) / np.square(T)
    X_final = np.hstack((X_nointercepts, Intercepts,A2,B2,C2,A3,B3,C3))
    return X_final

def transform_features4(X):
    T = X[:, 0].reshape(-1, 1)
    
    # Calculate various transformations of X based on T
    B = X / T                     # Divide each column of X by T
    C = X / np.square(T)          # Divide each column of X by T^2
    D = X / np.power(T, 3)        # Divide each column of X by T^3
    
    # Combine original and transformed data
    X_transformed = np.hstack((X, B, C, D))  # Stack X, B, C, D horizontally

    Intercepts = X_transformed[:, 7::8]
    Intercepts[:, 0] = 1  # Set the first column to 1
    Intercepts = Intercepts[:, 0][:, np.newaxis]

    # Remove the last column of each block from X_transformed
    X_nointercepts = np.delete(X_transformed, np.s_[7::8], axis=1)
    
    # and their divisions by T and T^2
    squared_features = np.square(X[:, 1:])
    squared_div_T = squared_features / T
    squared_div_T2 = squared_features / np.square(T)

    cubed_features = np.power(X[:, 1:], 3)
    cubed_div_T = cubed_features / T
    cubed_div_T2 = cubed_features / np.square(T)

    # Stack all the features to form the final transformed dataset
    X_final = np.hstack((X_nointercepts, Intercepts, squared_features, squared_div_T, 
                         squared_div_T2, cubed_features, cubed_div_T, cubed_div_T2))
    return X_final

class OrrickErbarPower2():
    """

    Transforms the Feature Matrix to a pseudo OrrickErbar equation representation with additional power terms
    For Viscoscity Liquid-Gas Equilibrium
    >> A + B/T + C/T^2 + D/T^3 + A^2 + B^2/T + C^3/T^2 + D^3/T^3 <<
    
    Assumes the first column is temperature (T) and the rest are functional groups (n)
    A = a dot n
    B = b dot n
    C = c dot n
    D = d dot n
    
    """
    def __init__(self):
        pass
    def Transform(self,X): 
        T=X[:,0][:,np.newaxis]



        Intercept=np.ones((len(T),1))


        A=X[:,1:]

        B=A/T

        C=A/T**2

        D=A/T**3

        A2=A**2

        B2=B**2/T

        C2=C**2/T**2

        D2=D**2/T**3

        A3=A**3

        B3=B**3/T**2

        C3=C**3/T**2

        D3=D**3/T**3
        


        Xf=np.hstack((Intercept, A, B, C, D, A2, B2, C2, D2, A3, B3, C3, D3))
        print(Xf.shape)
        return Xf


class OrickNorm():
    """

    Transforms the Feature Matrix to a pseudo Andre equation representation
    For Viscoscity Liquid-Gas Equilibrium
    >> A + B/(T+C) <<
    
    Assumes the first column is temperature (T) and the rest are functional groups (n)
    A = a dot n
    B = b dot n
    C = c dot n
    D = d dot n

    n is normalized and the sum of n is appended to the end of the feature matrix
    
    """
    def __init__(self):
        pass
    def Transform(self,X): 

        return transform_features4(X)
    

class Custom():
    """

    Transforms the Feature Matrix to a pseudo Andre equation representation
    For Viscoscity Liquid-Gas Equilibrium
    >> A + B/(T+C) <<
    
    Assumes the first column is temperature (T) and the rest are functional groups (n)
    A = a dot n
    B = b dot n
    C = c dot n
    D = d dot n

    n is normalized and the sum of n is appended to the end of the feature matrix
    
    """
    def __init__(self):
        pass
    def Transform(self,X): 
        T=X[:,0][:,np.newaxis]
        invT=1/T
        A=X[:,1:]
        Xin=np.hstack((A,invT,T))
        Xf=PolynomialFeatures(degree=3,interaction_only=True,include_bias=True).fit_transform(Xin)
   
        print(Xf.shape)
        Xf=np.hstack((T,Xf))
        return Xf
    
class Custom2():
    def __init__(self,y=None):
        self.selected_features = None
        self.first_transform = True
        self.y = y 
    def Transform(self, X, ):
        if self.y is None and self.first_transform:
            raise ValueError("Target variable y is required for the first transform.")

        T = X[:, 0][:, np.newaxis]
        invT = 1 / T
        A = X[:, 1:]
        Xin = np.hstack((A, invT, T))

        # Generate polynomial features
        X_poly = PolynomialFeatures(degree=3, interaction_only=True, include_bias=True).fit_transform(Xin)

        if self.first_transform:
            # Compute MI scores during the first transform
            mi_scores = mutual_info_regression(X_poly, self.y,n_neighbors=10)

            # Select features based on a threshold
            threshold_mi = 0.05  # Example threshold
            self.selected_features = np.where(mi_scores > threshold_mi)[0]

            # Update the flag
            self.first_transform = False

        # Apply feature selection
        X_selected = X_poly[:, self.selected_features]

        print(X_selected.shape)
        return X_selected
    




############################################################################################################## SURFACE TENSION PROPERTIES##############################################################################################################
class SurfT():

    def __init__(self):
        pass
    def Transform(self,X): 
        T=X[:,0][:,np.newaxis]
        intercept=np.ones((len(T),1))

        A=X[:,1:]
        B=A*T


        Xf=np.hstack((intercept,A,B))
        return Xf