import numpy as np
import TDEEquations as tdeEq
import ThermoMlPropEqGrabber as tml
import PropertyModels as pm
import FeatureSpaceConstructors as FX
import TargetTransformers as Ty
from DataSetBuilder import DataSetBuilder
import matplotlib.pyplot as plt
import pandas as pd
import sweetviz as sv


FGs=np.loadtxt('./data/FGs.csv', delimiter=',')
Names=np.loadtxt('./data/Names.txt', delimiter='\t',dtype=str)

#check for FGs rows where colum 5 is 0 if so remove that row and the same row index from names

# Names that you want to remove
names_to_remove = [
    'ethylbenzene',
    'decalin',
    '2-methylbutane',
    'n-propylbenzene',
]

# Create a boolean mask where True indicates the row should be kept
mask = ~np.isin(Names, names_to_remove)

# Apply the mask to filter FGs and Names
FGs = FGs[mask]
Names = Names[mask]


#EqDict={'TDE.PPDS14':tdeEq.PPDS14,'TDE.Watson':tdeEq.Watson}
EqDict={'TDE.PPDS9':tdeEq.PPDS9,'TDE.NVCLSExpansion.Version1':tdeEq.ViscosityL}
test=DataSetBuilder(EqDict)

test.build_dataset(Names.tolist(),FGs,step=.5)

#sum each functional group row
sFGs=np.sum(FGs,axis=1)





group_indices = test.GroupIndexs[test.X[:,0]>0]


transformer=FX.Custom()

test.BuildFeaturSpace(transformer)

test.TransformPredictorPipe(Ty.NatLog())

X=test.FeatureMatrix
y=test.Y



#use sklearn recursive feature elimination to find the best features
from sklearn.feature_selection import RFECV,RFE
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold
#gradiant boost
from sklearn.ensemble import GradientBoostingRegressor
# Create the RFE object and compute a cross-validated score.
# The "accuracy" scoring is proportional to the number of correct
# classifications
#gourp train test split

#train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=69)




model = GradientBoostingRegressor(random_state=0, n_estimators=10000)
regressor = model.fit(X_train, y_train)
print(regressor.get_params())

#plot train test 
y_pred_train=regressor.predict(X_train)
y_pred_test=regressor.predict(X_test)

plt.figure(figsize=(10,10))
plt.plot(1000*np.exp(y_train),1000*np.exp(y_pred_train),'o',label='Train')
plt.plot(1000*np.exp(y_test),1000*np.exp(y_pred_test),'o',label='Test')
plt.xlabel('True')
plt.ylabel('Predicted')
#dashed diagonal to 10
plt.plot([0,10],[0,10],'k--')
plt.legend()
plt.show()

print(f'RMSE Train: {mean_squared_error(y_train, y_pred_train,squared=False)}')
print(f'RMSE Test: {mean_squared_error(y_test, y_pred_test,squared=False)}')
print(f'MAE Train: {mean_absolute_error(y_train, y_pred_train)}')
print(f'MAE Test: {mean_absolute_error(y_test, y_pred_test)}')
print(f'R2 Train: {r2_score(y_train, y_pred_train)}')
print(f'R2 Test: {r2_score(y_test, y_pred_test)}')

#num of features

fuels = {
    'Jet-A POSF 10325': [[3.335467977, 1.140190557, 1.055652435, 1.590017098, 3.188735797, 0, 0],[273.15-40,273.15-20,40+273.15],[9.55,4.70,1.80]],
    'HRJ POSF 7720': [[3.271797853, 1.245511112, 1.393100123, 2.148655347, 3.720826794, 0, 0],[233.15,253.15,313.15],[14.00,6.10,1.50]],
}
colors=['green','blue']




for i in fuels:
    if i=='Jet-A POSF 10325':
        color='green'
    if i=='HRJ POSF 7720':
        color='blue'

    n=fuels[i][0]
    T=fuels[i][1]
    visc=fuels[i][2]
    #tile T and n to make X for every T add the n make T the first colum
    X=np.tile(n,(100,1))
    #remove the i
    #add T as the first column
    Temp=np.linspace(T[0],T[-1],100)[:,np.newaxis]
    X=np.hstack((Temp,X))
    X=transformer.Transform(X)
    #X=np.delete(X,removed_indices,axis=1)

    predV=regressor.predict(X)
    plt.plot(T,visc,'o',label=i,color=color)
    plt.plot(Temp,np.exp(predV)*1000,'--',label=f'{i} Predicted',color=color)
    plt.legend()
plt.xlabel('Temperature (K)')
plt.ylabel('Viscosity (mPa s)')
plt.show()

