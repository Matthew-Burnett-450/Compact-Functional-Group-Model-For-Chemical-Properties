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
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split


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




#normalize T such that it scale from 0 to 1

#fillter T less than 200 from Test.X and all other rows of Test.X




#80 20 split
#split the data into 80% training and 20% testing

transformer=FX.OrickNorm()


test.X,X_test,test.y,y_test=train_test_split(test.X,test.y,test_size=0.2,random_state=42)

test.BuildFeaturSpace(transformer)



X_test=transformer.Transform(X_test)



"""
########EDA##############################################
transformed_feature_names = [
    'Intercept', 
    'A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7',
    'B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7',
    'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7',
    'D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7',
    'A21', 'A22', 'A23', 'A24', 'A25', 'A26', 'A27',
    'B21', 'B22', 'B23', 'B24', 'B25', 'B26', 'B27', 
    'C21', 'C22', 'C23', 'C24', 'C25', 'C26', 'C27',
    'D21', 'D22', 'D23', 'D24', 'D25', 'D26', 'D27',
    'A31', 'A32', 'A33', 'A34', 'A35', 'A36', 'A37',
    'B31', 'B32', 'B33', 'B34', 'B35', 'B36', 'B37',
    'C31', 'C32', 'C33', 'C34', 'C35', 'C36', 'C37',
    'D31', 'D32', 'D33', 'D34', 'D35', 'D36', 'D37'
]
# Assuming test.FeatureMatrix is a NumPy array
feature_matrix_df = pd.DataFrame(test.FeatureMatrix, columns=[transformed_feature_names])  # Name the columns appropriately

# If test.y is a 1D NumPy array
target_df = pd.DataFrame(test.y, columns=['Target'])

df=pd.concat([feature_matrix_df,target_df],axis=1)
dfog=pd.concat([feature_matrix_df,target_df],axis=1)

#report = sv.analyze(df)
#report.show_html('Sweetviz_Report.html')  # Generates and opens a HTML report
num_removed=0
removed_indices = []  # List to keep track of removed feature indices
while num_removed < 5:
    # Calculate correlation matrix
    corr = df.corr(method='pearson')

    # Save correlation matrix to CSV
    corr.to_csv(f'corr{num_removed}.csv')

    # Find the feature with the closest to 0 correlation with the target
    corr_target = corr['Target'].drop('Target')  # Exclude the target variable itself
    feature_to_remove = corr_target.abs().idxmin()
    corr.to_csv(f'./corrdata/corr{num_removed}.csv')
    # Remove the feature from the DataFrame
    removed_index = dfog.columns.get_loc(feature_to_remove)
    removed_indices.append(removed_index)
    df.drop(columns=[feature_to_remove], inplace=True)
    

    # Increment the counter
    num_removed += 1
print(removed_index)
#remove the target from the df
df.drop(columns=['Target'], inplace=True)
#convert to numpy drop names
X=df.to_numpy()


test.FeatureMatrix=X"""



test.TransformPredictorPipe(Ty.NatLog())



test.TrainModel(pm.SimpleLinearRegressor())


y=test.Predict(test.FeatureMatrix)
y_test_pred=test.Predict(X_test)
#remove header names
########################################################
import matplotlib.pyplot as plt
import itertools

# Define a list of colors (length should be at least as long as the number of groups)
colors = ['grey']*1000

# Define a list of marker symbols
markers = 'oooooooooooooooooooooooooooooooooooooooooooooooooooooooooooo'

plt.figure(figsize=(8,6))
# Plot each group with a unique color and marker


plt.plot(test.y*1000, y*1000 ,alpha=.3, markersize=5,linestyle='None', marker='o', color='orange',label='Training Data')
plt.plot([1] ,alpha=0, label=f'R²: {r2_score(test.y,y) : .3f}')
plt.plot(y_test*1000, y_test_pred*1000 ,alpha=1, markersize=5,linestyle='None', marker='^', color='blue',label='Testing Data')
plt.plot([1] ,alpha=0, label=f'R²: {r2_score(y_test,y_test_pred) : .3f}')

plt.xlabel('predicted (mPa*s)',fontsize=28,fontweight='bold')
plt.ylabel('actual (mPa*s)',fontsize=28,fontweight='bold')
plt.plot([0, 5], [0, 5], 'k--')

plt.legend(frameon=False, fontsize=16)

for axis in ['top','bottom','left','right']:
    plt.gca().spines[axis].set_linewidth(2)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.minorticks_on()
plt.tick_params(axis='both', which='minor', direction='in')
plt.tick_params(axis='both', which='major', direction='in')
plt.tick_params(axis='both', which='major', length=4, width=2)
plt.tick_params(axis='both', which='minor', length=4, width=2)
plt.title('Compact Model Predictions', fontsize=36,weight='bold')
#figure size 

plt.show()

"""# Plot each group with a unique color and marker
for i in range(50):
    plt.plot(test.X [:,0], (test.y *1000) , markersize=3,linestyle='--', color=colors[i], label=f'{Names[i]}')
    plt.plot(test.X [:,0], (y *1000) , markersize=3, linestyle='',marker=markers[i % len(markers)], color=colors[i], label=f'{Names[i]}')


plt.xlabel('Temperature (K)')
plt.ylabel('Viscosity (mPa*s)')

plt.legend(bbox_to_anchor=(1.01, 1), loc='upper left', borderaxespad=0.,fontsize='x-small')
#figure size 

plt.show()"""



fuels = {
    'Jet-A POSF 10325': [[3.335467977, 1.140190557, 1.055652435, 1.590017098, 3.188735797, 0, 0],[273.15-40,273.15-20,40+273.15],[9.55,4.70,1.80]],
    'HRJ POSF 7720': [[3.271797853, 1.245511112, 1.393100123, 2.148655347, 3.720826794, 0, 0],[233.15,253.15,313.15],[14.00,6.10,1.50]],
}
colors=['green','blue']

plt.figure(figsize=(8,6))
linestyle=['.-','--']
marker='o^'
for num,i in enumerate(fuels):
    if i=='Jet-A POSF 10325':
        color='green'
    if i=='HRJ POSF 7720':
        color='blue'

    n=fuels[i][0]
    T=fuels[i][1]
    visc=fuels[i][2]
    #tile T and n to make X for every T add the n make T the first colum
    X=np.tile(n,(50,1))
    #remove the i
    #add T as the first column
    Temp=np.linspace(T[0],T[-1],50)[:,np.newaxis]
    X=np.hstack((Temp,X))
    X=transformer.Transform(X)
    #X=np.delete(X,removed_indices,axis=1)

    predV=test.Predict(X)
    plt.plot(T,visc,marker=marker[num],label=i,color='k',linestyle='None')
    plt.plot(Temp,predV*1000,linestyle[num],label=f'{i} Predicted',color='k')

plt.legend(frameon=False, fontsize=16)

for axis in ['top','bottom','left','right']:
    plt.gca().spines[axis].set_linewidth(2)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.minorticks_on()
plt.tick_params(axis='both', which='minor', direction='in')
plt.tick_params(axis='both', which='major', direction='in')
plt.tick_params(axis='both', which='major', length=4, width=2)
plt.tick_params(axis='both', which='minor', length=4, width=2)
plt.title('Compact Model Predictions', fontsize=36,weight='bold')
#figure size 

plt.show()
plt.xlabel('Temperature (K)')
plt.ylabel('Viscosity (mPa s)')
plt.show()




#save data from plots