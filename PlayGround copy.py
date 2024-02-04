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
import matplotlib.pyplot as plt
import itertools
from sklearn.model_selection import train_test_split

FGs=np.loadtxt('./data/FGs.csv', delimiter=',')
Names=np.loadtxt('./data/Names.txt', delimiter='\t',dtype=str)


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
fig, ax = plt.subplots(1, 2, layout="constrained")
n=250
for i in range(n):
    FGs_train, FGs_test, Names_train, Names_test = train_test_split(FGs, Names, test_size=0.05)


    ######################Train GROUP#################################

    #EqDict={'TDE.PPDS14':tdeEq.PPDS14,'TDE.Watson':tdeEq.Watson}
    EqDict={'TDE.PPDS9':tdeEq.PPDS9,'TDE.NVCLSExpansion.Version1':tdeEq.ViscosityL}

    Train=DataSetBuilder(EqDict)

    Train.build_dataset(Names_train.tolist(),FGs_train,step=1)


    transformer=FX.OrrickErbarPower2()

    Train.BuildFeaturSpace(transformer)


    Train.TransformPredictorPipe(Ty.NatLog())

    Train.TrainModel(pm.SimpleLinearRegressor())

    y_pred_train=Train.Predict(Train.FeatureMatrix)

    #plots########################################################

    test=DataSetBuilder(EqDict)

    test.build_dataset(Names_test.tolist(),FGs_test,step=1)


    test.BuildFeaturSpace(transformer)

    test.TransformPredictorPipe(Ty.NatLog())

    y_pred=Train.Predict(test.FeatureMatrix)

    ytransformer=Ty.NatLog()





    print(f'_________________{i/n *100}% done______________________')
    # Plot each group with a unique color and marker
    ax[0].scatter(y_pred_train*1000, ytransformer.InverseTransform(Train.Y)*1000 , color='blue',alpha=.1)
    ax[1].scatter(y_pred*1000, ytransformer.InverseTransform(test.Y) *1000, color='orange', alpha=.1)

ax[0].set_xlabel('predicted (mPa.s)')
ax[0].set_ylabel('actual  (mPa.s)')
ax[0].plot([0, 10], [0, 10], linestyle='--',color='black')
ax[1].set_xlabel('predicted (mPa.s)')
ax[1].set_ylabel('actual  (mPa.s)')
ax[1].plot([0, 10], [0, 10], linestyle='--',color='black')
ax[0].set_title('Training Set')
ax[1].set_title('Test Set')
plt.suptitle('Train-Test Split of Elementary Fuels Viscosity')
plt.show()


"""
# Plot each group with a unique color and marker
for i in range(50):
    group_mask = (group_indices == i)
    plt.plot(Train.X[:,0], (Train.y*1000) , markersize=3,linestyle='--', color=colors(i), label=f'{Names[i]}')
    plt.plot(Train.X[:,0], (y_pred_train*1000) , markersize=3, linestyle='',marker=markers[i % len(markers)], color=colors(i), label=f'{Names[i]}')


plt.xlabel('Temperature (K)')
plt.ylabel('Surface Tension (N/m)')

plt.legend(bbox_to_anchor=(1.01, 1), loc='upper left', borderaxespad=0.,fontsize='x-small')
#figure size 

plt.show()
"""