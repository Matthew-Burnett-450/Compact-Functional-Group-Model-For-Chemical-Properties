import xmltodict
import numpy as np
import TDEEquations as tdeEq
import ThermoMlPropEqGrabber as tml
import PropertyModels as pm
import FeatureSpaceConstructors as FX
import TargetTransformers as Ty

class DataSetBuilder():
    def __init__(self,StateEquations):
        self.StateEquations=StateEquations
    def build_dataset(self,xmlfiles,AddiitonalFeatures,step=1):

        if len(AddiitonalFeatures)!=len(xmlfiles):
            raise ValueError('The number of rows of Addiitonal Features and xml files must be equal')
        X=np.array([])
        y=np.array([])
        GroupIndexs=np.array([])
        for Compound in xmlfiles:

            name=Compound.replace('"','').replace("'",'')
            tml_parser = tml.ThermoMLParser(f'./data/{name}.xml')
            tml_parser.extract_properties()
            tml_parser.extract_equation_details()

            Properties=tml_parser.get_properties()


            idx = np.where(np.isin(Properties['equation_names'], list(self.StateEquations.keys())))[0]
            EqName = [Properties['equation_names'][i] for i in idx]

            idx=Properties['equation_names'].index(EqName[0])

            Params=Properties['equation_parameters'][idx]
            VarRange=Properties['variable_ranges'][idx]

            idx=xmlfiles.index(Compound)
            GroupIndex=idx
            AddiitonalFeatureRow = AddiitonalFeatures[idx]

            

            # Generate a linspace for each variable range and stack them horizontally.
            x = np.column_stack([np.arange(var_min, var_max, step) for var_min, var_max in VarRange])

            AddiitonalFeatureStack = np.repeat(AddiitonalFeatureRow[np.newaxis, :], len(x), axis=0)
            GroupIndexStack = np.repeat(GroupIndex, len(x), axis=0)
            # Concatenate the linspace matrix with the repeated functional groups horizontally.
            x = np.hstack((x, AddiitonalFeatureStack))
            StateEquation=self.StateEquations[EqName[0]]
            Y=StateEquation.run(VarRange,Params,step=step)
            

            

            if X.size==0:
                X=x
                y=Y
                GroupIndexs=GroupIndexStack
            else:
                X=np.vstack((X,x))
                y=np.append(y,Y)
                GroupIndexs=np.append(GroupIndexs,GroupIndexStack)

        self.X=X
        self.y=y
        self.GroupIndexs=GroupIndexs


    def BuildFeaturSpace(self,XConstructor):
        self.FeatureMatrix=XConstructor.Transform(self.X)

    def TransformPredictorPipe(self,Pipe):
        self.Pipe=Pipe
        self.Y=Pipe.Transform(self.y)

    def TrainModel(self,Model,*args):
        Y=self.Y
        X=self.FeatureMatrix
        self.Model=Model
        self.Model.fit(X,Y,*args)

    def getParams(self):
        return self.Model.get_params()
    
    def Predict(self,X):
        return self.Pipe.InverseTransform(self.Model.predict(X))

