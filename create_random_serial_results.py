import os
import math
import random
import argparse
import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn import metrics

def train_in_order(materialNames, allMaterials, randomized_order: bool, X_featureNames, Y_featureNames, num_of_instances = 10e6):

    # Initialize dictionary of errors per training data size
    mae_per_train_size = {}

    for indexOfTestInstance in range(min(len(materialNames), num_of_instances)):
        
        TrainArray = np.delete(materialNames, indexOfTestInstance)
        TestArray = materialNames[indexOfTestInstance]

        # Create a list containing a randomized or serial version of the training instance indices
        if randomized_order:
            zif_name_indices = random.sample(range(len(TrainArray)), len(TrainArray))
        else:
            zif_name_indices = range(len(TrainArray))

        currentTrainMaterials = None

        sizeOfTrainingDataset = 0
        # For each sample in the list
        for selected_zif_index_to_add in zif_name_indices:
            
            # Increase the size of the training dataset
            sizeOfTrainingDataset += 1
    
            newTrainZifName = TrainArray[selected_zif_index_to_add]
            newTrainZif = allMaterials[(allMaterials['type'] == newTrainZifName)]

            # Add currently selected zif to train dataframe
            if currentTrainMaterials is None:
                currentTrainMaterials = newTrainZif
            else:
                currentTrainMaterials = pd.concat([currentTrainMaterials, newTrainZif], axis=0, ignore_index=True)

            testMaterial = allMaterials[(allMaterials['type'] == TestArray)]
        
            # Empty dataframe error check.
            if  currentTrainMaterials.empty:
                raise RuntimeError("WARNING: Found empty newDb in iteration %d."%indexOfTestInstance)
            
            if  testMaterial.empty:
                raise RuntimeError("WARNING: Found empty testdf in iteration %d."%indexOfTestInstance)


            # Transform dataframes to np arrays.
            x_train = np.asanyarray(currentTrainMaterials[X_featureNames])
            y_train= np.array(currentTrainMaterials[Y_featureNames])

            x_test = np.asanyarray(testMaterial[X_featureNames])
            y_test= np.array(testMaterial[Y_featureNames])

            #Train the model.
            XGBR.fit(x_train, y_train.ravel())

            #Make predictions
            y_pred = XGBR.predict(x_test)

            # Average across all gases for the left out test instance
            mae = metrics.mean_absolute_error(y_test, y_pred)

            # Append mae to the corresponding dictionary list
            if sizeOfTrainingDataset not in mae_per_train_size.keys():
                mae_per_train_size[sizeOfTrainingDataset] = []

            mae_per_train_size[sizeOfTrainingDataset].append(mae)

            print("This is zif {}".format(sizeOfTrainingDataset))
            print(mae)
            
        print("This is round {}".format(indexOfTestInstance))
        
    result_df = pd.DataFrame()
    result_df["sizeOfTrainingSet"] = np.array([iCnt for iCnt in sorted(mae_per_train_size.keys()) ])
    result_df["averageError"] = [ np.array(mae_per_train_size[iCnt]).mean() for iCnt in mae_per_train_size.keys() ]
    result_df["stdErrorOfMeanError"] = [ np.array(mae_per_train_size[iCnt]).std() / math.sqrt(iCnt) for iCnt in mae_per_train_size.keys() ]      

    return result_df

if __name__ == "__main__":

        # Command line parameters
        parser = argparse.ArgumentParser()

        parser.add_argument('-t', '--train',    help='A file path containing the train data.', default='./TrainData.xlsx')
        parser.add_argument('-p', '--path',     help='The path for the save file.', default='./')
        parser.add_argument('-s', '--save',     help='Set the file name to be appended at the end of the results. Dont forget the extension .csv', default='.csv')
        parser.add_argument('-d', '--datatype', help='Select the data type of train data. Accepted inputs are 1. zif_diffusivity 2. zif_diffusivity_no_butane 3. zif_diffusivity_per_gas 4. mof_hydrogen', default='zif_diffusivity')
        parsed_args = parser.parse_args() # Actually parse

        trainData    = parsed_args.train
        savePath     = parsed_args.path
        saveName     = parsed_args.save
        dataType     = parsed_args.datatype

        XGBR = XGBRegressor(n_estimators=500, max_depth=5, eta=0.07, subsample=0.75, colsample_bytree=0.7, reg_lambda=0.4, reg_alpha=0.13,
                            n_jobs=6,
                         #  nthread=6,
                            random_state=6410
                            )


        trainDf = pd.DataFrame
        trainFileExtension = trainData.split('.')[-1]
        if trainFileExtension == 'excel' or trainFileExtension == 'xlsx':
            trainDf=pd.read_excel(trainData)
        elif trainFileExtension == 'csv':
            trainDf=pd.read_csv(trainData)
        else:
            raise RuntimeError("Error: Accepted data file types are excel and csv.")

        # Default X Y feature names
        Y = ["logD"]
        X = ['diameter','mass','ascentricF', 'kdiameter','ionicRad',
             'MetalNum','MetalMass','σ_1', 'e_1','linker_length1', 
             'linker_length2', 'linker_length3','linker_mass1', 
             'linker_mass2', 'linker_mass3','func1_length', 
             'func2_length', 'func3_length','func1_mass', 
             'func2_mass', 'func3_mass']
        perGasTrainDf = {}
        
        if dataType in ['zif_diffusivity', 'zif_diffusivity_no_butane']:

            trainDf['logD'] = np.log10(trainDf['diffusivity'])
            trainDf=trainDf[[ 'type', 'gas', 'aperture', 'MetalNum', 'MetalMass', 'size - van der Waals (Å)','mass', 'ascentricF', 'logD', 'size - kinetic diameter (Å)', 'ionicRad', 
                     'Μ-N_lff', 'Μ-N_kFF', 'MetalCharge',
                     'σ_1', 'e_1', 'linker_length1', 'linker_length2',
                     'linker_length3', 'linker_mass1', 'linker_mass2', 'linker_mass3',
                     'func1_length', 'func2_length', 'func3_length', 'func1_mass',  
                     'func2_mass', 'func3_mass', 'func1_charge', 'func2_charge',
                     'func3_charge']]
            trainDf = trainDf.rename(columns={'size - van der Waals (Å)':'diameter', 'size - kinetic diameter (Å)':'kdiameter', 'apertureAtom_e':'e' })
            trainDf = trainDf.dropna()
            trainDf.drop(trainDf[trainDf['type'] == 'dFm_Be'].index, inplace = True)
            trainDf.drop(trainDf[trainDf['type'] == 'Cd-I-ZIF-7-8'].index, inplace = True)

            trainDf=trainDf.reset_index(drop=True)

            if dataType == 'zif_diffusivity_no_butane':
                trainDf = trainDf[trainDf["gas"] != "butane"]

        elif dataType == 'mof_hydrogen':

            Y = ["UG"]
            X = ["Density", "GSA", "VSA","VF",
                "PV", "LCD", "PLD"]

        elif dataType == 'zif_diffusivity_per_gas':
            
            for filename in os.listdir("./DataPerGas"):
                gas = filename.split(".")[0]
                print("Gathering Data For File: " + gas)

                perGasTrainDf = pd.read_excel(os.path.join("./DataPerGas",str(filename)))
                perGasTrainDf=perGasTrainDf.rename(columns={'size - van der Waals (Å)':'diameter', 'size - kinetic diameter (Å)':'kdiameter', 'apertureAtom_e':'e' })
                materialNames = perGasTrainDf.type.unique()

                perGasTrainDf[gas] = materialNames


        if dataType == 'zif_diffusivity_per_gas':
            
            for data, materialNames in perGasTrainDf.items():
                # Train the model adding data points in random order.
                randomized_result_df = train_in_order(materialNames, data, True, X, Y,)        
                # Save random order results
                randomized_result_df.to_csv(os.path.join(savePath,saveName) + gas + '.csv', index=False)

                # Train the model adding data points in serial order.
                serial_result_df  = train_in_order(materialNames, data, False, X, Y,)
                # Save serial order results
                serial_result_df.to_csv(os.path.join(savePath,saveName) + gas + '.csv', index=False)

        if dataType in ['zif_diffusivity', 'zif_diffusivity_no_butane', 'mof_hydrogen']:

            materialNames = trainDf.type.unique()

            # Train the model adding data points in random order.
            randomized_result_df = train_in_order(materialNames, trainDf, True, X, Y,)        
            # Save random order results
            randomized_result_df.to_csv(os.path.join(savePath,saveName), index=False)

            if dataType != 'mof_hydrogen':
                # Train the model adding data points in serial order.
                serial_result_df  = train_in_order(materialNames, trainDf, False, X, Y,)
                # Save serial order results
                serial_result_df.to_csv(os.path.join(savePath,saveName), index=False)
