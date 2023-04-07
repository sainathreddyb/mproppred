import logging
import numpy as np
import pandas as pd

def get_logger():
    """
    Creates and returns a logger object.
    """
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.info('This is a test log message')
    return logger

def get_accuracy_mae(testsit):
    """
    Calculates and prints the unscaled mean absolute error (MAE) and the accuracy of the predictions for the 'pKi' and 'pIC50' measurement types.
    
    Parameters:
    testsit (DataFrame): Dataframe containing the test data with columns 'SMILES', 'measurement_value', 'preds', 'measurement_type', and 'Kinase_name'.
    
    Returns:
    None
    """
    testsit['abs']=abs(testsit['measurement_value']-testsit['preds'])
    print("unscaled mae: "+str(testsit['abs'].mean()))
    
    for kinase_name in ['pKi','pIC50']:
        test_pki = testsit[testsit.measurement_type==kinase_name]

        count=0
        total=0

        for i in np.unique(test_pki.SMILES):
            temp1= test_pki[test_pki.SMILES==i]
            if len(temp1)==1:
                continue
            max_predicted_Kinase = temp1.loc[temp1['preds'] == temp1['preds'].max()].Kinase_name.values[0]
            max_actual_Kinase = temp1.loc[temp1['measurement_value'] == temp1['measurement_value'].max()].Kinase_name.values[0]

            if max_actual_Kinase==max_predicted_Kinase:
                count=count+1
            total+=1
        print('Accuracy of '+kinase_name+" :  " + str(count/total))


        