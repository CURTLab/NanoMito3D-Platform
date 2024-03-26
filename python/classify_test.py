import sys 
import os 

sys.path.append(".") 
os.add_dll_directory(os.environ['CUDA_PATH'] + '/bin')

import pynanomito as nm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import cv2

def load_model(fname):
    """
    Load a random forest model from a file.

    Args:
        fname (str): The file name of the model.

    Returns:
        cv2.ml.RTrees: The loaded machine learning model.

    Raises:
        ValueError: If the model type is unknown.
    """
    if os.path.splitext(fname)[1] == '.json':
        return cv2.ml.RTrees_load(fname)
    elif os.path.splitext(fname)[1] == '.xml':
        return cv2.ml.RTrees_load(fname)
    elif os.path.splitext(fname)[1] == '.csv':
        model = cv2.ml.RTrees_create()
        df = pd.read_csv('mitoTrainDataSet.csv').to_numpy()
        model.train(df[:,:-1].astype(np.float32), cv2.ml.ROW_SAMPLE, df[:,-1, np.newaxis].astype(np.int32))
        return model
    else:
        raise ValueError('Unknown model type')
    
def classify(fname, model):
    """
    Classify the given localization microscopy file (*.tsf) using the specified model.

    Args:
        fname (str): The file name to be classified.
        model: The model used for classification.

    Returns:
        dict: A dictionary containing the percentage of voxels classified as 'Puncta', 'Rods', and 'Networks'.
    """
    ret = nm.segment(fname, verbose=False)
    ret['predictions'] = model.predict(ret.to_numpy()[:, 0:14].astype(np.float32))[1]
    class1 = ret[ret['predictions'] == 1]['voxels'].sum()
    class2 = ret[ret['predictions'] == 2]['voxels'].sum()
    class3 = ret[ret['predictions'] == 3]['voxels'].sum()
    return {'Puncta': class1 * 100.0 / (class1 + class2 + class3),
            'Rods': class2 * 100.0 / (class1 + class2 + class3),
            'Networks': class3 * 100.0 / (class1 + class2 + class3)}

#model_name = '../examples/mitoTrainDataSet.csv'
model_name = '../examples/mito-model.json'

fname = '../examples/Fig4A_20220323_CD62p_Mitos_phEC_Static_dSTORM_c1000_red_blue_016_v3_bleedCorr.tsf'

model = load_model(model_name)
classes = classify(fname, model)
print(classes)

fig = plt.figure(figsize = (5, 5))
 
plt.bar(classes.keys(), classes.values(), color = [(0,1,0), (0.25, 0.96, 0.82), (0,0,1)])
 
plt.ylabel('Percentage')
plt.title('Classification of voxels')
plt.show()