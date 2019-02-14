from DatasetFunctions import *
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import f1_score
import datetime
import time
'''     
    ***********************************************************
    ***     Compare two Manually Annotated (MA) Datasets    ***
    ***********************************************************
    
    Read datasets from labeled dataset folders with (empty) files (just to have the labels in appropriate format) created with GetFromMedx.py
    Make datasets parallel, i.e. comparable, which means that:
        - The same label (CUI) has the same collumn index in both datasets (target ndarray)
        - The same pmid has the same row index in both datasets (target ndarray)
    Calculate inter-annotator agreement (Cohen's Kappa) per label
'''
'''
#   *****************************************************
#   ***     Hardcoded settings for the classifier     ***
#   *****************************************************
'''

base = 'C:/Users/tasosnent/Desktop/Tasks/12 PhD/MeSH_to_UMLS/ManualDS/AD/v1'
annotators =[
    'Natasha',
    'Nikil',
]
# Make Pandas print all columns of a DataFrame when less or equal to 20
pd.set_option('display.max_columns', 20)

'''
#   **************************************
#   ***     Load the data from files   ***
#   **************************************
'''

labelDatasets = {}
for annotator in annotators:
    print("Load dataset from : ", base+'/'+annotator)
    test_dataset, labelNames = loadFromTextFiles(base+'/'+annotator)
    # Covert dataset to multi-label
    print("Covert dataset to multi-label")
    # Merge articles included in more than one folders into one with both labels
    # and replace "filenames" by just "pmids"
    makeDatasetMultiLabel(test_dataset)
    # Print Bunch fields and content
    # print("Bunch fields and content loaded:")
    # print("\t keys", test_dataset.keys())
    # print("\t target names", test_dataset.target_names)
    # print("\t filenames", test_dataset.filenames)
    # print("\t target", test_dataset.target)
    # print("\t size", len(test_dataset.data))

    '''
    #   *********************************
    #   ***     Make labels Binary    ***
    #   *********************************
    '''
    print('Make datasets binary')
    # Use MultiLabelBinarizer to create binary 2d matrix with one column per class as done here : [ref](http://scikit-learn.org/stable/modules/multiclass.html#multilabel-classification-format)
    test_dataset.target = MultiLabelBinarizer(classes = range(len(test_dataset.target_names))).fit_transform(test_dataset.target)
    labelDatasets[annotator] = test_dataset
    # print("\t keys", test_dataset.keys())
    # print("\t target names", test_dataset.target_names)
    # print("\t target", test_dataset.target)
    # print("\t filenames", test_dataset.filenames)
    # print("\t size", len(test_dataset.data))
    # print("\t target shape", test_dataset.target.shape)

'''
#   *************************************
#   ***     Make parallel datasets    ***
#   *************************************
'''
# Define an order for labels and pmids
# It is the one found in the 1st annotator
firstAnnotatorIndex = 0
stableDataset = labelDatasets[annotators[firstAnnotatorIndex]]
pmidsOrder = stableDataset.filenames
labelsOrder = stableDataset.target_names
print ('pmids',pmidsOrder)
print ('labels (CUIs)',labelsOrder)
# Reorder the dataset of the 2nd annotator to match the order of the 1st
secondAnnotatorIndex = 1
parallelDataset = labelDatasets[annotators[secondAnnotatorIndex]]
disorderedFilenames = parallelDataset.filenames.copy()
disorderedTarget_names = parallelDataset.target_names.copy()
disorderedTarget = parallelDataset.target.copy()
# disordereData = [] Data are anyway empty, no need to be reordered

# print('Bef',parallelDataset)
# print('Bef',parallelDataset.filenames)

# Reorder columns to match the label order
for labelCUI in disorderedTarget_names:
    # find the "correct" index of the label
    oldIndex = disorderedTarget_names.index(labelCUI)
    newIndex = labelsOrder.index(labelCUI)

    # Copy target values in the correct column
    parallelDataset.target[:,newIndex] = disorderedTarget[:,oldIndex]
    parallelDataset.target_names[newIndex] = labelCUI

# Reorder rows to match the pmid order
for pmid in disorderedFilenames:
    # find the "correct" index of the pmid
    oldIndex = disorderedFilenames.index(pmid)
    newIndex = pmidsOrder.index(pmid)

    # Copy target values in the correct column
    parallelDataset.target[newIndex,:] = disorderedTarget[oldIndex,:]
    parallelDataset.filenames[newIndex] = pmid

# print('Aft',parallelDataset)
# print('Aft',parallelDataset.filenames)

'''
#   *******************************
#   ***     Find Agreement      ***
#   *******************************
'''
print('Find Agreement')
f= open(base + "/Agreement.csv","w+")
f.write('Label CUI' + ';' + "Cohen\'s Kappa" + ';' + "F1 measure"  + ';' + annotators[firstAnnotatorIndex]  + ';' + annotators[secondAnnotatorIndex] + ';' + "Difference" + ';' + "Difference pmids" + '\n')
print('label/CUI', ' K', ' F1', annotators[firstAnnotatorIndex] , annotators[secondAnnotatorIndex] )
for labelCUI in labelsOrder:
    index = labelsOrder.index(labelCUI)
    a_targets = stableDataset.target[:,index]
    b_targets = parallelDataset.target[:,index]
    # Cohen's Kappa
    k = cohen_kappa_score(a_targets,b_targets)
    # F1 measure
    f1 = f1_score(a_targets,b_targets)
    # find difference between annotators
    diff = abs(a_targets - b_targets)
    diff_pmids = []
    for i in range(len(diff)):
        if diff[i]:
            diff_pmids.append(pmidsOrder[i])

    # show and write results
    print(labelCUI, k,  f1, a_targets.sum(),  b_targets.sum() )
    f.write(labelCUI + ';' + str(k) + ';' + str(f1)  + ';'+ str(a_targets.sum()) + ';'+ str(b_targets.sum()) + ';'+ str(diff.sum() ) + ';' + str(diff_pmids)+ '\n')

f.close()

