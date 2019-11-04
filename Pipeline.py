from DatasetFunctions import *
import pandas as pd
from sklearn.feature_selection import chi2, f_classif
import time
import datetime

'''
#   ***********************************
#   ***     Hardcoded settings      ***
#   ***********************************
'''
#   *******************************
#   ***   Case study details    ***
#   *******************************

# # Alzheimers Disease
# # The dominant label CUI to be used for baseline construction (e.g. C0013264 for DMD). Use a non existing CUI (e.g. dominantLabel = 'CCC')to be ignored
# caseStudy = 'AD';
# # The cui of the preferred concept to be ignored in MA2 creation as well as in some baselines.
# dominantLabel = 'C0002395' # Use a non existing CUI (e.g. dominantLabel = 'CCC')to be ignored
# # The cui that is higher in the Ct Hierarcy to be "excluded" from the classification modeling and evaluation
# cTop = 'C0002395'

# Duchenne Muscular Dystrophy
caseStudy = 'DMD';
# The cui of the preferred concept to be ignored in MA2 creation as well as in some baselines.
dominantLabel = 'C0013264'  # Use a non existing CUI (e.g. dominantLabel = 'CCC')to be ignored
# The cui that is higher in the Ct Hierarcy to be "excluded" from the classification modeling and evaluation
cTop = 'C3542021'

#   *******************************
#   ***     Experiment details  ***
#   *******************************

#   ***     Experiment folder  ***

# For the AD use case :
# To create all datasets from initial CSV files
# baseFolder = './Data/AD/'
# To re-use pre-processed datasets for DMD experiments replace with path of corresponding folder in your system
# baseFolder = '.../AD/LexicalAndSemanticFeatures/'
# baseFolder = '.../AD/LexicalAndSemanticFeaturesNoCi/'
# baseFolder = '.../AD/LexicalAndSemanticWSund/'
# baseFolder = '.../AD/LexicalFeaturesOnly/'

# For the DMD use case :
# To create all datasets from initial CSV files
baseFolder = './Data/DMD/'
# To re-use pre-processed datasets for DMD experiments replace with path of corresponding folder in your system
# baseFolder = '.../DMD/LexicalAndSemanticFeatures/'
# baseFolder = '.../DMD/LexicalAndSemanticFeaturesNoCi/'
# baseFolder = '.../DMD/LexicalAndSemanticWSund/'
# baseFolder = '.../DMD/LexicalFeaturesOnly/'

#   ***     Steps to do  ***

datasetCreate = 1  # STEP 1 : Whether to Create the dataset from initial CSV files or skipp this step
Transform = 1  # STEP 2 : Whether to transform/vectorize the training dataset (tokenization/TFIDF) and save
Analyze = 1  # STEP 3 : Whether to perform Feature Selection on the dataset and save
readMA = 1  # STEP 4 : Whether to parse Manual Annotations for a MA dataset
TransformMA = 1  # STEP 5 : Whether to transform the MA dataset based on the training datasets
classify = 1  # STEP 6 : Whether to train and evaluate classifiers

# Alternative Numbers of features to select in Feature selection (You can use just one value)
featureKs = [
    5,
    10,
    20,
    50,
    100,
    500,
    1000,
]

# The score functions to be used for Univariate feature selection
scoreFunctions = {
    'chi2': chi2,
    'f': f_classif,
}

#   ***     For Evaluation  ***

test_folder_names = ['MA1', 'MA2']  # The datasets to be transformed and analyzed for classifier evaluation

#   *******************************
#   ***     Naming conventions  ***
#   *******************************

# Initial CSV files for dataset creation
class_csv = baseFolder + caseStudy + '_Cts_inT.csv';
labelNames_csv = baseFolder + caseStudy + '_labelNames.csv';
fields_csv = baseFolder + caseStudy + '_abstract.csv';
# CSV file with concept occurrence frequency for each article (used for CUI feature addition)
occurrenceFrequency_csv = baseFolder + caseStudy + '_concepts.csv';

# Directories with article text into pmid name files, grouped into folders named after the class CUI
output_dir = baseFolder + 'ArticlesPerClass/' + caseStudy;
# class name for articles without occurrences
rest_name = 'rest';
rest_dir = output_dir + '/' + rest_name;
# Final directories with article text into pmid name files, grouped into folders named after the class CUI
final_dir = baseFolder + "FinalSplit/" + caseStudy
train_dir = final_dir + "/train"  # Directories with article text into pmid name files, grouped into folders named after the class CUI
test_dir = final_dir + "/test"  # Directories with article text into pmid name files, grouped into folders named after the class CUI
train_dir_vect = final_dir + '/trainVectorized';  # Final directory with analyzed data in vector representation (e.g. as TFIDF features)

# Feature prefix (used to discriminate between tokens and other added features)
feature_prefix = 'AddedFeatureCUI:'

# Make Pandas print all columns of a DataFrame when less or equal to 20
pd.set_option('display.max_columns', 20)

'''     
    STEP 1

    ******************************************************************
    ***     Create and Split Distantly Supervised (DS) Datasets    ***
    ******************************************************************

    This script is intended to create and split Distantly Supervised datasets of articles with concept level topic annotations
        Initial steps before running this script to create a data set:
            1) Create a CSV file (..._Cts_inT.csv) with articles per class based on Ct occurence in articles for t. 
            2) Create a CSV file (..._abstract.csv) with articles for topic t with harvested fields e.g. abstract text
            3) Create (manually) a CSV file (..._labelNames.csv) with CUI-abbreviation correspondence (i.e. the class/label names) 
            4) Create a CSV file (..._concepts.csv) with concepts per article. For each article add a record with the pmid and the list of conepts with corresponding frequency separated by ":" (e.g. 15305790	["C0243140:2","C0002395:1"])

    This Script combines for each article the classes read from the first CSV and the data (e.g. abstract and title) from the second CSV, creating a Dataset.
        A text file is created for each article named after its PMID and located in a folder named after the class CUI
        After loading the data, the annotations are converted into Multi-Labeled to be adequate for training Multi-Labeled Classifiers
        Manually annotated articles are removed from dataset to be used for testing and saved into a separate folder for further processing. 
        In particular: 
            1)  A MA1 dataset is selected randomly and removed from the dataset
            2)  A MA2 dataset is selected balancing the label-sets, excluding the preferred label, and removed from the dataset. To not exclude the preferred label, put a non existing CUI in the corresponding setting (i.e. preferredConceptCUI).

        The remaining articles are split into Trainig and Testing datasets retaining the Label-Set Distribution if a proportion different from 0.0 is provided in the setting.
'''

#   ***     For dataset creation  ***

# The ratio of the "test set" for splitting into training and test set.
test_ratio = 0;  # Set to zero (test_ratio = 0) for no splitting at all.
# test_ratio = 0.3;

# Manually annotated articles to be excluded from Weakly Supervised dataset and held out data
# The pmids of existing MA dataset, if any.
# manual_dataset_pmids = baseFolder + 'MA_pmids.txt'; # Use false if no list of MA_pmids.txt exist
manual_dataset_pmids = False;  # Use false if no list of MA_pmids.txt exist
manual_dataset1_pmids = baseFolder + 'MA1_pmids.txt';  # Comment this command if no list of MA_pmids.txt exist
manual_dataset2_pmids = baseFolder + 'MA2_pmids.txt';  # Comment this command if no list of MA_pmids.txt exist
# manual_dataset1_pmids = False; # Comment this command if list of MA_pmids.txt exist
# manual_dataset2_pmids = False; # Comment this command if list of MA_pmids.txt exist
# The size of MA1 and MA2 datasets to be selected, if any.
manual_dataset_1_size = 0;  # leave zero (manual_dataset_1_size = 0) to skipp MA1 selection
manual_dataset_2_size = 0;  # leave zero (manual_dataset_2_size = 0) to skipp MA2 selection

# Folders of Manually Annotated Datasets
manual_dataset_dir = final_dir + "/MA/";
manual_dataset_1_dir = final_dir + "/MA1/";
manual_dataset_2_dir = final_dir + "/MA2/";
# File paths for Manually Annotated Dataset pmid lists
manual_dataset_dir_pmids = manual_dataset_dir + "pmids.txt";
manual_dataset_1_dir_pmids = manual_dataset_1_dir + "pmids.txt";
manual_dataset_2_dir_pmids = manual_dataset_2_dir + "pmids.txt";
removed_pmids = final_dir + "removed_pmids.txt";
# Under-sampling: Number of "preferred class (only)" articles to be removed. If 0, no under-sampling performed.
majority_articles_subsample_size = 0

if datasetCreate:
    print(datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'),
          "-->> Start Trainig data creation")
    createDataset(labelNames_csv, class_csv, output_dir, manual_dataset_dir, manual_dataset_1_dir, manual_dataset_2_dir,
                  manual_dataset_pmids, fields_csv, rest_dir, manual_dataset_1_size, manual_dataset_1_dir_pmids,
                  manual_dataset_2_size, dominantLabel,
                  manual_dataset_2_dir_pmids, test_ratio, train_dir, test_dir, manual_dataset1_pmids,
                  manual_dataset2_pmids, majority_articles_subsample_size, removed_pmids)
    print(datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'), "-->> End Trainig data creation")
else:
    print(datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'),
          "-->> Trainig data creation skipped")

# Leave the "cTop" folder in training folder for model creation, these articles should be used as negative for other classes.
# TODO: Handle case where some labels are not found at all!

'''
    STEP 2

    ********************************************
    ***     Transform a Training dataset     ***
    ********************************************

    Make target a binary 2d matrix with one column per class
    Tokenize the text of each sample and calculate TFIDF features for the tokens
    Also use concept-occurrence features (from occurrenceFrequency_csv) 
    Save the Transformed dataset in a new directory
'''
useCUIS = True;  # Whether to add (concept occurrence) semantic features or not
binaryFrequency = True;  # Whether to use the binary frequency ( i.e. existence/non existence) or the actual frequency of concept occurrence
Undersample = False;  # Whether to perform (additional) under-sampling or not [ATTENTION: Not used in the reported experiments. majority_articles_subsample_size is used instead!]

if Transform:
    print(datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'),
          "-->> Start Trainig data tranformation")
    tranformTrainingDataset(train_dir, train_dir_vect, useCUIS, binaryFrequency, Undersample, occurrenceFrequency_csv,
                            feature_prefix)
    print(datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'),
          "-->> End Trainig data tranformation")
else:
    print(datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'),
          "-->> Trainig data tranformation skipped")

'''
    STEP 3

    ***********************************
    ***     Analyze the Dataset     ***
    ***********************************

    Perform Feature Selection and create new training datasets with selected Features only
'''
# Exclude the CUIs used for weak labeling (ci) from feature space
ignoreLabelFeatures = False;

if Analyze:
    print(datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'),
          "-->> Start Trainig data analysis")
    analyzeTrainigDataset(train_dir_vect, featureKs, scoreFunctions, feature_prefix, ignoreLabelFeatures)
    print(datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'), "-->> End Trainig data analysis")
else:
    print(datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'),
          "-->> Trainig data analysis skipped")

'''

    ***************************************
    ***     Details for MA datasets     ***
    ***************************************

'''
for test_folder_name in test_folder_names:
    print(datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'),
          "-->> Run for ", test_folder_name )

    evalLabels_csv = baseFolder + test_folder_name + "_Data.csv"

    # Testset folders
    wsTest_dir = final_dir + '/' + test_folder_name  # The folder with the DS labels for the validation dataset
    # wsTest_dir = final_dir + '/' + test_folder_name + '/ws' # The folder with the DS labels for the validation dataset
    labeledTest_dir = final_dir + '/' + test_folder_name + '_labeled'  # Directory with article text into pmid name files, grouped into folders named after the class CUI
    vectorizedTest_dir = final_dir + '/' + test_folder_name + '_vectorized';  # Final directory with analyzed data in vector representation (e.g. as TFIDF features)

    # Directories of corresponding training datasets to read Vocabulary and IDF for data transformation
    trainDatasets = {
        # 'NoFS_0': train_dir_vect
    }

    evalDatasets = {
        # 'NoFS_0': final_dir +  '/'+test_folder_name+'_vectorized_NoFS_0',
    }
    #   ***     Dynamically produce dataset paths   ***
    for scoreFunction in scoreFunctions.keys():
        for featureK in featureKs:
            datasetName = scoreFunction + "_" + str(featureK)
            trainDatasets[datasetName] = train_dir_vect + "_FSby_" + scoreFunction + "_" + str(featureK)
            evalDatasets[datasetName] = final_dir + '/' + test_folder_name + "_vectorized_" + scoreFunction + "_" + str(
                featureK)

    '''    
        STEP 4

        *********************************************************
        ***     Create a Manually Annotated (MA) Dataset      ***
        *********************************************************

        Read the text of articles included in the MA Dataset from corresponding folder (created during WS Dataset creation)
        Read the MA classes/labels from a CSV file created manually
        Write the MA Dataset in text files, in format adequate to be read by  
    '''

    if readMA:
        print(datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'),
              "-->> Start Test data creation from Manual Annotations ")
        creatTestDataByMA(train_dir_vect, wsTest_dir, evalLabels_csv, labeledTest_dir)
        print(datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'),
              "-->> End Test data creation from Manual Annotations")
    else:
        print(datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'),
              "-->> Test data creation from Manual Annotations skipped ")

    '''     
        STEP 5 

        ************************************
        ***     Transform MA dataset     ***
        ************************************
        Make target a binary 2d matrix with one column per class
        Tokenize the text of each sample and calculate TFIDF features for the tokens
        Save the Transformed dataset in a new direcotry
    '''
    if TransformMA:
        print(datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'),
              "-->> Start Test data transformation based on the trainig dataset")
        tranformTestData(trainDatasets, labeledTest_dir, occurrenceFrequency_csv, feature_prefix, vectorizedTest_dir,
                         binaryFrequency)
        print(datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'),
              "-->> End Test data transformation based on the trainig dataset")
    else:
        print(datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'),
              "-->> Test data transformation Skipped ")

    '''     
        STEP 6

        ***********************************
        ***     Train a Classifier      ***
        ***********************************
    '''

    clfTypes = {
        # 1. A Linear Support Vector Classifier
        'LinearSVC': LinearSVC(),
        # # 2. A Random Forest Classifier
        'RandomForestClassifier': RandomForestClassifier(),
        # # # 3. A desision Tree Classifier
        'DecisionTreeClassifier': DecisionTreeClassifier(),
        # 4. A LogisticRegression Classifier
        'LogisticRegression': LogisticRegression(),
    }

    #     ** Cross Validation parameters **
    cvs = [10] # Number of folds for cross-validation on training dataset

    #     ** LogisticRegression (LRC) parameters **
    # Alternative C values for regularization in LogisticRegression (default = 1)
    # regCs = [0.01, 0.1, 0.15, 0.3, 0.5, 1, 1.1, 1.5, 2, 5, 10, 15, 20, 50, 100,200,500,1000,2000,5000, 10000,20000,50000,100000,200000, 500000, 1000000,2000000,5000000, 10000000,20000000, 50000000, 100000000]
    regCs = [1]
    # Alternative C values for regularization type in LogisticRegression (default = l2)
    # regType = "l1"
    regType = "l2"

    # Folder used for iteration experiments
    train_dir_vect_relabeled = final_dir + '/trainVectorized_relabeled';  # Final directory with analyzed data in vector representation (e.g. as TFIDF features) with predicted labels

    if classify:
        print(datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'),
              "-->> Start Classifier creation and evaluation")

        # Train classifiers for each desired C value for regularization
        for regC in regCs:
            print("\t\t ", datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'), "for regularization C = ", regC, " and regularization type ", regType)
            createClassifiers( trainDatasets, evalDatasets, final_dir, test_folder_name, wsTest_dir, dominantLabel, cTop, clfTypes, regC, regType, cvs)

            # Train classifiers iteratively, on the predictions of previous classifiers
            # createClassifier('f_100', trainDatasets['f_100'],'f_100', evalDatasets['f_100'], final_dir, test_folder_name, wsTest_dir, dominantLabel, cTop, 'LogisticRegression', LogisticRegression(), train_dir_vect_relabeled, 0.15, regType,  0)
            # createClassifier('f_100', train_dir_vect_relabeled,'f_100', evalDatasets['f_100'], final_dir, test_folder_name, wsTest_dir, dominantLabel, cTop, 'LogisticRegression', LogisticRegression(), train_dir_vect_relabeled, 1, regType,  1)
            # createClassifier('f_100', train_dir_vect_relabeled,'f_100', evalDatasets['f_100'], final_dir, test_folder_name, wsTest_dir, dominantLabel, cTop, 'LogisticRegression', LogisticRegression(), train_dir_vect_relabeled, 1, regType,  2)
            # createClassifier('f_100', train_dir_vect_relabeled,'f_100', evalDatasets['f_100'], final_dir, test_folder_name, wsTest_dir, dominantLabel, cTop, 'LogisticRegression', LogisticRegression(), train_dir_vect_relabeled, 1, regType,  3)
            # createClassifier('f_100', train_dir_vect_relabeled,'f_100', evalDatasets['f_100'], final_dir, test_folder_name, wsTest_dir, dominantLabel, cTop, 'LogisticRegression', LogisticRegression(), train_dir_vect_relabeled, 1, regType,  4)

        print(datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'),
              "-->> End Classifier creation and evaluation")
    else:
        print(datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'),
              "-->> Classifier creation and evaluation skipped ")