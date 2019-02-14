from DatasetFunctions import *
import pandas as pd
from sklearn.feature_selection import chi2, f_classif, mutual_info_classif
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

# Alzheimers Disease
caseStudy = 'AD';
# The dominant label CUI to be used for baseline construction (e.g. C0013264 for DMD). Use a non existing CUI (e.g. dominantLabel = 'CCC')to be ignored
dominantLabel = 'C0002395' # Use a non existing CUI (e.g. dominantLabel = 'CCC')to be ignored
# The cui of the preferred concept to be ignored in MA2 creation. Use an invalid CUI to by-pass exlussion of the default CUI from MA2 creation.
# preferredConceptCUI = 'C0002395' # Use an invalid CUI to by-pass exclusion of the default CUI from MA2 creation.
# The cui that is higher in the Ct Hierarcy to be "excluded" from
#   1) The MA2 dataset creation (previously names "preferredConceptCUI")
#   2) The classification model results
#   3) The baseline results
cTop = 'C0002395'

# # The cui of the preferred concept to be ignored in MA2 creation. Use an invalid CUI to by-pass exlussion of the default CUI from MA2 creation.
# preferredConceptCUI = 'C00023959' # Use a non existing CUI (e.g. dominantLabel = 'CCC')to be ignored
# # The dominant label CUI to be used for baseline construction (e.g. C0013264 for DMD)
# dominantLabel = 'C0002395' # Use a non existing CUI (e.g. dominantLabel = 'CCC')to be ignored

#   *******************************
#   ***     Experiment details  ***
#   *******************************

#   ***     Experiment folder  ***

# For AD
# baseFolder = '/LexicalAndSemanticFeature/';
baseFolder = '/LexicalFeaturesOnly/';

#   ***     Steps to do  ***

datasetCreate = 0   #   STEP 1 : Whether to Create the dataset from initial CSV files or skipp this step
Transform = 0       #   STEP 2 : Whether to transorm/vectorize the training dataset (tokenization/TFIDF) and save
Analyze = 0         #   STEP 3 : Whether to perform Feature Selection on the dataset and save
readMA = 0          #   STEP 4 : Whether to parse Manual Annotations for a MA dataset
TransformMA = 0     #   STEP 5 : Whether to transofm the MA dataset based on the training datasets
classify = 1        #   STEP 6 : Whether to train and evaluate classifiers


#   ***     For dataset creation  ***

# The ratio of the "test set" for splitting into training and test set.
test_ratio = 0; # Set to zero (test_ratio = 0) for no splitting at all.
# test_ratio = 0.3;

# Manually annotated articles to be excluded from Weakly Supervised dataset and held out data
# The pmids of existing MA dataset, if any.
# manual_dataset_pmids = baseFolder + 'MA_pmids.txt'; # Use false if no list of MA_pmids.txt exist
manual_dataset_pmids = False; # Use false if no list of MA_pmids.txt exist
manual_dataset1_pmids = baseFolder + 'MA1_pmids.txt'; # Comment this command if no list of MA_pmids.txt exist
manual_dataset2_pmids = baseFolder + 'MA2_pmids.txt'; # Comment this command if no list of MA_pmids.txt exist
# The size of MA1 and MA2 datasets to be selected, if any.
manual_dataset_1_size = 0; # leave zero (manual_dataset_1_size = 0) to skipp MA1 selection
manual_dataset_2_size = 0; # leave zero (manual_dataset_2_size = 0) to skipp MA2 selection

#   ***     For dataset transformation/vectorization  ***

useCUIS = 1; # Whether to add concept occurrence features or not

#   ***     For Feature Selection  ***

# Alternative Numbers of features to select in Feature selection (You can use just one value)
featureKs=[
    5,
    10,
    20,
    50,
    100,
    500,
    1000,
]

# The score functions to be used for Univariate feature selection
scoreFunctions ={
    'chi2':chi2,
    'f':f_classif,

    # 'mi':mutual_info_classif
}

#   ***     For Evaluation  ***

test_folder_name = 'MA1' # The dataset to be transformed and analyzed for classfiviers evaluation
    #     !!! ATTENTION !!! : Create the corresponding inputText_dir with the weak superivision folders for this testset!!!

#   *******************************
#   ***     Naming conventions  ***
#   *******************************

# Initial CSV files for dataset creation
class_csv = baseFolder + caseStudy + '_Cts_inT.csv';
labelNames_csv = baseFolder + caseStudy + '_labelNames.csv';
fields_csv = baseFolder + caseStudy + '_abstract.csv';
# CSV file with concept occurrence for each article (used for CUI feature addition)
occurrence_csv = baseFolder + caseStudy + '_T.csv';
evalLabels_csv = baseFolder + test_folder_name + "_Data.csv"

# Directories with article text into pmid name files, grouped into folders named after the class CUI
output_dir = baseFolder + 'ArticlesPerClass/' + caseStudy;
# class name for articles without occurrences
rest_name = 'rest';
rest_dir = output_dir + '/' + rest_name;
# Final directories with article text into pmid name files, grouped into folders named after the class CUI
final_dir = baseFolder + "FinalSplit/" + caseStudy
train_dir = final_dir + "/train" # Directories with article text into pmid name files, grouped into folders named after the class CUI
test_dir = final_dir + "/test" # Directories with article text into pmid name files, grouped into folders named after the class CUI
train_dir_vect = final_dir + '/trainVectorized'; # Final directory with analyzed data in vector representation (e.g. as TFIDF features)

# Folders of Manually Annotated Datasets
manual_dataset_dir = final_dir + "/MA/";
manual_dataset_1_dir = final_dir + "/MA1/";
manual_dataset_2_dir = final_dir + "/MA2/";
# File paths for Manually Annotated Dataset pmid lists
manual_dataset_dir_pmids = manual_dataset_dir + "pmids.txt";
manual_dataset_1_dir_pmids = manual_dataset_1_dir + "pmids.txt";
manual_dataset_2_dir_pmids = manual_dataset_2_dir + "pmids.txt";

# Feature prefix (used to discriminate between tokens and other added features)
feature_prefix = 'AddedFeatureCUI:'

# Testset folders
wsTest_dir = final_dir + '/' + test_folder_name # The folder with the DS labels for the validation dataset
# wsTest_dir = final_dir + '/' + test_folder_name + '/ws' # The folder with the DS labels for the validation dataset
labeledTest_dir = final_dir + '/' + test_folder_name + '_labeled' # Directory with article text into pmid name files, grouped into folders named after the class CUI
vectorizedTest_dir = final_dir + '/' + test_folder_name + '_vectorized'; # Final directory with analyzed data in vector representation (e.g. as TFIDF features)
# Directories of corresponding training datasets to read Vocabulary and IDF for data transformation
trainDatasets ={
    'NoFS_0': train_dir_vect
}

evalDatasets ={
    'NoFS_0': final_dir +  '/'+test_folder_name+'_vectorized_NoFS_0',
    }
#   ***     Dynamically produce dataset paths   ***
for scoreFunction in scoreFunctions.keys():
    for featureK in featureKs:
        datasetName = scoreFunction + "_" + str(featureK)
        trainDatasets[datasetName] = trainDatasets['NoFS_0'] + "_FSby_" + scoreFunction + "_" + str(featureK)
        evalDatasets[datasetName] = final_dir + '/' + test_folder_name + "_vectorized_" + scoreFunction + "_" + str(featureK)

# Make Pandas print all columns of a DataFrame when less or equal to 20
pd.set_option('display.max_columns', 20)

'''     
    STEP 1
    
    ******************************************************************
    ***     Create and Split Distantly Supervised (DS) Datasets    ***
    ******************************************************************

    This script is intended to create and split Distantly Supervised datasets of articles with concept level topic annotations
        Initial steps before running this script to create a data set:
            1) [Run Java "SaveInMongo" and export with MongoChef] Create a CSV file (_Cts_inT.csv) with articles per class based on Ct occurence in articles for t. 
                This can be done querying the Neo4j Open Data Graph with the following query : MATCH (article:Article)-->(concept:Entity{id:"C0917713"})-[r]->(article)  WHERE ANY(rs in r.sent_id WHERE left(rs, length(article.id+"_abstract_")) = article.id+"_abstract_") RETURN concept.id, concept.label,article.id,article.title
            2) [Java "EntrezHarvester" and export with MongoChef] Create a CSV file with articles for t with harvested fields e.g. abstract text
            3)  Create (manually) a CSV file with CUI-abbreviation correspondence (i.e. the class/label names) (e.g. DMD_labelNames.csv)

    This Script combines for each article the classes read from the first CSV and the data (e.g. abstract and title) from the second CSV, creating a Dataset.
        A text file is created for each article named after its PMID and locaded in a folder named after the class CUI
        After loading the data, annotations are converted into Multi-Labeled (ML) to be adequate for training ML Classifiers
        Manually annotated articles are removed from dataset to be used for testing and saved into a separate folder for further processing. In particular: 
            1)  A MA1 dataset is seleceted randomlly and removed from the dataset
            2)  A MA2 dataset is selected balancing the labelsets, excluding the preferred label, and removed from the dataset. To not exclude the preferred label, put a non existing CUI in the corresponfing setting (i.e. preferredConceptCUI).

        The remaining articles are split into Trainig and Tesing datasets retaining the Label-Set Distribution if a proportion different from 0.0 is provided in the settings.

'''
if datasetCreate :
    print(datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'),"-->> Start Trainig data creation")
    createDataset(labelNames_csv,class_csv,output_dir, manual_dataset_dir,manual_dataset_1_dir,manual_dataset_2_dir,
                  manual_dataset_pmids,fields_csv,rest_dir,manual_dataset_1_size,manual_dataset_1_dir_pmids,manual_dataset_2_size,cTop,
                  manual_dataset_2_dir_pmids,test_ratio,train_dir,test_dir,manual_dataset1_pmids,manual_dataset2_pmids)
    print(datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'),"-->> End Trainig data creation")
else :
    print(datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'),"-->> Trainig data creation skipped")

# Leave the "cTop" folder in training folder for model creation, these articles should be used as negative for other classes.
# TODO: Handle case where some labels are not found at all! Currently these labels are manually removed from the original MA datasets

'''
    STEP 2
         
    ********************************************
    ***     Transform a Training dataset     ***
    ********************************************

    Make target a binary 2d matrix with one column per class
    Tokenize the text of each sample and calculate TFIDF features for the tokens
    Also use concept occurrence features (from occurrence_csv) 
    Save the Transformed dataset in a new direcotry
'''
if Transform:
    print(datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'),"-->> Start Trainig data tranformation")
    tranformTrainingDataset(train_dir, train_dir_vect,useCUIS, occurrence_csv,feature_prefix)
    print(datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'),"-->> End Trainig data tranformation")
else :
    print(datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'),"-->> Trainig data tranformation skipped")

'''
    STEP 3
    
    ***********************************
    ***     Analyze the Dataset     ***
    ***********************************
    
    Perform Feature Selection and create new training datasets with selected Features only
'''
if Analyze:
    print(datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'),"-->> Start Trainig data analysis")
    analyzeTrainigDataset(train_dir_vect,featureKs,scoreFunctions)
    print(datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'),"-->> End Trainig data analysis")
else:
    print(datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'),"-->> Trainig data analysis skipped")

'''    
    STEP 4
     
    *********************************************************
    ***     Create a Manually Annotated (MA) Dataset      ***
    *********************************************************

    Read the text of articles included in the MA Dataset from corresponding folder (created during WS Dataset creation)
    Read the MA classes/labels from a CSV file created manually
    Write the MA Dataset in text files, in format adequate to be read by  
'''

if readMA :
    print(datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'),"-->> Start Test data creation from Manual Annotations ")
    creatTestDataByMA(train_dir_vect, wsTest_dir, evalLabels_csv, labeledTest_dir)
    print(datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'),"-->> End Test data creation from Manual Annotations")
else :
    print(datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'),"-->> Test data creation from Manual Annotations skipped ")

'''     
    STEP 5 
    
    ************************************
    ***     Transform MA dataset     ***
    ************************************
    Make target a binary 2d matrix with one column per class
    Tokenize the text of each sample and calculate TFIDF features for the tokens
    Save the Transformed dataset in a new direcotry
'''
if TransformMA :
    print(datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'),"-->> Start Test data transformation based on the trainig dataset")
    tranformTestData( trainDatasets, labeledTest_dir, occurrence_csv, feature_prefix, vectorizedTest_dir)
    print(datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'),"-->> End Test data transformation based on the trainig dataset")
else:
    print(datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'),"-->> Test data transformation Skipped ")

'''     
    STEP 6
    
    ***********************************
    ***     Train a Classifier      ***
    ***********************************
'''

if classify:
    print(datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'),"-->> Start Classifier creation and evaluation")
    createClassifiers( trainDatasets, evalDatasets, final_dir, test_folder_name, wsTest_dir, dominantLabel, cTop)
    print(datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'),"-->> End Classifier creation and evaluation")
else:
    print(datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'),"-->> Classifier creation and evaluation skipped ")