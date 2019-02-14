import pandas as pd
import pickle
import scipy.sparse
from sklearn.feature_selection import SelectKBest, chi2, f_classif, mutual_info_classif
import sklearn.datasets
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.preprocessing import MultiLabelBinarizer
import random
import csv
from sklearn.model_selection import train_test_split
import sklearn.datasets
from scipy.sparse import *
import os
import codecs
from scipy import sparse
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from sklearn.multiclass import OneVsRestClassifier
import sklearn.datasets
from sklearn.metrics import *
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
import datetime
import time

'''
    *********************************************************
    ***     Various Functions for developing datasets     ***
    *********************************************************

    These functions are used by scripts to create, transform and use datasets to develop classifiers for MeSH topic annotations to UMLS comcept-level anotations.
'''

'''
#   ************************************************************
#   ***     Hardcoded settings for dataset read and write    ***
#   ************************************************************
'''
# Hardcoded file names for all files constituting the vectorized dataset
document_fn = 'documents.csv'           # A pandas.DataFrame with columns : 'pmid', 'text'
label_fn = 'labels.pickle'              # A numpy.ndarray with label annotations (e.g. [[0,1,0], [0,0,1], [1,1,0] ...]) {"pickled"/serialized for saving}
labelNames_fn = 'labelNames.csv'        # A pandas.DataFrame with columns : name, abbreviation
tfidf_fn = 'tfidf.npz'                  # A scipy.sparse.csr_matrix (document id, token id) with the TFIDF values for the dataset
count_fn = 'count.npz'                  # A scipy.sparse.csr_matrix (document id, token id) with the count (ie TF) values used as reference for TFIDF calculation. (Not necesarily calculated on documents of documents.csv)
tokenName_fn = 'tokenNames.txt'         # A pandas.DataFrame with columns : 'token'. Provides a mapping from token id to token string
# Hardcoded column name for feature selection (across classes)
MaxWeights = 'MaxWeights'
# Hardcoded temporary name for the directory/class/label of non annotated articles in a dataset when saving a subset of it
noClass = "noClass"
# Class name for articles without occurrences
rest_name = 'rest';
# System compatibility variables
# path_Separator = '\\' #for windows
path_Separator = '/' #for unix

'''     
        **********************************************
        ***     Read and write datasets functions  ***
        **********************************************
'''

def readTrainReference(train_dir):
    '''
    Read reference data from vectorized training dataset folder:
        1)  The vocabulary to be used for tokenization (token id -> token/term)
        2)  The Counts to be used for IDF calculation (a compressed scipy.sparse.csr_matrix)
    :param train_dir: The folder of the training dataset to read the reference date from. The datasets should have been written with corresponding functions saveDataset(FromBunch/FromDFs)
    :return: The reference data loaded
        1)  vocabulary  :   A pandas.DataFrame with columns : 'token'. Provides a mapping from token id to token string
        2)  trainCount  :   The vocabulary used for tokenization as a Dictionary token/term to token index/id (As expected/produced by Tokenizer)
    '''
    # Load trainCount
    trainCount_file = train_dir + "/" + count_fn
    trainCount = scipy.sparse.load_npz(trainCount_file)        # A scipy.sparse.csr_matrix (document id, token id)
    # Load the existing vocabulary
    vocabulary_csv = train_dir + "/" + tokenName_fn;
    vocabulary = readVocabulary(vocabulary_csv)
    # Read label/class names
    labelNames_csv = train_dir + "/" + labelNames_fn
    labelNames = pd.read_csv(labelNames_csv, ";",index_col=0)  # A pandas.DataFrame with columns : name, abbreviation
    classes = labelNames['name']
    return vocabulary, trainCount, classes

def saveLabelNames(output_dir,labelNames):
    '''
    Save the labelsNames DataFrame in an adequate csv file in the given directory
    :param output_dir: The folder where to write
    :param labelNames: A pandas.DataFrame with columns : name, abbreviation
    :return: None
    '''
    labelNames.to_csv(output_dir + '/' + labelNames_fn, ";")

def loadFromTextFiles(dir, categories = None):
    '''
    Load a dataset from text files distributed into corresponding folders.
    A wrapper for sklearn.datasets.load_files which also considers an accompanying DataFrame with label details
    :param dir: The directory with the text files of the articles distributed in sub-directories named after the classes/labels CUIs (i.e. C0013264, C0917713 etc)
    :return:
        1)  A Bunch object representing the datasets (as created by sklearn.datasets.load_files)
        2)  A DataFrame with label CUIs (name comuln) and abbreviations (abbreviation column) mapping
    '''
    # Read label/class names
    labelNames_csv = dir + '/' + labelNames_fn
    labelNames = pd.read_csv(labelNames_csv, ";",index_col=0)  # A pandas.DataFrame with columns : name, abbreviation
    print(labelNames)
    # print(dir)
    dataset = sklearn.datasets.load_files(dir, encoding="utf8")

    return dataset, labelNames

def readDataset(input_dir):
    '''
    Load a vectorized dataset from a folder given including all corresponding files
    In particular the files required are the ones described in "Hardcoded settings for dataset read and write" above
    :param input_dir: The folder path (as a string) where to read the dataset files from
    :return: All variables required to work with a dataset. In particular the following variables are returned:
        1)  documents:      A pandas.DataFrame with columns : pmid, text
        2)  labels:         A numpy.ndarray with label annotations (e.g. [[0,1,0], [0,0,1], [1,1,0] ...]) {"pickled"/serialized for saving}
        3)  labelNames:     A pandas.DataFrame with columns : name, abbreviation
        4)  tfidf:          A scipy.sparse.csr_matrix (document id, token id) wth TFIDFs
        5)  count:          A scipy.sparse.csr_matrix (document id, token id) with TFs
        6)  tokens:         A pandas.DataFrame with columns : token
    '''
    # Hardcoded file names for all files constituting the vectorized dataset
    documentFile = makeFile(input_dir) + "/" + document_fn
    labelFile = makeFile(input_dir) + '/' + label_fn
    labelNamesFile = makeFile(input_dir) + "/" + labelNames_fn
    tfidfFile = makeFile(input_dir) + "/" + tfidf_fn
    countFile = makeFile(input_dir) + "/" + count_fn
    tokenNamesFile = makeFile(input_dir) + "/" + tokenName_fn

    # The pmid and text for each document
    documents = pd.read_csv(documentFile, ";",index_col=0)      # A pandas.DataFrame with columns : pmid, text
    # print(documents)

    # The label ids (e.g. [0,1,0]) for each document
    with open(labelFile, "rb") as lf:
          labels = pickle.load(lf)                  # A "pickled" (serialized) python object with label annotations (e.g. [[0,1,0], [0,0,1], [1,1,0] ...])
    # print(labels)

    # The labelNames for each label id
    labelNames = pd.read_csv(labelNamesFile, ";",index_col=0)   # A pandas.DataFrame with columns : name, abbreviation
    # print(documents)

    # The TFIDF features for each token-document combination as a
    tfidf = scipy.sparse.load_npz(tfidfFile)        # A scipy.sparse.csr_matrix (document id, token id)
    # print(tfidf)

    # The TF features for each token-document combination as a
    count = scipy.sparse.load_npz(countFile)        # A scipy.sparse.csr_matrix (document id, token id)
    # print(tfidf)

    # The string value of each token
    tokens = pd.read_csv(tokenNamesFile, ";",index_col=0)       # A pandas.DataFrame with columns : token
    # print(tokens)

    return documents, labels, labelNames, tfidf, count, tokens

def saveDatasetFromDFs(outut_dir, documents, target, labelNames, tfidf, count, tokenFeatureNames):
    '''
    Write all required data to reconstruct the vectorized dataset into corresponding files in the given folder path
    In particular the files created are the ones described in "Hardcoded settings for dataset read and write" above
    :param outut_dir: The folder path (as a string) where to write the dataset files
    :param documents:   A pandas.DataFrame with columns : pmid, text
    :param target:      A numpy.ndarray with label annotations (e.g. [[0,1,0], [0,0,1], [1,1,0] ...]) {"pickled"/serialized for saving}
    :param labelNames:  A pandas.DataFrame with columns : name, abbreviation
    :param tfidf:       A scipy.sparse.csr_matrix (document id, token id) with TFIDF values.
    :param count:       A scipy.sparse.csr_matrix (size: num of documents, num of tokens) with TF value for each token-document combination
    :param tokenFeatureNames: A pandas.DataFrame with columns : token
    :return: None
    '''
    # Hardcoded file names for all files constituting the vectorized dataset
    documentFile = makeFile(outut_dir) + "/" + document_fn
    labelFile = outut_dir + '/' + label_fn
    labelNamesFile = outut_dir + "/" + labelNames_fn
    tfidfFile = outut_dir + "/" + tfidf_fn
    countFile = outut_dir + "/" + count_fn
    tokenNamesFile = outut_dir + "/" + tokenName_fn

    # Save TFIDF features as a scipy.sparse.csr_matrix
    scipy.sparse.save_npz(tfidfFile, tfidf, compressed=True)  # A scipy.sparse.csr_matrix (document id, token id)

    # save arrays
    scipy.sparse.save_npz(countFile, count, compressed=True)  # A scipy.sparse.csr_matrix (document id, token id) with counts

    # Save pmid, text and labels per document
    documents.to_csv(documentFile, ";")  # A pandas.DataFrame with columns : pmid, text

    # Save target
    with open(labelFile, "wb") as lf:
        pickle.dump(target, lf)  # A "pickled" (serialized) python object with label annotations (e.g. [[0,1,0], [0,0,1], [1,1,0] ...])

    # # Print an example element of documents datafarme
    # index = 10;
    # print("\nAn example, the", index, "th element :")
    # print(documents.iloc[index])

    # Save label names
    labelNames.to_csv(labelNamesFile, ";")

    # Save token names per token id
    tokenFeatureNames.to_csv(tokenNamesFile, ";")

def saveDatasetFromBunch(outut_dir, training_dataset, tfidf, count, feature_names, labelNames):
    '''
    Write all required data to reconstruct the vectorized dataset into corresponding files in the given folder path
    In particular the files created are the ones described in "Hardcoded settings for dataset read and write" above
    :param outut_dir: The folder path (as a string) where to write the dataset files
    :param training_dataset: A Bunch object of an article dataset as loaded from folder with sklearn.datasets.load_files
    :param tfidf: A scipy.sparse.csr_matrix (size: num of documents, num of tokens) with TFIDF value for each token-document combination
    :param count: A scipy.sparse.csr_matrix (size: num of documents, num of tokens) with TF value for each token-document combination
    :param feature_names: A list (size: num of tokens) with each "token string" in the corresponding index. (So that the index of a token in feature_names corresponds to the id used in train_counts for this token)
    :param labelNames: A pandas.DataFrame with columns : name, abbreviation
    :return: None
    '''
    # Hardcoded file names for all files constituting the vectorized dataset
    documentFile = makeFile(outut_dir) + "/" + document_fn
    labelFile = makeFile(outut_dir) + '/' + label_fn
    labelNamesFile = makeFile(outut_dir) + "/" + labelNames_fn
    tfidfFile = makeFile(outut_dir) + "/" + tfidf_fn
    countFile = makeFile(outut_dir) + "/" + count_fn
    tokenNamesFile = makeFile(outut_dir) + "/" + tokenName_fn

    # Save TFIDF features as a scipy.sparse.csr_matrix
    scipy.sparse.save_npz(tfidfFile, tfidf, compressed=True)  # A scipy.sparse.csr_matrix (document id, token id) with tfidf values

    # save arrays
    scipy.sparse.save_npz(countFile, count, compressed=True)  # A scipy.sparse.csr_matrix (document id, token id) with counts

    # Save pmid, text and labels per document
    documents = pd.DataFrame({"pmid": training_dataset.filenames, "text": training_dataset.data})
    documents.to_csv(documentFile, ";")  # A pandas.DataFrame with columns : pmid, text

    # Save target
    with open(labelFile, "wb") as lf:
        pickle.dump(training_dataset.target,
                    lf)  # A "pickled" (serialized) python object with label annotations (e.g. [[0], [1,2], [0,3] ...])

    # # Print an example element of documents datafarme
    # index = 10;
    # print("\nAn example, the", index, "th element :")
    # print(documents.iloc[index])

    # Save label names
    # Reorder labels to match the keys
    orderedLabelNames = pd.DataFrame({"name":[],"abbreviation":[]})
    for cui in training_dataset.target_names:
        index = training_dataset.target_names.index(cui)
        if cui in labelNames.index: # Normal CUIs should be in the labelNames CVS file
            abbreviation = labelNames.loc[cui, 'abbreviation']
        else : # The "CUI" "rest" is not in the labelNames CVS file
            abbreviation = cui
        orderedLabelNames.loc[index] = [ cui, abbreviation]
    orderedLabelNames.to_csv(labelNamesFile, ";")# A pandas.DataFrame with columns : name, abbreviation

    # Save token names per token id
    tokenFeatureNames = pd.DataFrame({"token": feature_names})  # A pandas.DataFrame with columns : token
    tokenFeatureNames.to_csv(tokenNamesFile, ";")

def readVocabulary(vocabulary_csv):
    '''
    Read the vocabulary from a CSV file (in :param vocabulary_csv:) containing token index in first column (Unnamed on index 0) and the token on the next (named "token")
    :param vocabulary_csv:  A CSV file containing token index in first column (Unnamed on index 0) and the token on the next (named "token")
    :return: The vocabulary used for tokenization as a Dictionary token/term to token index/id (As expected/produced by Tokenizer)
    '''
    # Load the (inverted) vocabulary
    vocabulary = pd.read_csv(vocabulary_csv, ";", index_col=0, keep_default_na=False, na_values=[''])  # A pandas.DataFrame with column : token index (Unnamed on index 0) and "token"
    # Convert vocabulary from DataFrame to a Dictionary
    vocabulary = vocabulary.to_dict()["token"]
    # Invert it to be adequate for use by Tokenizer
    inv_vocabulary = {v: k for k, v in vocabulary.items()}
    return inv_vocabulary

def saveDatasetTextFiles(folderToWrite, label_keys,labelNames, X, y, names, ignoreRest = None):
    '''
    Write articles into given folder, as text files organized in one folder per label.
        Articles with multiple labels saved multiple times (in multiple folders). This duplications should be handled when loading the dataset (converting the dataset to Multi-Label)
    :param folderToWrite: The folder where the articles will be written (e.g. "D:/05 MeSHToUMLS/DSTDS/Final/DMD/train/")
    :param label_keys: The names of the labels as a list (including "rest"). The indexes in this list should be the same used in :param y (e.g. 0 for C0013264, 1 for C0917713, 2 for C3542021, 3 for rest)
    :param labelNames: A pandas.DataFrame with columns : name, abbreviation. It provides the correspondence of CUIs to abbreviations or other label information.
    :param X: The data, i.e. the text, of the articles. As a list of string elements following the same indexing (i.e. beeing parallel) to :param y and :param names
    :param y: The labels of the articles as a list of lists (e.g. [[0], [0, 1], [2] ... ] following the same indexing (i.e. beeing parallel) to :param X and :param names.
                The indexes in the inner lists should be the same with the ones used in :param label_keys
    :param names: The pmids of the articles as a list of strings following the same indexing (i.e. beeing parallel) to :param X and :param y.
    :return: Nothing
    '''

    # Save label names
    labelNamesFile = makeFile(folderToWrite) + labelNames_fn
    labelNames.to_csv(labelNamesFile, ";")

    # Create adequate folders if not exist
    for key in label_keys:
        if not os.path.exists(folderToWrite + key):
            write = True
            if key == rest_name and ignoreRest:
                write = False
            if write:
                os.makedirs(folderToWrite + key)

    # Write the articles in files
    print("writing ", len(X), " files")
    for index in range(0, len(X)):
        for keyIndex in y[index]:
            write = True
            if label_keys[keyIndex] == rest_name and ignoreRest:
                write = False
            if write:
                file = codecs.open(folderToWrite + label_keys[keyIndex] + '/' + names[index] + ".txt", "w+", "utf-8")
                file.write(X[index])
                file.close()

def makeFile(folderToWrite):
    '''
    Create a file if not exists
    :param folderToWrite: String path to the folder to be crated
    :return: folderToWrite the path to the folder created (the same as the input)
    '''
    import os
    if not os.path.exists(folderToWrite):
        os.makedirs(folderToWrite)
    return folderToWrite

def saveDatasetTextFilesSubset(pmidListFile, folderToWrite, label_keys, labelNames, X, y, names):
    '''
    Saves the articles included in the list of :param pmidListFile, as text files, in :param folderToWrite:
        Also save the labels/predictions for those articles according to the DataSet for dataset evaluation
        TODO: remove noClass. What was the intended use of it?
        Note: all folders (NoClass and DS label folders) are saved together. Distinction of the datasets should be done manually before further use.
    :param pmidListFile: A file with pmids of articles (subset) to be saved from the dataset. The file should contain one pmid per line.
    :param folderToWrite: The folder where the articles will be written (e.g. "D:/05 MeSHToUMLS/DSTDS/Final/DMD/validation/")
    :param label_keys: The names of the labels as a list (including "rest"). The indexes in this list should be the same used in :param y (e.g. 0 for C0013264, 1 for C0917713, 2 for C3542021, 3 for rest)
    :param labelNames:  A pandas.DataFrame with columns : name, abbreviation
    :param X: The data, i.e. the text, of the articles. As a list of string elements following the same indexing (i.e. beeing parallel) to :param y and :param names
    :param y: The labels of the articles as a list of lists (e.g. [[0], [0, 1], [2] ... ] following the same indexing (i.e. beeing parallel) to :param X and :param names.
    :param names: The pmids of the articles as a list of strings following the same indexing (i.e. being parallel) to :param X and :param y.
    :return: Nothing
    '''

    # Create adequate folders if not exist
    for key in label_keys:
        if not os.path.exists(folderToWrite + key):
            os.makedirs(folderToWrite + key)

    # Save label names
    labelNamesFile = folderToWrite + "/" + labelNames_fn
    labelNames.to_csv(labelNamesFile, ";")

    # All articles will be added into one class/label, that of noClass
    # folderToWriteNoClass = makeFile(folderToWrite + '/' + noClass)

    # Read ids of articles from corresponding files
    ValidationDataset = open(pmidListFile, "r")
    pmids = ValidationDataset.readlines()

    # Iterate through all pmids of validation dataset
    print("writing ", len(pmids), " files")
    for pmid in pmids:
        # clean the pmid from extra characters
        pmid = pmid.replace("\n", "")
        pmid = pmid.replace("\'", "")
        # if this article is also included in the training/test data
        if pmid in names:
            # Find corresponding text and write it in one folder
            index = names.index(pmid)
            # file = codecs.open(folderToWriteNoClass + '/' + names[index] + ".txt", "w+", "utf-8")
            # file.write(X[index])
            # file.close()
            # add corresponding prediction in y_new
            for keyIndex in y[index]:
                file = codecs.open((folderToWrite + label_keys[keyIndex]) + '/' + names[index] + ".txt", "w+", "utf-8")
                file.write(X[index])
                file.close()

    ValidationDataset.close()

'''     
        *************************************
        ***     Data splitting functions  ***
        *************************************
'''

def selectRandomSubset(numOfArticles, names):
    '''
    Select a random subset of :param numOfArticles: articles from the dataset
    :param numOfArticles: The number of articles to select
    :param names: The pmids of the articles as a list of strings following the same indexing (i.e. being parallel) to :param X and :param y.
    :return: selectedPmids : List of pmids selected for the subset
    '''
    validnames = list(names)
    selectedPmids = []

    # ***   Select pmids for the subset to be created ***
    while (len(selectedPmids) < numOfArticles) and len(validnames) > 0 :
        pmid = validnames[random.randint(0, len(validnames) - 1)]
        selectedPmids.append(pmid)
        validnames.remove(pmid)
    return selectedPmids

def selectBalancedSubset(numOfArticles, y, names, proportion, preferredLabelIndex):
    '''
    Select a dataset of :param numOfArticles: articles from the datasets balancing the labelsets appearing
        In particular select articles from each labelset until:     1) Enough articles have been selected
                                                                OR  2) Less than a given proportion of articles have remain available in a labelset
        *   For labelest initially having just one article, randomly select if should be included in the remainig set or the subset
        **  The preffered label is ignored as labelset (e.g. 'C0002395' for the AD). However combinations with other labels is stil calculated.
                If This is not desired use a non valid index, e.g. -1 which will not me removed as the corresponding code will not be found.
    :param numOfArticles: The number of articles to select
    :param y: The labels of the articles as a list of lists (e.g. [[0], [0, 1], [2] ... ] following the same indexing (i.e. beeing parallel) to :param X and :param names.
    :param names: The pmids of the articles as a list of strings following the same indexing (i.e. being parallel) to :param X and :param y.
    :param proportion: The proportion of articles that should remain in the datasets for each labelset (when reached, stop selecting articles from this labelset).
    :param preferredLabelIndex: The index of the preferred label to be ignored as a labelset (only when alone - combinations with other labels will be still calculated). This index corresponds to the order in dataset.target variable.
    :return: selectedPmids : List of pmids selected for the subset
    '''

    selectedPmids = []
    # ***   Create labelsets and corresponding lists of pmids   ***

    # a hash table of "labelset code" (e.g. "[1, 2]") to pmid list
    labelSetArticles = {}
    for i in range(0,len(y)): # For all articles in y and names (which are parallel)
        # print(y[i], " - ", names[i])
        # create the "labelset code" using the string representation of the sorted list of lables - Sort is necessary so that the order of labels do not create equivalent different codes for the same labelset.
        labelSet = y[i]
        labelSet.sort()
        labelSetCode = str(labelSet)
        if labelSetCode in labelSetArticles.keys():
            labelSetArticles[labelSetCode].append(names[i])
        else:
            labelSetArticles[labelSetCode] = [names[i]]

    # ***   Calculate labelset size ***

    # a hash table of "labelset code" (e.g. "[1, 2]") to pmid list initial length
    labelSetSize = {}
    for labelSet in labelSetArticles.keys():
        labelSetSize[labelSet]=len(labelSetArticles[labelSet])
    print(labelSetSize)

    # ***   Labelsets to select pmids from ***
    validLabelSets = list(labelSetArticles.keys())
    defaultLabelCode = str([preferredLabelIndex])
    print( " Default label code to be ingnored: " + defaultLabelCode)
    if defaultLabelCode in validLabelSets:
        print(" Default label ingnored!" )
        validLabelSets.remove(defaultLabelCode)

    # ***   Select pmids for the subset to be created ***
    currentLabelSetIndex = 0
    while (len(selectedPmids) < numOfArticles) and len(validLabelSets) > 0 :
        # Get the code of the next labelset in list of validLabelSets
        currentLabelSet = validLabelSets[currentLabelSetIndex]
        availableArticles = len(labelSetArticles[currentLabelSet])
        # The proportion of initial labelset articles remaining available if we remove an article
        availableArticleProportion = (availableArticles -1) /labelSetSize[currentLabelSet]
        # If there is only one article in the labelset, by chance decide to include or exclude it from the subset
        if labelSetSize[currentLabelSet] == 1:
            pmid = labelSetArticles[currentLabelSet][0]
            if random.random() > 0.5:
                selectedPmids.append(pmid)
                labelSetArticles[currentLabelSet].remove(pmid)
            # In any case remove the labelset from valid labelsets. We should not visit it again if the chance was to not include the article in the subset.
            validLabelSets.remove(currentLabelSet)
        elif availableArticleProportion > proportion:
            # get a random article
            pmid = labelSetArticles[currentLabelSet][random.randint(0, availableArticles - 1)]
            selectedPmids.append(pmid)
            labelSetArticles[currentLabelSet].remove(pmid)
        else: # This labelset have less that the given proportion of articles available, remove it to not be checked again
            validLabelSets.remove(currentLabelSet)

        # Update the index of the current labelset
        if currentLabelSet in validLabelSets: # If the current label has not been removed (which makes it's index pointing the next labelset whithout changing it) increase the index
            currentLabelSetIndex += 1 # Go to the next labelSetCode
        if currentLabelSetIndex >= len(validLabelSets) : # If you pass last labelSetCode
            currentLabelSetIndex = 0 # reset to the first

    return selectedPmids

def removeArticlesFromDataset(pmidListFile, X, y, names):
    '''
    Removes articles included in the list of :param pmidListFile from the dataset defined by :param X, :param y and :param names
    :param pmidListFile: A file with pmids of articles to be excluded from the dataset. The file should contain one pmid per line.
    :param X: The data, i.e. the text, of the articles. As a list of string elements following the same indexing (i.e. being parallel) to :param y and :param names
    :param y: The labels of the articles as a list of lists (e.g. [[0], [0, 1], [2] ... ] following the same indexing (i.e. being parallel) to :param X and :param names.
    :param names: The pmids of the articles as a list of strings following the same indexing (i.e. being parallel) to :param X and :param y.
    :return: Nothing
    '''
    # Read ids of articles from corresponding files
    ValidationDataset = open(pmidListFile, "r")
    pmids = ValidationDataset.readlines()

    # Itterate through all pmids of validation dataset
    count = 0
    for pmid in pmids:
        # clean the pmid from extra characters
        pmid = pmid.replace("\n", "")
        pmid = pmid.replace("\'", "")
        # if this article is also included in the training/test data remove it
        if pmid in names:
            index = names.index(pmid)
            #         print(index)
            count += 1
            #         print (" ", count)
            # print("deleting : ",names[index],y[index])
            del X[index]
            del names[index]
            del y[index]
    ValidationDataset.close()

'''     
        *****************************************
        ***     Data transormation functions  ***
        *****************************************
'''

def getConceptToArticleOccurrenceMap(occurrenceCSV, dataset):
    '''
    Create a dictionary from CUI to corresponding list of pmidIndexes for articles of the dataset where it has been recognized
    ATTENTION: Only presence of concept is currently taken into account, no the number of occurrences
    :param occurrenceCSV:   The primary CSV file to read occurrence from: In this version contains two columns, one with the cui, and one with the pmids where it occurs.
    :param dataset:         A Bunch object representing the dataset (as created by sklearn.datasets.load_files)
    :return:        A dictionary from CUI to a list of pmidIndexes (in the corresponding dataset) where the corresponding concept occurs (e.g. "C0013264" -> ["2345433","34432221",...])
    '''
    # Read concept occurence from CSV
    concept_occurrences = pd.read_csv(occurrenceCSV, ";", index_col=0)
    # print(type(train_counts))
    # print(concepts)

    # Create a hash table CUI to corresponding PMID indexes of articles where it occurs
    conceptToArticle = {}  # cui to pmidIndex
    for pmid in dataset.filenames:
        if int(pmid) in concept_occurrences.index:
            cuiListString = concept_occurrences.loc[int(pmid)]['cuis'];  # e.g. "[ "C2964138", "C0441833", "C0444706", ... ]'
            cuiListString = cuiListString[1: -1].replace('"', '').replace(' ', '')  # i.e. 'C2964138,C0441833,C0444706,...'
            cuis = cuiListString.split(',')  # i.e. ['C2964138','C0441833','C0444706','...'] as a python list
            pmidIndex = dataset.filenames.index(pmid)
            # print(pmid, train_counts[pmidIndex, 3])
            for cui in cuis:
                if not cui in conceptToArticle.keys():
                    conceptToArticle[cui] = []
                conceptToArticle[cui].append(pmidIndex)
        else:
            print("Warining: pmid " + pmid + " has no CUI occurences at all!")
    return conceptToArticle

def getTFIDF(sample_counts, corpus_counts=None):
    '''
    Calculate the TFIDF value for each token in each document (in :param sample_counts:) based on token counts.
        The optinal :param corpus_counts: is only used for normalization (i.e. calculation of IDF) for TFIDF calculation.
        When called with one parameter, both TF and IF are calculated in the same set of documents.
        ATTENTION: Both csr_matrices should use the same vocabulary (i.e. the same ids for the features/tokens). In particular, it makes sense that both use a vocabulary based on :param sample_counts: articles.
    :param sample_counts:  Counts of the articles (e.g. test data) to calculate the TF. It is a scipy.sparse.csr_matrix (size: num of documents, num of token ids) with token count for each token-document combination
    :param corpus_counts:  (Optinal) Counts of the corpus (e.g. training data) to calculate the IDF. It is a scipy.sparse.csr_matrix (size: num of documents, num of token ids) with token count for each token-document combination
    :return:        A scipy.sparse.csr_matrix (size: num of documents in :param sample_counts:, num of tokens in :param corpus_counts:) with TFIDF value for each token-document combination, for all documents in :param sample_counts:
    '''

    # If no separate corpus_counts is provided, use sample_counts as the coprus for IDF calculation
    if corpus_counts is None:
        corpus_counts = sample_counts

    tfidf_transformer = TfidfTransformer()
    # Fit to corpus data
    tfidf_transformer.fit(corpus_counts)
    # Transform samples to a tf-idf matrix.
    train_tfidf = tfidf_transformer.transform(sample_counts)
    # Print sample No (articles) and feature No (tokens)
    print(train_tfidf.shape)
    return train_tfidf

def tokenizeArticleText(X, vocabulary = None):
    '''
    Tokenize text data using a predefined vocabulary (:param vocabulary:) and produce a sparse representation of each token count in each document.
    If optional :param vocabulary: is not present then fit/create a vocabulary with all tokens present in the dataset
    Represent token counts as a scipy.sparse.csr_matrix with dimensions (number of documents, number of tokens)
    Based on this example : http://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html
    :param X: An Array-like structure with text data to be tokenized. (i.e. one document per index)
    :param vocabulary: (Optional) The vocabulary used for tokenization as a Dictionary token/term to token index/id (As expected/produced by Tokenizer)
    :return: counts, feature_names
        1)  counts :  A scipy.sparse.csr_matrix (size: num of documents, num of token ids) with token count for each token-document combination
        2)  feature_names : A list (size: num of tokens) with each "token string" in the corresponding index. (So that the index of a token in feature_names corresponds to the id used in counts for this token)
    '''

    if not vocabulary is None:
        count_vect = CountVectorizer(vocabulary=vocabulary)
    else:
        count_vect = CountVectorizer()

    # Fit and return term-document matrix (transform).
    counts = count_vect.fit_transform(X)
    # Print sample No (articles) and feature No (tokens)
    # print(counts.shape)
    # print (sample no: article, feature no: word id)  number of occurences
    # print(counts)

    # get inverted vocabulary
    inv_vocabulary = {v: k for k, v in count_vect.vocabulary_.items()}
    # print(inv_vocabulary)

    # Create a list with the names of the features
    feature_names = []
    for i in range(len(count_vect.vocabulary_)):
        feature_names.insert(i, inv_vocabulary[i])
    return counts, feature_names

def makeDatasetMultiLabel(initial_dataset):
    '''
    Convert the dataset of articles into a Multi Label dataset updating the three basic list-variables of :param initial_dataset
        The three basic list-variables initial_dataset.data, initial_dataset.filenames, initial_dataset.target that are updated as described below:
            1) Removing "duplicate" articles from all list-variables (included in more than one label-folders)
            2) Changing the representation of initial_dataset.target from a simple list e.g. [1, 0, 2, ...] to a list of lists e.g. [[1], [0], [0, 2], ...]
            3) Changing the content of initial_dataset.filenames from "filenames" by just "pmids" based on "pmid.txt" naming convention. E.g. converts "C://.../1213424.txt" to "1213424"
    :param initial_dataset: A Bunch object of an article dataset as loaded from folder with sklearn.datasets.load_files
    :return: Nothing
    '''

    # Iterate the whole dataset
    ml_filenames = []
    ml_data = []
    # alternative representation of multiple classes, used to call MultiLabelBinarizer
    ml_targets_alt = []

    # for all articles in the dataset, including "duplicate" articles appearing in more than one topic folders
    for data, filename, target in zip(initial_dataset.data, initial_dataset.filenames, initial_dataset.target):
        # get the pmid of the article
        words = filename.split(path_Separator)
        pmid = words[-1].replace(".txt","")
        # print(pmid, " ", target)

        # add the article in ml_filenames list
        if ml_filenames.count(pmid) == 0:
            ml_filenames.append(pmid)
            ml_targets_alt.append([])
            # add corresponding data in ml_data
            ml_data.insert(ml_filenames.index(pmid), data)
        # Updated corresponding labels in ml_targets
        #     ml_targets[ml_filenames.index(pmid)][target] = 1
        ml_targets_alt[ml_filenames.index(pmid)].append(target)
        # print(ml_targets_alt[ml_filenames.index(pmid)])

    # update DMD_train dataset
    initial_dataset.data = ml_data
    initial_dataset.filenames = ml_filenames
    initial_dataset.target = ml_targets_alt
    # print(ml_targets_alt)

'''     
        *******************************************
        ***     Feature selection functions     ***
        *******************************************
'''

def getTopFeatures(tfidf, count, Y, tokens, labelNames_list, scoreFunction, kFeatures):
    '''
    Transform the dataset (i.e. :param tfidf:, :param count: and :param tokens: variables) to include only the top :param kFeatures: features
        Performs Feature selection using the scores calculated by the :param scoreFunction:
    :param tfidf:           A scipy.sparse.csr_matrix (document id, token id) with TFIDF values
    :param count:           A scipy.sparse.csr_matrix (document id, token id) with token counts
    :param Y:               A numpy.ndarray with label annotations (e.g. [[0,1,0], [0,0,1], [1,1,0] ...]) {"pickled"/serialized for saving}
    :param tokens:          A pandas.DataFrame with columns : 'token'. Provides a mapping from token id to token string
    :param labelNames_list: A pandas.DataFrame with columns : name, abbreviation
    :param scoreFunction:   The function to be used for scoring during feature selection [For classification: chi2, f_classif, mutual_info_classif]
    :param kFeatures:       The number of top features to be kept in the dataset
    :return:    The transformed dataset considering only the top :param kFeatures: features selected by feature selection
        1) tfidf_selectedFeatures   A scipy.sparse.csr_matrix (document id, token id) considering only the :param kFeatures: features selected
        2) count_selectedFeatures   A scipy.sparse.csr_matrix (document id, token id) considering only the :param kFeatures: features selected
        3) tokens_selectedFeatures  A pandas.DataFrame with columns : 'token'. Provides a mapping from (new) token id to token string only for the :param kFeatures: features selected
    '''
    #TODO: split weigth calculation and k Top featire selection so that we do not reweight them to select with orther value for k.
    X = tfidf.toarray()
    X_df = getFeatureWeights(X, Y, tokens, labelNames_list, scoreFunction)
    # Print basic statistics
    print(X_df.describe())

    # Remove un-selected features from the dataset (i.e. tokens/rows from 'tfidf' and 'tokens' variables)
    # Sort descending by Max Weight across all labels
    X_df = X_df.sort_values([MaxWeights], ascending=0)

    # Print top 50 labels with corresponding weights in a box plot
    # X_df.iloc[0:50, :].plot.bar(x='token',y=labelNames_list);
    # plt.show(block=True)

    # Select top features/token ids only
    # Create a mask list with True for the selected features (i.e. top n)
    featureSelectionMask = [True] * kFeatures + [False] * (len(tokens) - kFeatures)
    # Add the mask column in the DataFrame
    X_df['FeatureMask'] = featureSelectionMask
    # Get the ids of the selected features
    featuresSelected = X_df.index[featureSelectionMask].tolist()
    # print(featuresSelected)

    # Transofm tfidf matrix (keep selected features only)
    tfidf_selectedFeatures = sparse.lil_matrix(sparse.csr_matrix(tfidf)[:, featuresSelected])
    tfidf_selectedFeatures = scipy.sparse.csr_matrix(tfidf_selectedFeatures)
    # print(tfidf_selectedFeatures.shape)

    # Transofm count matrix (keep selected features only)
    count_selectedFeatures = sparse.lil_matrix(sparse.csr_matrix(count)[:, featuresSelected])
    count_selectedFeatures = scipy.sparse.csr_matrix(count_selectedFeatures)
    # print(count_selectedFeatures.shape)

    # Tranform tokens DataFrame (keep selected features only)
    tokens_selectedFeatures = tokens.iloc[featuresSelected]
    # Selected Token/term strings (in the order appearing in tfidf and count matrices)
    new_tokens = list(tokens_selectedFeatures['token'])
    # Create new index
    new_index = range(len(featuresSelected))
    tokens_selectedFeatures = pd.DataFrame(new_tokens, index=new_index, columns={'token'})

    return tfidf_selectedFeatures, count_selectedFeatures, tokens_selectedFeatures

def getFeatureWeights(X,Y,tokens,labelsNames,scoreFunction):
    '''
    Calculate weights for the features of a dataset performing univariate feature selection based on :param scoreFunction:
        Weights are calculated per label and the Maximum weight (across labels) is selected to create an additional column
    :param X: ndarray (documents, tokens) with tfidf for each document-token combination
    :param Y: ndarray (documents, labels) with 1/0 for each document-label combination
    :param tokens: A pandas.DataFrame with columns : 'token'. Provides a mapping from token id to token string
    :param labelsNames: The names of the labels as a list of strings in the specified order used in :param Y: columns
    :param scoreFunction: The function to be used for scoring during feature selection [For classification: chi2, f_classif, mutual_info_classif]
    :return: The weights for each feature as a pandas.DataFrame with size (features, labels+2) with:
            - One column for each label (e.g. C0013264, C0917713, C3542021)
            - One column with the tokens ('token') and
            - One column with the maximum weight for each token across all labels
    '''
    # Select k best features
    # Create feature selector
    selector = SelectKBest(score_func=scoreFunction, k='all')
    # Repeat for each label (e.g. C0013264, C0917713, C3542021)
    X_new_df = pd.DataFrame(list(tokens.loc[:, 'token']),columns=['token'])
    # print(X_new_df)
    for label in range(len(labelsNames)):
        # Fit scores per feature
        selector_scores = selector.fit(X, Y[:,label])
        # print(selector_scores.scores_ )
        # Add scores in the dataframe (as a pd.Series)
        X_new_df[labelsNames[label]] = pd.Series(selector_scores.scores_)
    X_new_df[MaxWeights] = X_new_df[labelsNames].max(axis=1)
    return X_new_df

'''     
        ************************************************
        ***     Preformance evaluation functions     ***
        ************************************************
'''

def getBaselinePredictions(evalDatasetVectorized, baselineDataset, dominantLabel, categories = None):
    '''
    Create some 'baseline' predictions for a specific test datasets parsing the WS predictions for the same dataset, in particular:
        WSLabels:   Predict exactly what the weak supervision heuristic suggests (concept occurrence)
                    For articles without any concept occurrence (rest) do not provide any prediction (no labels assigned).
        WSRestAll:  Predict exactly what WSLabels does. For articles without any concept occurrence (rest) predict all (sub)concepts available.
        AllAll:     For all articles predict all (sub)concepts available. (Ignore WS)
    If the dominantLabel is valid, some baselines using the dominant label as "Major Topic" are also produced. in particular:
        AllM        For all articles predict the dominantLabel. (Ignore WS)
        WSRestM:    Predict exactly what WSLabels does. For articles without any concept occurrence (rest) predict the dominantLabel.
    :param evalDatasetVectorized:   The vectorized evaluation dataset (manually labeled)
    :param baselineDataset:         The folder with the WS labels (as folders/files) for the same set of articles
    :param dominantLabel:           (ignored when invalid) The dominant label used by some baselines
    :param categories:              (optional) The classes to me read from the folder of the WS dataset. Not currently used.
    :return: baselines              A dictionary of baseline predictions for the specific evaluation dataset
    '''
    #   ***     Read evaluation dataset    ***

    # MA test dataset
    scoresDF_avgd = pd.DataFrame()
    scoresDF_perLabel = pd.DataFrame()
    print("Load evaluation dataset:", evalDatasetVectorized)
    documents_eval, labels_eval, labelNames_eval, tfidf_eval, count_eval, tokens_eval = readDataset(
        evalDatasetVectorized)
    print(labelNames_eval)
    # print("data ", documents_eval["pmid"])
    # print("target ", labels_eval)

    # The baselines
    #   ***     Read the WS dataset   ***

    print("Load dataset:", baselineDataset)
    baseline_dataset, baseline_labelNames = loadFromTextFiles(baselineDataset, categories = categories )
    # Covert dataset to multi-label
    print("Covert dataset to multi-label")
    # Merge articles included in more than one folders into one with both labels
    # and replace "filenames" by just "pmids"
    makeDatasetMultiLabel(baseline_dataset)
    # Use MultiLabelBinarizer to create binary 2d matrix with one column per class as done here : [ref](http://scikit-learn.org/stable/modules/multiclass.html#multilabel-classification-format)
    baseline_dataset.target = MultiLabelBinarizer(classes = range(len(baseline_dataset.target_names))).fit_transform(baseline_dataset.target)

    baselines = {
        # Annotate all articles with DS labels only
        "WSLabels": np.zeros(shape=(len(documents_eval), len(labelNames_eval))),
        # Annotate all articles with DS labels and rest articles with all labels
        "WSRestAll": np.zeros(shape=(len(documents_eval), len(labelNames_eval))),
        # Annotate all articles with all labels
        "AllAll": np.zeros(shape=(len(documents_eval), len(labelNames_eval)))
    }

    baselines["AllAll"][:, :] = 1

    # The dominant label used by some baseline methods
    dominantLabelFound = False
    print("dominantLabel:",dominantLabel)
    print("list(labelNames_eval['name'])",list(labelNames_eval["name"]))
    if dominantLabel in list(labelNames_eval["name"]):
        dominantLabelFound = True
        dominantLabelIndex = list(labelNames_eval["name"]).index(dominantLabel)
        print("dominantLabelIndex",dominantLabelIndex)

    if dominantLabelFound:
        # Annotate all articles with with DMD label only
        allM = np.zeros(shape=(len(documents_eval), len(labelNames_eval)))
        allM[:, dominantLabelIndex] = 1
        baselines["AllM"] = allM
        # Annotate all articles with DS labels and rest articles with DMD
        baselines["WSRestM"] = np.zeros(shape=(len(documents_eval), len(labelNames_eval)))

    for pmid, labelPredictions in zip(baseline_dataset.filenames, baseline_dataset.target):
        # Find the index of the specific article in evaluation dataset
        if int(pmid) in list(documents_eval["pmid"]):
            pmidIndex = list(documents_eval["pmid"]).index(int(pmid))
            for i in range(len(baseline_dataset.target_names)):
                target_name = baseline_dataset.target_names[i]
                # if labelPredictions[i]: print(pmid, target_name)
                if not target_name == rest_name:
                    if target_name in list(labelNames_eval["name"]):
                        targetIndex = list(labelNames_eval["name"]).index(target_name)
                        # print("zipped:",pmid,labelPredictions,pmidIndex, targetIndex, target_name,"->",labelPredictions[i])
                        baselines["WSLabels"][pmidIndex, targetIndex] = labelPredictions[i]
                        if dominantLabelFound:
                            baselines["WSRestM"][pmidIndex, targetIndex] = labelPredictions[i]
                        baselines["WSRestAll"][pmidIndex, targetIndex] = labelPredictions[i]
                    # elif target_name == cTop:
                    #     baselines["WSLabels"][pmidIndex, :] = labelPredictions[i]
                    #     if dominantLabelFound:
                    #         baselines["WSRestM"][pmidIndex, :] = labelPredictions[i]
                    #     baselines["WSRestAll"][pmidIndex, :] = labelPredictions[i]

                else:
                    # print("zipped:",pmid,labelPredictions,pmidIndex, target_name,"->",labelPredictions[i])
                    if labelPredictions[i]:
                        if dominantLabelFound:
                            # All "rest" as dominant label
                            baselines["WSRestM"][pmidIndex, dominantLabelIndex] = labelPredictions[i]
                        # All "rest" as all labels
                        baselines["WSRestAll"][pmidIndex, :] = labelPredictions[i]
        else:
            print("Warning! pmid (" + pmid + ") not found in evaluation dataset. (Will be ignored)")

    # clean baseline predictions by overlapping/invalid combinations (if any)
    # # skipped
    # for baseline in baselines:
    #     baselines[baseline] = cleanPredictionOverlaps(baselines[baseline])
        # print(baselines[baseline])

    return baselines

def calculateScoresPerLabel(labels_eval, predicted, label):
    '''
    Calculate evaluation measures per lable for this (:param predicted:) prediction and return them as a Dictionary
    :param labels_eval: The golden labels for evaluation as an ndarray with dimensions (No of samples, No of labels) e.g. [[0,1,1],[1,0,0],[0,1,0]...]
    :param predicted: The predicted labels be evaluated as an ndarray with dimensions (No of samples, No of labels) e.g. [[0,1,1],[1,0,0],[0,1,0]...]
    :param label: The index of the label to calculate measures for (e.g. 0, 1 etc)
    :return: A Dictionary from evaluation measure (e.g. "Acc ex") to corresponding value (e.g. 0.899999)
    '''
    scores = {}
    scores["Acc"] = accuracy_score(labels_eval[:, label], predicted[:, label])
    scores["P"] = precision_score(labels_eval[:, label], predicted[:, label])
    scores["R"] = recall_score(labels_eval[:, label], predicted[:, label])
    # scores["F1 sci"] = f1_score(labels_eval[:,label], predicted[:,label])
    if (scores["P"] + scores["R"]) > 0:
        scores["F1"] = 2 * scores["P"] * scores["R"] / (scores["P"] + scores["R"])
    else:
        scores["F1"] = 0;
    return scores

def calculateScoresAveraged(labels_eval, predicted, cTopIndex = None):
    '''
    Calculate various averaged evaluation measures for this (:param predicted:) prediction and return them as a Dictionary
    :param labels_eval: The golden labels for evaluation as an ndarray with dimensions (No of samples, No of labels) e.g. [[0,1,1],[1,0,0],[0,1,0]...]
    :param predicted: The predicted labels be evaluated as an ndarray with dimensions (No of samples, No of labels) e.g. [[0,1,1],[1,0,0],[0,1,0]...]
    :return: A Dictionary from evaluation measure (e.g. "Acc ex") to corresponding value (e.g. 0.899999)
    '''
    labels = list(range(len(predicted[0])))
    if cTopIndex is not None:
        if cTopIndex in labels:
            labels.remove(cTopIndex)

    print("labels for evaluation: ", labels)
    scores = {}
    # Example based averaging measures
    # subset accuracy: the set of labels predicted for a sample must exactly match the corresponding set of labels in y_true.
    scores["Acc ex"] = accuracy_score(labels_eval, predicted)
    avg_score = np.mean(predicted == labels_eval)
    scores["P ex"] = precision_score(labels_eval, predicted, average='samples', labels=labels)
    scores["R ex"] = recall_score(labels_eval, predicted, average='samples', labels=labels)
    scores["F1 ex"] = f1_score(labels_eval, predicted, average='samples', labels=labels)

    # Micro averaged measures
    # micro averaged accuracy, including "non-labels"
    # for i,j in zip (predicted,labels_eval):
    #     print(i, j,i == j)
    scores["Acc mi"] = avg_score
    scores["P mi"] = precision_score(labels_eval, predicted, average='micro', labels=labels)
    scores["R mi"] = recall_score(labels_eval, predicted, average='micro', labels=labels)
    scores["F1 mi"] = f1_score(labels_eval, predicted, average='micro', labels=labels)

    # Macro averaged measures
    scores["P ma"] = precision_score(labels_eval, predicted, average='macro', labels=labels)
    scores["R ma"] = recall_score(labels_eval, predicted, average='macro', labels=labels)
    if (scores["P ma"] + scores["R ma"]) > 0:
        scores["F1(P,R ma)"] = 2 * scores["P ma"] * scores["R ma"] / (
                    scores["P ma"] + scores["R ma"])
    else:
        scores["F1(P,R ma)"] = 0;
    scores["F1 ma"] = f1_score(labels_eval, predicted, average='macro', labels=labels)
    # print("F1 ma" , f1_score(labels_eval, predicted, average = 'macro', labels = [0,1,2]))
    # for other Fs with parameter e.g. b=0.4 use:
    # fbeta_score(labels_eval, predicted, 0.4, average = 'macro')
    # print(scores)
    return scores

def saveScoreTables(scoresDictionary, pathToCSV):
    '''
    Save calculated scores acheived by classifiers in different datasets into corresponding CSV files
    :param scoresDictionary: A Dictionary from Dataset name (e.g. 'F_50') to corresponding scores as a DataFrame with the scores of different classifiers achieved in this dataset.
                            The index of each DataFrame is the evaluation measure names list (e.g. 'Acc ex', 'F1 mi' etc.) and the columns of the DataFrame are the classifier names (e.g. LinearSVC_F_5,RandomForestClassifier_F_5 etc)
    :param pathToCSV: The full path where to write the CSV file
    :return: None
    '''
    pivotPerDataset = pd.DataFrame()
    for evalDataset in scoresDictionary:
        # if empty, initialize with First DataFrame
        if pivotPerDataset.empty:
            pivotPerDataset = pd.DataFrame(scoresDictionary[evalDataset])
        else:
            pivotPerDataset = pd.concat([pivotPerDataset, scoresDictionary[evalDataset]], axis=1, sort=False)

    names = []
    fsTypes = []
    fsKs = []
    labels = []
    for column in pivotPerDataset:
        fsType = "-"
        fsK = "-"
        label = "-"
        if column.count("_")==2 :
            name, fsType, fsK = column.split("_")
        elif column.count("_") == 3:
            name, fsType, fsK, label = column.split("_")
        elif column.count("_") == 1:
            name, label = column.split("_")
        else :
            name = column
        names.append(name)
        fsTypes.append(fsType)
        fsKs.append(fsK)
        labels.append(label)
    pivotPerDataset.loc['clfType'] = names
    pivotPerDataset.loc['fsType'] = fsTypes
    pivotPerDataset.loc['fsK'] = fsKs
    pivotPerDataset.loc['label'] = labels
        # print(name,fsType,fsK,label)
        # print(pivotPerDataset[column])

    # Save scores in CSV files
    pivotPerDataset.T.to_csv(pathToCSV, ";")

def scoresPerFeatureSelectionType(socresPerDataset, clfTypes, fsType, measure, exp_dir = None):
    '''
    Create a plot of :param measure: scores (e.g. F1 ma) of classifiers across different number of selected features.
    If the optional :param exp_dir: is provided save the plot as a png in the specified folder instead of printing it in the screen.
    :param socresPerDataset: A Dictionary from Dataset name (e.g. 'F_50') to corresponding scores as a DataFrame with the scores of different classifiers achieved in this dataset.
                            The index of each DataFrame is the evaluation measure names list (e.g. 'Acc ex', 'F1 mi' etc.) and the columns of the DataFrame are the classifier names (e.g. LinearSVC_F_5,RandomForestClassifier_F_5 etc)
    :param clfTypes: The classifier types to be considered as a Dictionary from classifier type name (e.g. 'LinearSVC') to corresponding callable class (e.g. LinearSVC())
    :param fsType: The type of the Feature Selection to be considered for the plot as a srting (e.g. 'NoFS', 'chi2', 'f' of 'MI')
    :param measure: The evaluation measure to be considered for the plot as a string (e.g. 'P ma', 'R mi', 'F1 ex' etc)
    :param exp_dir:  (optional) the file to save the plot file (png)
    :return: None
    '''
    # print(socresPerDataset)
    crrent_scores = {}
    df = pd.DataFrame()
    # The minimum value to be used for horizontal line
    total_minimum_value = 1
    for clfType in clfTypes.keys():
        for k in crrent_scores : crrent_scores[k] = 0
        for dataset in socresPerDataset:
            fs, k = dataset.split("_")
            # print(fs, k)
            if fs == fsType:
                # clf name to score
                scores = socresPerDataset[dataset].loc[measure].to_dict()
                for clf in scores.keys():
                    type, fs, k = clf.split("_")
                    if type == clfType:
                        crrent_scores[k] = scores[clf]
        # if empty, initialize with ks
        print("score names", crrent_scores.keys())
        if df.empty:
            df = pd.DataFrame(index=crrent_scores.keys())
        print(crrent_scores.values())
        df[clfType + " with " + fsType] = crrent_scores.values()
        # Update the total minimum value
        total_minimum_value = min(min(crrent_scores.values()),total_minimum_value)

    print(df)
    diagramm = df.plot.bar(title=measure + " " + fsType + " Feature Selection", rot=0, figsize= (7,6))
    diagramm.set_xlabel("k best features")
    diagramm.set_ylabel(measure)
    diagramm.set_ylim(0, 1)

    # Hardcoded horizontal line at the total minimum value
    diagramm.axhline(y=total_minimum_value, xmin=0, xmax=1, color='LightGray', linestyle='--', lw=0.5)

    if exp_dir is None:
        plt.show(block=True)
    else:
        makeFile(exp_dir)
        diagramm.get_figure().savefig(exp_dir + "/" + fsType + "_" + measure + " Feature Selection.png")
    plt.close()

def plotScoresPerDataset(scoresDF, evalDataset, exp_dir = None):
    '''
    Create a plot of multiple (R,P,F1 etc) classifier scores in :param scoresDF: for the dataset :param evalDataset:
    If the optional :param exp_dir: is provided save the plot as a png in the specified folder instead of printing it in the screen.
    :param scoresDF: A DataFrame with the scores of different classifiers achieved in :param evalDataset: dataset.
                        The index of the DataFrame is the evaluation measure names list (e.g. 'Acc ex', 'F1 mi' etc.) and the columns of the DataFrame are the classifier names (e.g. LinearSVC_F_5,RandomForestClassifier_F_5 etc)
    :param evalDataset: The name of the dataset to be used as the title of the plot and corresponding file (e.g. F_20 for dataset with 20 best featrues selected by F of ANOVA)
    :param exp_dir:  (optional) the file to save the plot file (png)
    :return: None
    '''
    # print(scoresDF.T)
    # print(scoresDF)
    figure = scoresDF.plot.bar(title=evalDataset, rot=0, figsize= (15,7))
    figure.set_ylim(0, 1)
    figure.set_ylabel("score")
    figure.set_xlabel("measure")
    # Hardcoded horizontal line at specified y value
    figure.axhline(y=0.4, xmin=0, xmax=1, color='LightGray', linestyle='--', lw=0.5)
    figure.axhline(y=0.3, xmin=0, xmax=1, color='LightGray', linestyle='--', lw=0.5)
    figure.axhline(y=0.5, xmin=0, xmax=1, color='LightGray', linestyle='--', lw=0.5)
    if exp_dir is None:
        plt.show(block=True)
    else:
        makeFile(exp_dir)
        figure.get_figure().savefig(exp_dir + "/" + evalDataset + ".png")
    plt.close()

def cleanPredictionOverlaps(perdiction):
    '''
        Remove "overlapping" annotations i.e. remove DBMD annotation when DMD or BMD are present
    ATTENTION : Only works for DMD, BMD, DBMD case, with these labels having indexes 0,1,2!
    TODO: Implement a general solution for this cleaning such overlaps
    (sparse) array-like, shape = [n_samples, n_classes].
    :param perdiction: The multi-label predictions provided by a classfier as a (sparse) array-like, shape = [n_samples, n_classes].
    :return:  The multi-label predictions with "overlaps" (e.g. DBMB and BMD true) removed as a (sparse) array-like, shape = [n_samples, n_classes].
    '''

    for i in range(len(perdiction)):
        if perdiction[i,0] == 1 or perdiction[i,1] == 1:
            perdiction[i, 2] = 0
    return perdiction

'''     
        *******************************************
        ***     Functions for complete steps    ***
        *******************************************
'''

def createDataset(labelNames_csv,class_csv,output_dir, manual_dataset_dir,manual_dataset_1_dir,manual_dataset_2_dir,
                  manual_dataset_pmids,fields_csv,rest_dir,manual_dataset_1_size,manual_dataset_1_dir_pmids,manual_dataset_2_size,preferredConceptCUI,
                  manual_dataset_2_dir_pmids,test_ratio,train_dir, test_dir,manual_dataset1_pmids,manual_dataset2_pmids):
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

    TODO: In valid combinations (due to hierarchy) should be removed from the dataset for training and evaluation.
        For splitting no problem is caused, since, although invalid combinations are handled as distinct annotations, splitting by the same ratio lead to retaining the general distribution too.

    '''

    # Copy labelNames to ouput folder
    labelNames = pd.read_csv(labelNames_csv, ";", index_col=0)
    saveLabelNames(makeFile(output_dir), labelNames)
    if manual_dataset_pmids:
        saveLabelNames(makeFile(manual_dataset_dir), labelNames)
    if manual_dataset1_pmids or manual_dataset_1_dir_pmids:
        saveLabelNames(makeFile(manual_dataset_1_dir), labelNames)
    if manual_dataset2_pmids or manual_dataset_2_dir_pmids:
        saveLabelNames(makeFile(manual_dataset_2_dir), labelNames)

    # Read articles per Ct (i.e. class)
    with open(class_csv, encoding="utf8") as csvfile:
        csvreader = csv.reader(csvfile, delimiter=';', quotechar='"')
        datasets = {}
        for row in csvreader:
            if row[1] != "cui":
                if row[1] in datasets:
                    datasets[row[1]].append(row[3])
                else:
                    datasets[row[1]] = [row[3]]
        for key in datasets:
            print(key, " ", len(datasets[key]))

    # Create directories for output
    keys_plus = list(datasets.keys())
    keys_plus.append(rest_name)
    for key in keys_plus:
        print(output_dir + "/" + key)
        if not os.path.exists(output_dir + "/" + key):
            os.makedirs(output_dir + "/" + key)

    # Read and write article texts
    # Articles as text files organized in one folder per class

    with open(fields_csv, encoding="utf8") as csvfile:
        csvreader = csv.reader(csvfile, delimiter=';', quotechar='"')
        for row in csvreader:
            if row[0] != "pmid":  # Skipp the first line with headers
                found = 0
                for key in datasets:
                    if row[0] in datasets[key]:
                        # print(row[0]," add in ",key)
                        found = 1
                        file = codecs.open(output_dir + '/' + key + '/' + row[0] + ".txt", "w+", "utf-8")
                        file.write(row[2])  # Tile
                        file.write(row[1])  # Abstract
                        file.close()
                if not found:
                    file = codecs.open(rest_dir + '/' + row[0] + ".txt", "w+", "utf-8")
                    file.write(row[2])
                    file.write(row[1])
                    file.close()
    '''
    #   *********************************************************
    #   ***     Dataset Finalization (MA, MA1, MA2 removal)   ***
    #   *********************************************************
    '''

    # Load dataset from files
    # Into a "Bunch" object, directly from text files organized in one folder per class (ref)

    print("Load dataset from : ", output_dir)
    initial_dataset, labelNames = loadFromTextFiles(output_dir)

    # Print Bunch fields and content
    print("Bunch fields and content loaded:")
    print("\t keys", initial_dataset.keys())
    print("\t target names", initial_dataset.target_names)

    # Covert dataset to multi-label
    print("Covert dataset to multi-label")
    # Merge articles included in more than one folders into one with both labels
    # and replace "filenames" by just "pmids"
    makeDatasetMultiLabel(initial_dataset)

    # print dataset number of elements
    print("dataset number of elements")
    print("\t data text :", len(initial_dataset.data))
    print("\t data ids :", len(initial_dataset.filenames))
    print("\t label lists :", len(initial_dataset.target))

    if manual_dataset_pmids:
        # Save article text for MA dataset articles in MA folder
        saveDatasetTextFilesSubset(manual_dataset_pmids, manual_dataset_dir, initial_dataset.target_names,
                                   labelNames, initial_dataset.data, initial_dataset.target, initial_dataset.filenames)
        removeArticlesFromDataset(manual_dataset_pmids, initial_dataset.data, initial_dataset.target,
                                  initial_dataset.filenames)
    if manual_dataset1_pmids:
        # Save article text for MA1 dataset articles in MA1 folder
        saveDatasetTextFilesSubset(manual_dataset1_pmids, manual_dataset_1_dir, initial_dataset.target_names,
                                   labelNames, initial_dataset.data, initial_dataset.target, initial_dataset.filenames)
        removeArticlesFromDataset(manual_dataset1_pmids, initial_dataset.data, initial_dataset.target,
                                  initial_dataset.filenames)
    elif manual_dataset_1_size:
        # Select articles for the MA1
        pmids_MA1 = selectRandomSubset(manual_dataset_1_size, initial_dataset.filenames)
        # Write MA1 pmids in file
        pmidFile_MA1 = open(manual_dataset_1_dir_pmids, "w")
        for pmid in pmids_MA1:
            print(pmid, file=pmidFile_MA1)
        pmidFile_MA1.close()

        # Save article text for MA1 dataset articles in MA1 folder
        saveDatasetTextFilesSubset(manual_dataset_1_dir_pmids, manual_dataset_1_dir, initial_dataset.target_names,
                                   labelNames, initial_dataset.data, initial_dataset.target, initial_dataset.filenames)

        # Remove MA1 dataset articles
        # IMPORTANT: This leaves the dataset without "non valid instances" i.e. combinations of BDMD label with DMD or BMD)
        print("Remove MA1 dataset articles : ", manual_dataset_1_dir_pmids)
        # Articles used in the MA1 dataset should not be also included in the training and evaluation datasets

        removeArticlesFromDataset(manual_dataset_1_dir_pmids, initial_dataset.data, initial_dataset.target,
                                  initial_dataset.filenames)

        # Check new (reduced) size of dataset
        # print dataset number of elements
        print("print dataset number of elements")
        print("\t data text :", len(initial_dataset.data))
        print("\t data ids :", len(initial_dataset.filenames))
        print("\t label lists :", len(initial_dataset.target))
        print("\t label lists :", initial_dataset.target)
    if manual_dataset2_pmids:
        # Save article text for MA1 dataset articles in MA1 folder
        saveDatasetTextFilesSubset(manual_dataset2_pmids, manual_dataset_2_dir, initial_dataset.target_names,
                                   labelNames, initial_dataset.data, initial_dataset.target, initial_dataset.filenames)
        removeArticlesFromDataset(manual_dataset2_pmids, initial_dataset.data, initial_dataset.target,
                                  initial_dataset.filenames)
    elif manual_dataset_2_size:
        preferredLabelIndex = -1
        # Save article text for MA2 dataset articles in MA2 folder
        if preferredConceptCUI in initial_dataset.target_names:
            preferredLabelIndex = initial_dataset.target_names.index(preferredConceptCUI)

        # Select articles for the MA2
        pmids_MA2 = selectBalancedSubset(manual_dataset_2_size, initial_dataset.target, initial_dataset.filenames, 0.5,
                                         preferredLabelIndex)
        # Write MA2 pmids in file
        pmidFile_MA2 = open(manual_dataset_2_dir_pmids, "w")
        for pmid in pmids_MA2:
            print(pmid, file=pmidFile_MA2)
        pmidFile_MA2.close()

        # Save article text for MA2 dataset articles in MA1 folder
        saveDatasetTextFilesSubset(manual_dataset_2_dir_pmids, manual_dataset_2_dir, initial_dataset.target_names,
                                   labelNames, initial_dataset.data, initial_dataset.target, initial_dataset.filenames)

        print("Remove MA2 dataset articles : ", manual_dataset_2_dir_pmids)

        # Remove MA2 dataset articles
        removeArticlesFromDataset(manual_dataset_2_dir_pmids, initial_dataset.data, initial_dataset.target,
                                  initial_dataset.filenames)

    # Check new (reduced) size of dataset
    # print dataset number of elements
    print("print dataset number of elements")
    print("\t data text :", len(initial_dataset.data))
    print("\t data ids :", len(initial_dataset.filenames))
    print("\t label lists :", len(initial_dataset.target))
    print("\t label lists :", initial_dataset.target)

    # Split datasets
    if test_ratio != 0:
        print("Split dataset into training and test")
        # Split into training and test datasets (ref: http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html)

        X_train, X_test, Xnames_train, Xnames_test, y_train, y_test = train_test_split(initial_dataset.data,
                                                                                       initial_dataset.filenames,
                                                                                       initial_dataset.target,
                                                                                       test_size=test_ratio,
                                                                                       random_state=0,
                                                                                       stratify=initial_dataset.target)
        # Training dataset
        # print("y train",y_train)
        print(len(X_train))
        print(len(y_train))
        print(len(Xnames_train))
        # Test dataset
        # print("y test",y_test)
        print(len(X_test))
        print(len(y_test))
        print(len(Xnames_test))
    else:
        print("Split dataset into training and test skipped")

    '''
    #   *******************************
    #   ***     Save the datasets   ***
    #   *******************************
    '''

    # Write datasets into folders with text files per Class
    print("Write files ")

    if test_ratio != 0:
        # Training dataset
        saveDatasetTextFiles(train_dir + "/", initial_dataset.target_names, labelNames, X_train, y_train,
                             Xnames_train, True)
        # Test dataset
        saveDatasetTextFiles(test_dir + "/", initial_dataset.target_names, labelNames, X_test, y_test,
                             Xnames_test)
    else:
        # No Split
        # Training dataset
        saveDatasetTextFiles(train_dir + "/", initial_dataset.target_names, labelNames, initial_dataset.data,
                             initial_dataset.target, initial_dataset.filenames, True)

def tranformTrainingDataset(input_dir, outut_dir,useCUIS, occurrence_csv,feature_prefix):
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

    '''
    #   *******************************
    #   ***     load the dataset    ***
    #   *******************************
    '''

    print("Load dataset from : ", input_dir)
    training_dataset, labelNames = loadFromTextFiles(input_dir)

    # Covert dataset to multi-label
    print("Covert dataset to multi-label")
    # Merge articles included in more than one folders into one with both labels
    # and replace "filenames" by just "pmids"
    makeDatasetMultiLabel(training_dataset)
    # Print Bunch fields and content
    print("Bunch fields and content loaded:")
    print("\t keys", training_dataset.keys())
    print("\t target names", training_dataset.target_names)
    print("\t size", len(training_dataset.data))

    '''
    #   *********************************
    #   ***     Make labels Binary    ***
    #   *********************************
    '''
    # Use MultiLabelBinarizer to create binary 2d matrix with one column per class as done here : [ref](http://scikit-learn.org/stable/modules/multiclass.html#multilabel-classification-format)
    training_dataset.target = MultiLabelBinarizer().fit_transform(training_dataset.target)
    '''
    #   ************************************
    #   ***     Vectorize the dataset    ***
    #   ************************************
    '''

    print("Tokenize dataset")
    # Tokenize
    train_counts, feature_names = tokenizeArticleText(training_dataset.data)

    # print("Sparse matrix :",train_counts.shape())
    print("feature_names :", feature_names)

    if useCUIS:
        # Get a dictionary from CUI to corresponding list of pmidIndexes for articles of the dataset where it has been recognized
        conceptToArticle = getConceptToArticleOccurrenceMap(occurrence_csv, training_dataset)

        # for each CUI
        for cui in conceptToArticle.keys():
            # Create a column of zeros for this CUI, with the length of the sample articles
            tmpColumn = np.array([[0]] * len(training_dataset.filenames))
            # For each article where this CUI occurs
            for pmidIndex in conceptToArticle[cui]:
                # Update the count of occurrences for this article TODO: Use actual occurrence number instead of default 1
                tmpColumn[pmidIndex] = 1
            # Add CUI counts in train_counts matrix
            train_counts = hstack((train_counts, tmpColumn))
            # Add CUI in feature names list
            feature_names.append(feature_prefix + cui)

    # print(train_counts)
    # print(feature_names)

    # get tfidf
    tfidf = getTFIDF(train_counts)


    '''
    #   **************************************
    #   ***     Save vectorized dataset    ***
    #   **************************************
    '''
    saveDatasetFromBunch(outut_dir, training_dataset, tfidf, train_counts, feature_names, labelNames)

def analyzeTrainigDataset(input_dir,featureKs,scoreFunctions):
    '''
        STEP 3

        ***********************************
        ***     Analyze the Dataset     ***
        ***********************************

        Perform Feature Selection and create new training datasets with selected Features only
    '''

    '''
    #   **************************************
    #   ***     Read vectorized dataset    ***
    #   **************************************
    '''
    print(datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'))

    # documents:      A pandas.DataFrame with columns : 'pmid', 'text'
    # labels:         A numpy.ndarray with label annotations (e.g. [[0,1,0], [0,0,1], [1,1,0] ...]) {"pickled"/serialized for saving}
    # labelNames:     A pandas.DataFrame with columns : 'name'
    # tfidf:          A scipy.sparse.csr_matrix (document id, token id)
    # tokens:         A pandas.DataFrame with columns : 'token'. Provides a mapping from token id to token string
    documents, labels, labelNames, tfidf, count, tokens = readDataset(input_dir)

    # Get labelNames as a list
    # labelNames_list = list(labelNames.loc[:,'name'])
    labelNames_list = list(labelNames.loc[:, 'abbreviation'])
    token_list = list(tokens.loc[:, 'token'])

    '''
    #   *********************************
    #   ***     Feature selection     ***
    #   *********************************
    '''
    # Basic statistics for the dataset
    # tfidf_df = pd.DataFrame(X_train_tfidf)
    # print(tfidf_df.describe())

    #  ***  Feature selection   ***
    print("Feature selection")

    # Based in this examples ([ref](https://github.com/MSc-in-Data-Science/class_material/blob/master/semester_1/Machine_Learning/Lecture_11-FeatureSelection/FeatureSelection.ipynb))

    # Convert tfidf data to ndarray from crs matrix ([ref](https://stackoverflow.com/questions/31228303/scikit-learns-pipeline-error-with-multilabel-classification-a-sparse-matrix-w?utm_medium=organic&utm_source=google_rich_qa&utm_campaign=google_rich_qa))
    print("labels, Y (documents, labels): ", labels.shape)
    Y = labels
    print()

    print("1. Univariate Selection")

    for kFeatures in featureKs:
        for scoreFunction in scoreFunctions:
            print("Selecting ", kFeatures, " by", scoreFunction, ":")
            # Select top features/token only and transofm tfidf matrix and tokens DataFrame accordingly
            tfidf_selectedFeatures, count_selectedFeatures, tokens_selectedFeatures = getTopFeatures(tfidf, count, Y,
                                                                                                     tokens,
                                                                                                     labelNames_list,
                                                                                                     scoreFunctions[
                                                                                                         scoreFunction],
                                                                                                     kFeatures)
            #  ***  Save reduced dataset   ***
            saveDatasetFromDFs(input_dir + '_FSby_' + scoreFunction + '_' + str(kFeatures), documents, labels,
                               labelNames, tfidf_selectedFeatures, count_selectedFeatures, tokens_selectedFeatures)

    # TODO: Use more sophisticated FS methods
    # print("2.  Recursive feature elimination")
    # print("3.  Feature selection using SelectFromModel")

def creatTestDataByMA(train_dir,inputText_dir,evalLabels_csv,ouputText_dir):
    '''
        STEP 4

        *********************************************************
        ***     Create a Manually Annotated (MA) Dataset      ***
        *********************************************************

        Read the text of articles included in the MA Dataset from corresponding folder (created during WS Dataset creation)
        Read the MA classes/labels from a CSV file created manually
        Write the MA Dataset in text files, in format adequate to be read by the reading scripts
    '''
    '''
    #   **************************************
    #   ***     Load the data from files   ***
    #   **************************************
    '''
    print("Load classes from the trainig dataset : ", train_dir)
    vocabulary, trainCount, classes = readTrainReference(train_dir)
    print(">>>", classes)

    # Load evaluation articles and label/class names
    evalText_dataset, labelNames = loadFromTextFiles(inputText_dir)
    # print('evalText_dataset.filenames:', evalText_dataset.filenames)
    # Covert dataset to multi-label
    print("Covert dataset to multi-label")
    # Merge articles included in more than one folders into one with both labels
    # and replace "filenames" by just "pmids"
    makeDatasetMultiLabel(evalText_dataset)

    # Load evaluation (manual) article labels
    evalLabels = pd.read_csv(evalLabels_csv, ";")
    # Create abbreviation-name map
    abbrTocui = {}

    print(labelNames)
    for cui in labelNames.index:
        abbrTocui[labelNames.loc[cui]['abbreviation']] = cui
    # Create label folders if not exist
    for dir in abbrTocui.values():
        if not os.path.exists(ouputText_dir + "/" + dir):
            os.makedirs(ouputText_dir + "/" + dir)

    '''
    #   ************************************
    #   ***     Save as labeled files    ***
    #   ************************************
    '''

    # Convert evaluation (manual) article labels to CUIs
    # Foreach annotated article
    for i in range(len(evalLabels)):
        # Get all manual classes of this article from CSV
        parts = evalLabels.loc[i, 'Manual Class'].split(',')
        # Get the pmid of this article from CSV
        pmid = str(evalLabels.loc[i, 'pmid'])
        # foreach manual label of this article
        for part in parts:
            # write the test in the specified folder named after the CUI of the label
            labelCUI = abbrTocui[part.strip()]
            file = codecs.open(ouputText_dir + "/" + labelCUI + '/' + pmid + ".txt", "w+", "utf-8")
            # Find corresponding article text
            index = evalText_dataset.filenames.index(pmid)
            # index = evalText_dataset
            file.write(evalText_dataset.data[index])
            file.close()

    saveLabelNames(ouputText_dir, labelNames)

def tranformTestData( train_dirs, input_dir,occurrence_csv,feature_prefix,outut_dir):
    '''
        STEP 5

        ************************************
        ***     Transform MA dataset     ***
        ************************************
        Make target a binary 2d matrix with one column per class
        Tokenize the text of each sample and calculate TFIDF features for the tokens
        Save the Transformed datasets in a new direcotries
    '''
    '''
    #   *******************************
    #   ***     load the dataset    ***
    #   *******************************
    '''
    firstTraindir = train_dirs[list(train_dirs.keys())[0]]
    print("Load classes from the first reference dataset : ", firstTraindir)
    print("\t all alternative reference datasets should have the same classes anyway, their feautre-space only deffers ")
    vocabulary, trainCount, classes = readTrainReference(firstTraindir)
    print(">>>", classes)

    print("Load dataset from : ", input_dir)
    # No categories give, these "categories" will be the reference for the other datasets
    test_dataset, labelNames = loadFromTextFiles(input_dir)

    # Covert dataset to multi-label
    print("Covert dataset to multi-label")
    # Merge articles included in more than one folders into one with both labels
    # and replace "filenames" by just "pmids"
    makeDatasetMultiLabel(test_dataset)
    # Print Bunch fields and content
    print("Bunch fields and content loaded:")
    print("\t keys", test_dataset.keys())
    print("\t target names", test_dataset.target_names)
    print("\t size", len(test_dataset.data))

    '''
    #   *********************************
    #   ***     Make labels Binary    ***
    #   *********************************
    '''
    # Use MultiLabelBinarizer to create binary 2d matrix with one column per class as done here : [ref](http://scikit-learn.org/stable/modules/multiclass.html#multilabel-classification-format)
    test_dataset.target = MultiLabelBinarizer().fit_transform(test_dataset.target)
    '''
    #   ************************************
    #   ***     Vectorize the dataset    ***
    #   ************************************
    '''

    # Vectorization based on corresponding training dataset reference Vocabulary and IDFs
    for train_name in train_dirs:
        train_dir = train_dirs[train_name]
        print("Load reference dataset from : ", train_dir)
        vocabulary, trainCount, classes = readTrainReference(train_dir)
        print(">>>>", classes)
        # print(">>>>", vocabulary)
        # print(">>>>", trainCount)

        print("Tokenize dataset")
        # Tokenize
        count, feature_names = tokenizeArticleText(test_dataset.data, vocabulary)

        # print("Sparse matrix :",counts.shape())
        print("feature_names :", feature_names)

        # Get a dictionary from CUI to corresponding list of pmidIndexes for articles of the dataset where it has been recognized
        conceptToArticle = getConceptToArticleOccurrenceMap(occurrence_csv, test_dataset)

        print(conceptToArticle)
        # Update concept occurrences
        for feature in feature_names:
            # For concept features (i.e. CUIs)
            if str(feature).startswith(feature_prefix):
                # Update counts for articles where the cui occurs
                featureIndex = feature_names.index(feature)
                cui = str(feature.replace(feature_prefix, ''))
                # If the CUI occurs in some articles
                if cui in conceptToArticle.keys():
                    # For each article where this CUI occurs
                    for pmidIndex in conceptToArticle[cui]:
                        # Update the count of occurrences for this article TODO: Use actual occurrence number instead of default 1
                        count[pmidIndex, featureIndex] = 1

        # # Print term frequency and actual term for specific pair of article and termid
        # article_index = 0;
        # term_index = 14755
        # # term_id = 2932
        # print("Article ", training_dataset.filenames[article_index],
        #       " contains ", counts[article_index, term_index],
        #       " times the term : ", feature_names[term_index],
        #       "in text :", training_dataset.data[article_index])

        # get tfidf
        tfidf = getTFIDF(count, trainCount)
        # # Print tfidf and actual term for specific pair of article and termid
        # article_index = 0;
        # term_index = 10664
        # # term_id = 2932
        # print("Article ", training_dataset.filenames[article_index],
        #       " contains ", counts[article_index, term_index],
        #       " times the term : ", feature_names[term_index],
        #       " with tfidf : ", tfidf[article_index, term_index],
        #       "in text :", training_dataset.data[article_index])

        '''
        #   **************************************
        #   ***     Save vectorized dataset    ***
        #   **************************************
        '''
        saveDatasetFromBunch(outut_dir + "_" + train_name, test_dataset, tfidf, trainCount, feature_names, labelNames)

def createClassifiers(datasets,evalDatasets,exp_dir,test_folder_name,baselineDataset, dominantLabel, cTop):
    '''
        STEP 6

        ***********************************
        ***     Train a Classifier      ***
        ***********************************
    '''

    '''
    #   ***************************************
    #   ***     Read vectorized datasets    ***
    #   ******+*********************************
    '''
    # documents:      A pandas.DataFrame with collumns : pmid, text
    # labels:         A "pickled" (serialized) python object with label annotations (e.g. [[0,1,0], [0,0,1], [1,1,0] ...])
    # labelNames:     A pandas.DataFrame with collumns : name
    # tfidf:          A scipy.sparse.csr_matrix (document id, token id)
    # tokens:         A pandas.DataFrame with collumns : token

    labelNames_list = ''
    # The tfidf features
    tfidf_dict = {}
    # The target/label values
    labels_dict = {}
    # The feature/token names
    tokens_dict = {}
    for dataset in datasets:
        print("Load dataset:", dataset)
        documents, labels, labelNames, tfidf, count, tokens = readDataset(datasets[dataset])
        # Get labelNames as a list
        labelNames_list = list(labelNames.loc[:,'name'])
        # labelNames_list = list(labelNames.loc[:, 'abbreviation'])
        tokens_list = list(tokens.loc[:, 'token'])
        tokens_dict[dataset] = tokens_list
        # Convert training dataset to ndarray from crs matrix ([ref](https://stackoverflow.com/questions/31228303/scikit-learns-pipeline-error-with-multilabel-classification-a-sparse-matrix-w?utm_medium=organic&utm_source=google_rich_qa&utm_campaign=google_rich_qa))
        # print("convert dataset to ndarray",tfidf.shape)
        X_train_tfidf = tfidf.toarray()
        tfidf_dict[dataset] = X_train_tfidf
        labels_dict[dataset] = labels
        # print(labels)

    '''
    #   **********************************
    #   ***     Train classifiers      ***
    #   **********************************
    '''
    print("Train the classifiers")
    # *** OneVsRest Classifier *** ([ref](http://scikit-learn.org/stable/modules/multiclass.html#multiclass-learning))
    clfTypes = {
        # 1. A Linear Support Vector Classifier
        'LinearSVC': LinearSVC(),
        # 2. A Random Forest Classifier
        'RandomForestClassifier': RandomForestClassifier(),
        # 3. A desision Tree Classifier
        'DecisionTreeClassifier': DecisionTreeClassifier(),
        # 4. A LogisticRegression Classifier
        'LogisticRegression': LogisticRegression(),
    }

    # cTop index initialization
    cTopIndex = None

    cvScores = {}

    # The classifiers
    clf_dict = {}
    for dataset in datasets:
        clfs = {}

        # Simple training
        for clfType in clfTypes:
            clf = OneVsRestClassifier(clfTypes[clfType])
            if "LogisticRegression" in clfType:
                clf.set_params(estimator__penalty='l2')
            if "RandomForestClassifier" in clfType:
                clf.set_params(estimator__random_state=0)
            if 'DecisionTreeClassifier' in clfType:
                # clf.set_params(estimator__max_leaf_nodes=len(documents)/len(labelNames))
                clf.set_params(estimator__max_leaf_nodes=50)
                # print(clf.get_params().keys())
            # train the model
            clf.fit(tfidf_dict[dataset], labels_dict[dataset])
            clf_name = clfType + '_' + dataset
            clfs[clf_name] = clf
            print("train ", clf_name, "on", dataset)

            # cTop index calculation
            if cTopIndex is None:
                for label_index in range(len(labels_dict[dataset][0])):
                    if labelNames_list[label_index] == cTop:
                        cTopIndex = label_index

        clf_dict[dataset] = clfs

    '''
    #   *************************************
    #   ***     Evaluate classifiers      ***
    #   *************************************
    '''
    averaged_socresPerDataset = {}
    socresPerDatasetPerLabel = {}

    # ***   By test dataset     ***
    print("Evaluate on test dataset")

    for evalDataset in evalDatasets:
        evalText_dir = evalDatasets[evalDataset]
        scoresDF_avgd = pd.DataFrame()
        scoresDF_perLabel = pd.DataFrame()
        print("Load evaluation dataset:", evalDataset)
        documents_eval, labels_eval, labelNames_eval, tfidf_eval, count_eval, tokens_eval = readDataset(evalText_dir)
        for clfName in clf_dict[evalDataset]:
            # One score per classifier
            averaged_scores = {}
            clf = clf_dict[evalDataset][clfName]
            predicted = clf.predict(tfidf_eval)

            # print some basic scores
            print("Some scores for:", clfName)
            print('labels_eval')
            print('\t names:',labelNames_eval)
            print('\t shape: ',labels_eval)
            print('labels_predict')
            print('\t names:', clf.classes_)
            print('\t names_list:', labelNames_list)
            print('\t shape: ', predicted)
            # print(predicted)

            print(classification_report(labels_eval, predicted))
            averaged_scores = calculateScoresAveraged(labels_eval, predicted, cTopIndex)

            # if empty, initialize with score names
            if scoresDF_avgd.empty:
                scoresDF_avgd = pd.DataFrame(index=averaged_scores.keys())
            # Add socres to the DataFrame
            scoresDF_avgd[clfName] = averaged_scores.values()

            #  Once score per classifier-label combination
            for label in range(len(predicted[0])):
                if labelNames_list[label] != cTop:
                    print("Label :", labelNames_list[label])
                    print("Golden :", labels_eval[:, label], )
                    print("Prediction :", predicted[:, label], )
                    scores = calculateScoresPerLabel(labels_eval, predicted, label)

                    # if empty, initialize with score names
                    if scoresDF_perLabel.empty:
                        scoresDF_perLabel = pd.DataFrame(index=scores.keys())
                    # Add socres to the DataFrame
                    scoresDF_perLabel[clfName + "_" + labelNames_list[label]] = scores.values()

        socresPerDatasetPerLabel[evalDataset] = scoresDF_perLabel

        # print(scoresDF)
        averaged_socresPerDataset[evalDataset] = scoresDF_avgd

    '''
    #   *************************************
    #   ***     Create some baselines     ***
    #   *************************************
    '''

    baselinePredictions = getBaselinePredictions(evalDatasets["NoFS_0"], baselineDataset, dominantLabel, cTop)
    # print(baselinePredictions)
    scoresDF_avgd = pd.DataFrame()
    scoresDF_perLabel = pd.DataFrame()
    for baseline in baselinePredictions:
        predicted = baselinePredictions[baseline]
        print("Some scores for:", baseline)
        print(classification_report(labels_eval, predicted))
        averaged_scores = calculateScoresAveraged(labels_eval, predicted, cTopIndex)

        # if empty, initialize with score names
        if scoresDF_avgd.empty:
            scoresDF_avgd = pd.DataFrame(index=averaged_scores.keys())
        # Add socres to the DataFrame
        scoresDF_avgd[baseline] = averaged_scores.values()

        #  Once score per classifier-label combination
        for label in range(len(predicted[0])):
            if labelNames_list[label] != cTop:
                print("Label :", labelNames_list[label])
                print("Golden :", labels_eval[:, label], )
                print("Prediction :", predicted[:, label], )
                scores = calculateScoresPerLabel(labels_eval, predicted, label)

                # if empty, initialize with score names
                if scoresDF_perLabel.empty:
                    scoresDF_perLabel = pd.DataFrame(index=scores.keys())
                # Add socres to the DataFrame
                scoresDF_perLabel[baseline + "_" + labelNames_list[label]] = scores.values()

    socresPerDatasetPerLabel["baselines"] = scoresDF_perLabel

    # print(scoresDF)
    averaged_socresPerDataset["baselines"] = scoresDF_avgd

    print("CV scores",cvScores)
    '''
    #   *******************************************
    #   ***     Create plots of preformance     ***
    #   *******************************************
    '''

    # ***   Create pivot tables with scores     ***
    saveScoreTables(averaged_socresPerDataset, exp_dir + '/' + "ScoresPerDataset_" + test_folder_name + ".csv")
    saveScoreTables(socresPerDatasetPerLabel, exp_dir + '/' + "socresPerDatasetPerLabel_" + test_folder_name + ".csv")
