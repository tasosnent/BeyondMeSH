from DatasetFunctions import *
from random import shuffle
import csv
import pandas as pd
from sklearn.utils import shuffle

'''
    ******************************************************************************
    ***     Export a Set of articles in a TSV format to be imported in ATSI    ***
    ******************************************************************************
'''
'''
#   **************************************************
#   ***     Hardcoded settings for the dataset     ***
#   **************************************************
'''

# The case study
from typing import List, Any

# caseStudy = 'DMD';
    # 1 :   'C0013264'  DMD
    # 2 :   'C0917713'  BMD
    # 3 :   'C3542021'  BDMD
caseStudy = 'AD';
# caseStudy = 'LC';
baseFolder = 'D:/05 MeSHToUMLS/AD_Dataset_20180927/'
# Final directories with article text into pmid name files, grouped into folders named after the class CUI
final_dir = baseFolder + "FinalSplit"
final_dir = final_dir + "/" + caseStudy

# Initial CSV files for dataset creation
maFolder = final_dir + '/forATSI'
articleFile = maFolder + '/articles-to-annotate-csv'
assignmentFile = maFolder + '/assignments.tsv'

# Manually annotated articles to be excluded from Distantly Supervised dataset and held out data
maDir = baseFolder + 'MA_Data/' + caseStudy + '/'

# User IDs (from Medx pltaform) corresponding to users that should be assigned with this set of articles
userIDs = ["26"]

'''
#   ********************************
#   ***     Load MA datasets     ***
#   ********************************
'''

export_tsv = pd.DataFrame({"User ID":[],"Article ID":[]})

# Read list of pmids with Nids
with open(articleFile, encoding="utf8") as csvfile:
    csvreader = csv.reader(csvfile, delimiter=',', quotechar='"')
    index = 0
    for row in csvreader:
        if row[0] != "Nid":
            for user in userIDs:
                Nid = row[0]
                export_tsv.loc[index] = [user,Nid]
                index += 1
    csvreader

export_tsv.to_csv(assignmentFile, "\t", quoting=csv.QUOTE_NONE, index=False)