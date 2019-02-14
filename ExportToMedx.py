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

fields_csv = baseFolder + caseStudy + '_abstract.csv';

# Initial CSV files for dataset creation
maFolder = final_dir + '/forATSI'
maFile = maFolder + '/articles.tsv'

# Manually annotated articles to be excluded from Distantly Supervised dataset and held out data
maDir = baseFolder + 'MA_Data/' + caseStudy + '/'
# manual_dataset_1_pmids = maDir + '/MA1_pmids.txt'; deprecated
manual_dataset_1_dir = final_dir + "/MA1/";
manual_dataset_2_dir = final_dir + "/MA2/";
manual_dataset_1_dir_pmids = manual_dataset_1_dir + "pmids.txt";
manual_dataset_2_dir_pmids = manual_dataset_2_dir + "pmids.txt";

'''
#   ********************************
#   ***     Load MA datasets     ***
#   ********************************
'''

makeFile(maFolder)

# Read ids of articles from corresponding files
ma1PmidsFile = open(manual_dataset_1_dir_pmids, "r")
#  list of pmids in MA1
ma1Pmids = ma1PmidsFile.readlines()
ma1PmidsFile.close()
ma2PmidsFile = open(manual_dataset_2_dir_pmids, "r")
#  list of pmids in MA2
ma2Pmids = ma2PmidsFile.readlines()
ma2PmidsFile.close()

# Create a list of pmids in both MA datasets
maPmids = []
for pmid in (ma1Pmids + ma2Pmids):
    # clean the pmid from extra characters
    pmid = pmid.replace("\n", "")
    pmid = pmid.replace("\'", "")
    maPmids.append(pmid)

# shuffle(maPmids)
# print(maPmids)

# Create a Pandas.DataFrame
export_tsv = pd.DataFrame({"pmid":[],"abstractText":[],"title":[],'labels':[], '':[]})
export_tsv.set_index("pmid")
with open(fields_csv,encoding="utf8") as csvfile:
    csvreader = csv.reader(csvfile, delimiter=';', quotechar='"')
    for row in csvreader:
        if row[0] != "pmid": # Skipp the first line with headers
            currentPmid = row[0] # pmid
            currentAbstract = row[1] # Abstract
            currentTitle = row[2] # Tile
            if currentPmid in maPmids:
                export_tsv.loc[currentPmid] = [currentPmid,currentAbstract, currentTitle,'','']
export_tsv = shuffle(export_tsv)
export_tsv.to_csv(maFile,"\t",quoting=csv.QUOTE_NONE, index=False)