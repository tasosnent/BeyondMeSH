from DatasetFunctions import *
import datetime
import time

'''     
    ****************************************************
    ***     Read Manually Annotated (MA) Datasets    ***
    ****************************************************

    Read Manually Annotated (MA) Datasets (as CSV files exported from the MedX tool)
    Convert into :
        - CSV files adequate to be read from "TestDatasetCreateByMA.py" (columns "pmid" and "Manual Class")
        - Labeled dataset folders with (empty) files (just to have the labels in appropriate format)
'''
'''
#   *****************************************************
#   ***     Hardcoded settings for the classifier     ***
#   *****************************************************
'''

base = 'C:/Users/tasosnent/Desktop/Tasks/12 PhD/MeSH_to_UMLS/ManualDS/AD/v1'
labelNames_csv = base + '/AD_labelNames.csv';
labelFiles ={
    'Natasha':  base + "/annotations_per_user_2019-01-03T18-55-47_Natasha.csv",
    'Nikil':    base + "/annotations_per_user_2019-01-03T18-55-16_Nikil.csv",
}
labels ={
    'FAD':  '[M0584546]',
    'EOAD': '[M0333931]',
    'LOAD': '[M0333941]',
    'PD':   '[M0005798]',
    'FOAD': '[M0333961]',
    'ACSD': '[M0333929]',
    'AD':   'None of the above',
}
# Make Pandas print all columns of a DataFrame when less or equal to 20
pd.set_option('display.max_columns', 20)

'''
#   **************************************
#   ***     Load the data from files   ***
#   **************************************
'''
labelNames = pd.read_csv(labelNames_csv, ";", index_col=0)

print(datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'))

'''
#   ************************************
#   ***     Parse labels per PMID    ***
#   ************************************
'''

for annotator in labelFiles:
    annotations = pd.DataFrame()
    evalLabels = pd.read_csv(labelFiles[annotator], ",")
    print (annotator)
    for i in range(len(evalLabels)):
        labelsAssigned = evalLabels.loc[i, 'Labels']
        pmid = str(evalLabels.loc[i,'pmid'])
        labelAbbreviations = []
        for label in labels:
            if labels[label] in labelsAssigned:
                labelAbbreviations.append(label)
        jointLabels = ', '.join(labelAbbreviations)
        data = pd.DataFrame({"pmid": [pmid], "Manual Class": [jointLabels]})

        annotations = annotations.append(data, ignore_index=True)
    '''
    #   *******************************
    #   ***     Save as CSV file    ***
    #   *******************************
    '''

    csvPath = base + '/' + annotator + '_Dataset.csv'
    annotations.to_csv(csvPath, ";", index=0)

    '''
    #   ********************************************
    #   ***     Save as labeled (empty) files    ***
    #   ********************************************
    '''

    ouputText_dir = base + '/' + annotator
    makeFile(ouputText_dir )
    # Load evaluation (manual) article labels
    evalLabels = pd.read_csv(csvPath, ";")
    # Create abbreviation-name map
    abbrTocui = {}
    # TODO: Not use labelNames order/indices!!!
    # print(labelNames)
    for cui in labelNames.index:
        abbrTocui[labelNames.loc[cui]['abbreviation']] = cui
    # Create label folders if not exist
    for dir in abbrTocui.values():
        if not os.path.exists(ouputText_dir + "/" + dir):
            os.makedirs(ouputText_dir + "/" + dir)

    # Copy labelNames to ouput folder
    saveLabelNames(ouputText_dir,labelNames)

    # Convert evaluation (manual) article labels to CUIs
    # Foreach annotated article
    for i in range(len(evalLabels)):
       # Get all manual classes of this article from CSV
       parts = evalLabels.loc[i,'Manual Class'].split(',')
       # Get the pmid of this article from CSV
       pmid = str(evalLabels.loc[i,'pmid'])
       # foreach manual label of this article
       for part in parts:
           # write the test in the specified folder named after the CUI of the label
           labelCUI = abbrTocui[part.strip()]
           file = codecs.open(ouputText_dir + "/" + labelCUI + '/' + pmid + ".txt", "w+", "utf-8")
           # Write empty files, just to keep the pmid
           # Find corresponding article text
           # index = evalText_dataset.filenames.index(pmid)
           # index = evalText_dataset
           # file.write(evalText_dataset.data[index])
           file.close()

print(datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'))
