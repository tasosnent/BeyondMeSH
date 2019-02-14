# BeyondMeSH
### A weakly supervised approach for fine-grained semantic indexing of biomedical literature

This repository includes the source code and some data for the development of models for fine-grained semantic indexing of biomedical articles.
In particular, it includes:
1. The folder **Data** with dataset files for the Alzheimer's Disease use case. These files are required to develop the weakly-supervised fine-grained semantic indexing models. For a detailed description of these files see the corresponding README file.
    * To avoid the data processing steps and directly develop the models, some pre-processed datasets are also available [here](https://owncloud.skel.iit.demokritos.gr/index.php/s/UkxtpqeCuZjsXld) (1.1 Gb) as a zipped file and [here](https://owncloud.skel.iit.demokritos.gr/index.php/s/UKy3DZjTzuk8xUn) (2.7 Gb) as unzipped folders for selective download of specific subfolders or files
2. The scripts **Pipeline.py** and **DatasetFunctions.py** for processing the initial dataset files to develop weakly supervised models for fine-grained semantic indexing. 
3. The **requirements.txt** file with the libraries and versions required for running the scripts.

## How to use

## Requirements
The script is written in Python 3.6
All libraries and versions required are listed in requirements.txt

Memory requirements: For big datasets, some steps of the pipeline can be very demanding in terms of memory. The experiments with the datasets provided in the "Data" folder have been executed in a system with more than 100Gb of memory. For less memory-demanding experiments please use smaller datasets.

### Configure
 Update specific configurations in the scripts:
 * In **DatasetFunctions.py** update:
    * (required) The **path_Separator** variable depending on your system. The default value is '/' for Unix systems.
    * (optional) Other variables (e.g. document_fn, label_fn, noClass) to customize the names of saved files etc. Details for each variable available in the corresponding comments.
 * In **Pipeline.py** update:
    * (required) The **baseFolder** variable. The value of this variable should be the absolute path to the folder with the datasets in your system. 
        * To run the experiment from the beginning use the path to the accompanying folder "Data". 
        * To skip the data processing steps, using the pre-processed datasets, and go directly to model development, use the path to folder "LexicalAndSemanticFeatures" or "LexicalFeaturesOnly".
    * (required) The **Steps to do** variables. These boolean variables define which steps of the process will be executed or skipped:
        * To run the experiment from the beginning update all these variables to have value 1.
        * To skip the data processing steps, using the pre-processed datasets, and go directly to model development update all these variables to have value 0, except from "classify" which should have value 1.
    * (required) The **useCUIS** variable. Needed only if step 3 "Analyze" is performed. To ignore semantic features set to 0. To consider semantic features set to 1. 
        * If you skip the data processing steps, using the pre-processed datasets, this variable has no effect. 
    * (required) The **test_folder_name** variable. Defines which test set will be used for evaluation. 
        * Set to 'MA1' to use the consensus manual annotations for the randomly selected MA1 dataset.        
        * Set to 'MA2' to use the consensus manual annotations for the weak balanced MA2 dataset.
        
### Run

Example call:

> python3.6 Pipeline.py

The results will be stored in this folder: 

> baseFolder\FinalSplit\AD

Where baseFolder is the absolute path provided in the configuration above.

* Execution of model development and evaluation step should result in the creation of corresponding CSV files with performance metrics for the models developed. The CSV files will be named **ScoresPerDataset_D.csv** and **socresPerDatasetPerLabel_D.csv**, where D can take values 'MA1' or 'MA2' depending on the dataset configured as testset (e.g. ScoresPerDataset_MA1.csv and socresPerDatasetPerLabel_MA1.csv). 
* Execution of data processing steps should result in the creation of corresponding intermediate files. (Like the pre-processed ones provided [here](https://owncloud.skel.iit.demokritos.gr/index.php/s/UKy3DZjTzuk8xUn)
