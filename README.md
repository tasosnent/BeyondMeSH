# BeyondMeSH
### A weakly supervised approach for fine-grained semantic indexing of biomedical literature

This repository includes the source code and some data for the development of models for fine-grained semantic indexing of biomedical articles.
In particular, it includes:
1. The folder **Data** with dataset files for the Alzheimer's Disease (AD) and Duchenne Muscular Dystrophy (DMD) use cases. These files are required to develop the the datasets for weakly-supervised fine-grained semantic indexing models. For a detailed description of these files see the corresponding README file.
    * To avoid the data processing steps and directly develop the models, some pre-processed datasets are also available as a single zipped file and as separate files for selective download of specific sub-folders or files:
        * For the AD use case the zipped and unzipped pre-processed datasets are available [here](https://owncloud.skel.iit.demokritos.gr/index.php/s/AfEgLnVPoD2mDfO) (1.6 Gb) and [here](https://owncloud.skel.iit.demokritos.gr/index.php/s/mNQvA9kN9d0xItD) (4.43 Gb) respectively. 
        * For the DMD use case the zipped and unzipped pre-processed datasets are available [here](https://owncloud.skel.iit.demokritos.gr/index.php/s/cQvY1reNrFdYSZF) (125.6 Mb) and [here](https://owncloud.skel.iit.demokritos.gr/index.php/s/8LynUN0y3mJVgpW) (321 Mb) respectively. 
2. The scripts **Pipeline.py** and **DatasetFunctions.py** for processing the initial dataset files to develop weakly supervised models for fine-grained semantic indexing. 
3. The **requirements.txt** file with the libraries and versions required for running the scripts.

## How to use

## Requirements
The script is written in Python 3.6.

All libraries and versions required are listed in requirements.txt.

Memory requirements: For big datasets (e,g, for the AD use case), some steps of the pipeline (e.g. Feature Selection) can be very demanding in terms of memory. The experiments with the AD datasets provided in the "Data" folder have been executed in a system with more than 100Gb of memory. For less memory-demanding experiments please use smaller datasets (e.g. DMD or under-sampled AD).

### Configure
 Update specific configurations in the scripts:
 * In **DatasetFunctions.py** update:
    * (required) The **path_Separator** variable depending on your system. The default value is '/' for Unix systems.
    * (optional) Other variables (e.g. document_fn, label_fn, noClass) to customize the names of saved files etc. Details for each variable available in the corresponding comments.
 * In **Pipeline.py** update:
	* (required) The **Use-Case** variables. These variebles indicate on wich use case we are experimenting. For the reported AD and DMD experiments the values are already in the script, un-commenting the correct ones is enough.
		* *caseStudy*: An abbreviation for the use case (e.g. "AD" or "DMD"). Used for folder naming etc. 
		* *dominantLabel*: The UMLS CUI of the prefered concept for this uce case. Ignored in MA2 creation as well as in some baselines.
		* *cTop*: The UMLS CUI that is higher in the Ct Hierarcy for this uce case. Ignored for modeling and evaluation.
    * (required) The **baseFolder** variable. The value of this variable should be the path to the folder with the datasets in your system. 
        * To run the experiment from the beginning use the path to the accompanying folder "Data". 
        * To skip the data processing steps, using the pre-processed datasets, and go directly to model development, use the path to corresponding folder (e.g. "LexicalAndSemanticFeatures" etc).
    * (required) The **Steps-to-do** variables. These boolean variables define which steps of the process will be executed or skipped:
        * To run the experiment from the beginning update all these variables to have value 1.
        * To skip the data processing steps, using the pre-processed datasets, and go directly to model development update all these variables to have value 0, except from "classify" which should have value 1.
	* (required) The **Feature-Selection** variables. These variables define the alternative feature selection configurations to be considered. 
		* The default values correspond to the pre-processed files provided. If you add new values here, steps 3 and 5 must be excecuted before step 6 for modeling.
		* *featureKs*: Alternative Numbers of features to select in Feature selection.
		* *scoreFunctions*: Alternative score functions to be used for Univariate feature selection.
	* (optional) The **test_folder_names** variable. Defines which testsets will be used for evaluation. 
        * Set to ['MA1'] to use the consensus manual annotations for the randomly selected MA1 dataset.        
        * Set to ['MA2'] to use the consensus manual annotations for the weak balanced MA2 dataset.
        * The default is ['MA1','MA2'] to use both. 
	* (optional) Naming conventions variables (e.g. class_csv, final_dir, feature_prefix etc) to customize the names of saved files etc. Details for each variable available in the corresponding comments.
		* The default values correspond to the pre-processed files provided. 
	* (optional) **Step 1** variables: Configure dataset creation
		* The default values correspond to the pre-processed files provided. 
		* *manual_dataset1_pmids* and *manual_dataset2_pmids*: Path to files with the pmids of the MA1 and MA2 datasets to be removed from the training dataset creation. If no MA1 and MA2 datasets are available, set to False.
		* *manual_dataset_1_size* and *manual_dataset_1_size*: How many articles to select for creation of new MA1 and MA2 datasets. If MA1 and MA2 datasets are already selected set to 0.
		* *majority_articles_subsample_size*: Number of "preferred class (only)" articles to be removed from the trainig dataset for undesampling. If 0, no under-sampling is performed.
	* (optional) **Step 2 & 3** variables: Configure feature types considered
        * If you skip the data processing steps, using the pre-processed datasets, this variable has no effect.         
		* *useCUIS*: When *true* consider semantic features (concept occurrence), when *false* only lexical. 
		* *binaryFrequency*:  When *true* consider binary semantic features (concept occurs or not), when *false* use absolute frequency of concept occurrences. 
		* *ignoreLabelFeatures*:  When *true* exclude concepts used for weak supervision (*ci*) from the feature representation. 		
	* (optional) **Step 6** variables: Configure model training
		* The default values correspond to the pre-processed files provided.
		* *clfTypes*: A dictionary with alternative classifier types to be trained
		* *cvs*: A list with alternative numbers of folds for cross-validation on training dataset
		* *regCs*: A list with alternative values for Logistic Regression Classifier regularization levels (parameter C) 
		* *regType*: The type of regularization ("l2" or "l1") to be performed in Logistic Regression Classifier models     
		
### Run

Example call:

> python3.6 Pipeline.py

The results will be stored in this folder: 

> *baseFolder*\FinalSplit\\*UseCase*

Where *baseFolder* is the absolute path provided in the configuration above and *UseCase* is the abbreviation of the use case (i.e. AD or DMD). 

* Execution of model development and evaluation step should result in the creation of corresponding CSV files with performance metrics for the models developed. The CSV files will be named **ScoresPerDataset_*D*_*C*.csv** and **socresPerDatasetPerLabel_*D*_*C*.csv**, where *D* can take values 'MA1' or 'MA2' depending on the dataset configured as testset and *C* is the regularization level for Logistic Regression models (e.g. ScoresPerDataset_MA1_1.csv and socresPerDatasetPerLabel_MA1_1.csv).
    * For iteration experiments the corresponding CSV files will be named **ScoresPerDataset_*D*\_*C*\_*I*.csv** and **socresPerDatasetPerLabel_*D*\_*C*\_*I*.csv**, where *I* will be the number of corresponding iteration (i.e. ScoresPerDataset_MA1_1_0.csv for the first iteration, ScoresPerDataset_MA1_1_1.csv for the second one etc).
* Execution of data processing steps should result in the creation of corresponding intermediate files. (Like the pre-processed ones provided [here](https://owncloud.skel.iit.demokritos.gr/index.php/s/cQvY1reNrFdYSZF) and [here](https://owncloud.skel.iit.demokritos.gr/index.php/s/AfEgLnVPoD2mDfO) 
