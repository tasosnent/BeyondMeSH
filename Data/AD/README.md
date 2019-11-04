## Data for fine-grained semantic indexing for Alzheimer's Disease.

### Input datasets for the pipeline

The files included in this folder:
* **AD_abstract.csv**	:	A CSV file with the initial dataset of articles annotated with AD descriptor (D000544) on 17/4/2018. 
							Includes the following collumns:
								pmid			:	The unique identifier of the article in PubMed
								abstractText 	:	The abstract of the article
								title 			:	The title of the article
* **AD_Cts_inT.csv**	:	A CSV file with the weak labels for the articles included in the intitial dataset.  
							Includes the following collumns:
								_id				:	An id coming from the database where we store the results of MetaMap
								cui				:	The Concept Unique Identifier (CUI) in the UMLS for each disease sub-type (ci) used as a weak label for the article.
								label			:	The label corresponding to the CUI above.
								pmid			: 	The unique identifier of the article in PubMed
								title			:	The title of the article
* **AD_labelNames.csv**	:	A CSV file providing the correspondence from CUI to abbreviation used in MA datasets
							Includes the following collumns:
								name			:	The Concept Unique Identifier (CUI) in the UMLS for each disease sub-type (ci) used as a weak label
								abbreviation	:	The abbreviation corresponding to each CUI
* **AD_concepts.csv**	:	A CSV file with all concepts recognized in an article by MetaMap with corresponding occurrence frequency
								Includes the following collumns:
									pmid			:	The unique identifier of the article in PubMed
									cuis			:	A list with the Concept Unique Identifiers (CUIs) of all concepts recognized in the specific article and the corresponding occurrence frequency separated by ":". (e.g. ["C0205307:1","C0040649:3",...])
* **MA1_Data.csv**		:	A CSV file with concencus manual annotations (MA) for the randomly selected MA1 dataset.
							Includes the following collumns:
								pmid			:	The unique identifier of the article in PubMed
								Manual Class	:	A lists with the abbreviations of the concsnsus labels manually assigned by the experts.
* **MA2_Data.csv**		:	A CSV file with concencus manual annotations (MA) for the weak balanced MA2 dataset.
							Includes the following collumns:
								pmid			:	The unique identifier of the article in PubMed
								Manual Class	:	A lists with the abbreviations of the concsnsus labels manually assigned by the experts.
* **MA1_pmids.txt**		: 	A list of the pmids of the articles included in MA1 dataset.
* **MA2_pmids.txt**		: 	A list of the pmids of the articles included in MA2 dataset.

### Pre-processed data
The processed dataset for direct development of the fine-grained semantic indexing models are available:

* As a zipped file [here](https://owncloud.skel.iit.demokritos.gr/index.php/s/UkxtpqeCuZjsXld) (1.1 Gb) and 
* As unzipped folders for selective download of specific subfolders or files [here](https://owncloud.skel.iit.demokritos.gr/index.php/s/UKy3DZjTzuk8xUn) (2.7 Gb).
