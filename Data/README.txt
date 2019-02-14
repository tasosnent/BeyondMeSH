The files included in this folder:
- 	AD_abstract.csv	:		A CSV file with the initial dataset of articles anotated with AD descriptor (D000544) on 17/4/29018. 
								Includes the following collumns:
									pmid			:	The unique identifier of the article in PubMed
									abstractText 	:	The abstract of the article
									title 			:	The title of the article
-	AD_Cts_inT.csv		:	A CSV file with the weak labels for the articles included in the intitial dataset.  
							Includes the following collumns:
								_id				:	An id coming from the database where we store the results of MetaMap
								cui				:	The Concept Unique Identifier (CUI) in the UMLS for each disease sub-type (ci) used as a weak label for the article.
								label			:	The label corresponding to the CUI above.
								pmid			: 	The unique identifier of the article in PubMed
								title			:	The title of the article
-	AD_labelNames.csv	:	A CSV file providing the correspondence from CUI to abbreviation used in MA datasets
							Includes the following collumns:
								name			:	The Concept Unique Identifier (CUI) in the UMLS for each disease sub-type (ci) used as a weak label
								abbreviation	:	The abbreviation corresponding to each CUI
- 	AD_T.csv			:	A CSV file with all concepts recognized in an article by MetaMap
							Includes the following collumns:
								pmid			:	The unique identifier of the article in PubMed
								cuis			:	A list with the Concept Unique Identifiers (CUIs) of all concepts recognized in the specific article.
-	MA1_Data.csv		:	A CSV file with concencus manual annotations (MA) for the randomly selected MA1 dataset.
							Includes the following collumns:
								pmid			:	The unique identifier of the article in PubMed
								Manual Class	:	A lists with the abbreviations of the concsnsus labels manually assigned by the experts.
-	MA2_Data.csv		:	A CSV file with concencus manual annotations (MA) for the weak balanced MA2 dataset.
							Includes the following collumns:
								pmid			:	The unique identifier of the article in PubMed
								Manual Class	:	A lists with the abbreviations of the concsnsus labels manually assigned by the experts.
-	MA1_pmids.txt		: 	A list of the pmids of the articles included in MA1 dataset.
-	MA2_pmids.txt		: 	A list of the pmids of the articles included in MA2 dataset.