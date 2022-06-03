#  A Big Data Approach in Sentiment Analysis of Tweets

##  User Manual

###  Cloning of this Repository from GitLab

* Using the Command Line Tool in your desired IDE, run:

		git clone https://projects.cs.nott.ac.uk/comp4103/2021-2022/team11.git

* This will allow the latest version of source code to be cloned into the workspace.

### Utilising Databrick for Code Execution
* Login into your dedicated Databrick portal using your Azure AD.

* Head to the "Workspace" section to import the cloned repository into Databricks. In this case, Databricks accepts the import of .zip files, therefore, you can directly have the cloned repository imported into Databricks.


#  Prerequisites

###  Installation of Git

#### Windows

* Go to this link:

		https://git-scm.com/download/win

* Select the version based on your machine's information.

* Extract the files followed by running of the installer.

#### macOS
* Please have Homebrew installed if you don't already have it by typing in this command into your command prompt:

		/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

* Next, please have this command executed within your desired command prompt window:

		$ brew install git

# Datasets

Several datasets have been uploaded into this repository:
* Original dataset to be used by Lexicon-based Sentiment Analysis (Sequential Computing Procedure).ipynb and Lexicon-based Sentiment Analysis (Distributed Computing Procedure).ipynb.

* Parquet datasets that have been pre-processed by the big data pre-processing techniques presented within this project. The inclusion of this dataset is to allow both code files, Machine Learning Models(Sequential Computing Procedure).ipynb and Machine Learning Models (Distributed Computing Procedure).ipynb to function in cases where the pre-processing techniques failed to be executed. 

# Suggested Flow of Running Source Code

* Lexicon-based Sentiment Analysis (Sequential Computing Procedure).ipynb or Lexicon-based Sentiment Analysis (Distributed Computing Procedure).ipynb that serves the purpose to pre-processing datasets, followed by conducting sentiment analysis on tweets.

* Machine Learning Models (Sequential Computing Procedure).ipynb or Machine Learning Models (Distributed Computing Procedure).ipynb to visualise the efficiency difference between both the approaches. 


#  Running of Source Code

* After having all the prerequisites done, you are now ready to run the imported source code.

* You can navigate around each of the .ipynb files to access the developed codes.

* Simply attach yourself onto a cluster, followed by clicking on "Run all cells" on the top navigation bar to execute all the cells in sequence. 

* The code execution will automatically start where a series of outputs will be presented.


# Resource Utilisation
* Within this project, execution of code files has been handled solely on Databricks, using resources provided by Databricks' mainframes rather than resources offered by your local machine.

#  Attribute

> All files included inside the lib folder are written in-house.

###  List of packages (Standard Libraries) that has been included into the project are as follow:

* pyspark
* collections
* pandas
* numpy
* re
* nltk
* sklearn
* string
* matplotlib
* seaborn
* langdetect
* itertools
* pathlib
* time

# References
#### Natural Language Toolkit: vader
Copyright (C) 2001-2022 NLTK Project

 Author: 
 * C.J. Hutto <Clayton.Hutto@gtri.gatech.edu>
* Ewan Klein <ewan@inf.ed.ac.uk> (modifications)
* Pierpaolo Pantone <24alsecondo@gmail.com> (modifications)
* George Berry <geb97@cornell.edu> (modifications)
* Malavika Suresh <malavika.suresh0794@gmail.com> (modifications)

URL: <https://www.nltk.org/>
For license information, see LICENSE.TXT

Modifications to the original VADER code have been made to integrate it into NLTK. These have involved changes to ensure Python 3 compatibility, and refactoring to achieve greater modularity.
