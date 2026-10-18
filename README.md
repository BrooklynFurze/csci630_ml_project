# CSCI 630 Small Scale Bird Photo ID

Project developers: Theo Furze

Abstract: The goal of this project is to recreate traditional computer vision-based bird
identification systems on a smaller scale and on lighter hardware. In the long term, 
having an accurate bird photo ID system that can run on simple hardware will be helpful in
tracking migration patterns and populations. This project will compare and contrast different
methods and fine tunings of different hyperparameters to attempt to create the most accurate
small-scale bird photo ID system.

Note: This project is using a small dataset from Kaggle, as opposed to the bird photo 
dataset published by Cornell, as the Cornell dataset was far too large.

How to run this project:
1. Clone this project from the Github repository. Note that the data folder does not have test
or train folders

2. Go to https://www.kaggle.com/datasets/klu2000030172/birds-image-dataset and download the dataset as a zip file. Unzip and place the test and train folders into the project such that
the file directory looks like this:

csci630_ml_project
|
| code 
| | main.py
|
| data
| | test
| | train
| | DATA.md
| | test.csv
| | train.csv
| 
| resources
| | placeholder.txt
|
| .gitignore
| README.md

This step is required to avoid pushing the 3000 raw training and test files to Github

3. Run main.py and view results in the terminal
