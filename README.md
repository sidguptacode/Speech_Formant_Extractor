# Speech Formant Analysis

## Project description
A repository for all code related to speech processing with formant analysis. A Django server can also be found here, in the formant_extractor_server folder.

## Setup instructions
First, I recommend creating a Python virtual environment, and installing all of the dependencies in the adjacent requirements.txt file. You can do this by running the command 'pip install ___' in your terminal, while inside a Python virtual environment.

## Running the Jupyter Notebook 
Simply `cd` into the formant_extractor_notebook folder, and run 'jupyter notebook' in a terminal. From here you can open the 'Formant Analysis.ipynb' file in the Jupyter browser, and run the code segments through that interface.

## Running the Django server
`cd` into the formant_extractor_server folder, and run 'python runserver ./manage.py 0:8000' in a terminal. Now, the Python server will be listening for activity from a frontend.
