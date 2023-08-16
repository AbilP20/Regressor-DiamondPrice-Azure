## End To End ML Project

### created environment
```
conda create -p venv python==3.11
conda activate path/venv
```

### Install necessary libraries
```
pip install -r requirements.txt
```

### setup.py - used to make the folders into packages so that we can upload it on python and use it in any other programs just like nump, pandas
```
python setup.py install
```

### create src - which contains the entire lifecycle of ML program
```
create __init__.py in all folders which you want to be used as a package
```

### notebooks folder - contains all ipynb files where we do EDA, analyze etc. - no need for init.py here as we dont want it as package

### create exception.py for custom exception handling in src and logger.py to log info

### utils.py will contain any common utilities, variables or functions(like databases, reading mongodb or mysql etc or some common functions also) that will be used in any number of python files in the project.

