# Project Onboarding Steps

Follow these steps to set up your development environment for the project.


## Step 1: Install Python 3.7

Python 3.7 is required for this project. Download and install it from the official Python website.

### For Windows:
Download from https://www.python.org/downloads/release/python-370/
Follow the installer steps

### For macOS and Linux:
Install Python 3.7 using Homebrew (macOS) or a package manager of your choice (Linux)
```
 brew install python@3.7
```

## Step 2: Install Jupyter Notebook

Jupyter Notebook is necessary for running and testing notebook files.
Install Jupyter Notebook using pip
```
    pip install jupyter
```

## Step 3: Install Pyenv

Pyenv is used to manage multiple Python versions.
Install pyenv. Visit https://github.com/pyenv/pyenv#installation for detailed installation instructions.

### For macOS:
```
    brew install pyenv
```

### For Linux:
```
    curl https://pyenv.run | bash
```

## Step 4: Create a Virtual Environment

Create a virtual environment to manage the project's dependencies separately.
Create a virtual environment using Python 3.7
```
    pyenv install 3.7.0
    pyenv virtualenv 3.7.0 chest_xray_env
```

## Step 5: Install Required Libraries

Activate your virtual environment and install the required libraries.
```
    pyenv activate chest_xray_env
    pip install -r requirements.txt
```
