# Introvert-Extrovert-Predictor-With-AI-ML

## Folder structure

```bash
D:.
│   .gitignore
│   README.md
│   
└───Backend
    │   requirements.txt
    │   
    ├───app
    ├───data
    │   ├───processed
    │   └───raw
    │           personality_dataset.csv
    │           
    ├───model
    └───notebook
```

## Project setup

- Install python virtual environment

```bash
py -3.12 -m venv venv
```

- Then activate venv

```bash
venv\Scripts\activate
```

- Install essentials libraries and save it to the requirements.txt file

```bash
pip install pandas numpy scikit-learn fastapi uvicorn joblib jupyter
```

- First i downloaded the dataset and place it inside raw/ folder

- then create .gitignore file and put raw/ venv/ and other relevent files.