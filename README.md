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

- Open the jupyter notebook

```bash
jupyter notebook
```

## Explore the dataset

- Step 1 - import most commonly using libraries

    ```bash
    import pandas as pd
    import numpy as np
    ```

- Step 2 - Load the dataset

    ```bash
    df = pd.read_csv("../data/raw/personality_dataset.csv")
    ```

- Step 3 - View the sample of the dataset

    ```bash

    # using this only can view 5 data sample
    df.head()

    df.sample(10)
    ```

- Step 4 - Dataset inspection

- This step identifies numerical and categorical attributes and helps determine required preprocessing operations.

    - Shape of the dataset

        ```bash
        df.shape  # (2900, 8)
        ```

    - Column Names

        ```bash
        df.columns
        ```

    - Data types

        ```bash
        df.dtypes

        or

        df.info()
        ```


