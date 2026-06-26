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

- Step 5 - Data Quality

    - Find the Missing Values

        ```bash
        df.isnull().sum()
        ```
    - Output

        ```bash
        Time_spent_Alone             63
        Stage_fear                   73
        Social_event_attendance      62
        Going_outside                66
        Drained_after_socializing    52
        Friends_circle_size          77
        Post_frequency               65
        Personality                   0
        dtype: int64
        ```
    - Duplicate Records

        ```bash
        df.duplicated().sum()  # np.int64(388)
        ```

    - Unique Values

        ```bash
        for col in df.columns:
        print(col)
        print(df[col].nunique())
        print()
        ```
    - Target Distribution - This analysis determines whether the dataset is balanced or imbalanced.

        ```bash
        df["Personality"].value_counts()
        ```
    - Output

        ```bash
        Personality
        Extrovert    1491
        Introvert    1409
        Name: count, dtype: int64
        ```
- Step 6 - Statistical Summary

    - Numerical columns - Looking for min, max, mean, std

        ```bash
        df.describe()
        ```

    - Output

        ```bash
                    Time_spent_Alone	Social_event_attendance	    Going_outside	 Friends_circle_size	Post_frequency
        count	    2837.000000	        2838.000000	                2834.000000	     2823.000000	        2835.000000
        mean	       4.505816	           3.963354	                   3.000000	        6.268863	           3.564727
        std	           3.479192	           2.903827	                   2.247327	        4.289693	           2.926582
        min	           0.000000	           0.000000	                   0.000000	        0.000000	           0.000000
        25%	           2.000000	           2.000000	                   1.000000	        3.000000	           1.000000
        50%	           4.000000	           3.000000	                   3.000000	        5.000000	           3.000000
        75%	           8.000000	           6.000000	                   5.000000	       10.000000	           6.000000
        max	          11.000000	          10.000000	                   7.000000	       15.000000	          10.000000

        ```
- Step 7 - Outlier Detection

    - Boxplots

        ```bash
        numeric_cols = df.select_dtypes(
            include=np.number
        ).columns

        for col in numeric_cols:
            plt.figure(figsize=(6,2))
            sns.boxplot(
                x=df[col]
            )
            plt.title(col)
            plt.show()
        ```

    - IQR Method

        ```bash
        Q1 = df[numeric_cols].quantile(0.25)
        Q3 = df[numeric_cols].quantile(0.75)

        IQR = Q3 - Q1
        ```

    - Count outliers

        ```bash
        outliers = (
            (
                df[numeric_cols] <
                (Q1 - 1.5*IQR)
            ) |
            (
                df[numeric_cols] >
                (Q3 + 1.5*IQR)
            )
        )

        outliers.sum()

        ```

- Step 8 - Exploratory Data Analysis (EDA)

    - Personality Distribution

        ```bash
        sns.countplot(
            x="Personality",
            data=df
        )
        ```

    - Time Alone vs Personality - Introverts generally spend more time alone compared to extroverts.

        ```bash
        sns.boxplot(
            x="Personality",
            y="Time_spent_Alone",
            data=df
        )
        ```

    - Social Events vs Personality

        ```bash
        sns.boxplot(
            x="Personality",
            y="Social_event_attendance",
            data=df
        )
        ```

    - Friend Circle Size

        ```bash
        sns.boxplot(
            x="Personality",
            y="Friends_circle_size",
            data=df
        )
        ```

    - Stage Fear Analysis

        ```bash
        pd.crosstab(
            df["Stage_fear"],
            df["Personality"]
        )
        ```
- Step 9 - Correlation Analysis - Correlation analysis identifies relationships among variables and potential multicollinearity issues.

    - Encode temporary copy

        ```bash
        eda_df = df.copy()
        ```

    - Convert categories

        ```bash
        for col in eda_df.select_dtypes(
            include="object"
        ):
            eda_df[col] = LabelEncoder().fit_transform(
                eda_df[col]
            )
        ```

    - Correlation matrix

        ```bash
        corr = eda_df.corr()

        plt.figure(figsize=(10,8))

        sns.heatmap(
            corr,
            annot=True,
            cmap="coolwarm"
        )

        plt.show()
        ```

- Step 10 - Feature Engineering 

    - means creating new columns (features) from your existing data to help the machine learning model understand patterns more easily.

    - It is one of the most important and creative parts of any ML project. Good feature engineering often improves model accuracy more than changing the model itself.

    **What is Happening in This Code?**

    -  In here i'm creating two new features:

        - 1. Social_Activity_Score
            ```bash
            Pythondf["Social_Activity_Score"] = (
                df["Social_event_attendance"] +
                df["Going_outside"] +
                df["Post_frequency"]
            )
            ```

            - What it does:
            - It adds up 3 related columns that represent how socially active a person is.
            - → Higher score = More socially active person (likely Extrovert)

        - 2. Isolation_Index

            ```bash
            Pythondf["Isolation_Index"] = (
                df["Time_spent_Alone"] -
                df["Social_event_attendance"]
            )
            ```

            - What it does:
            - It measures the difference between time spent alone and social activity.
            - → Higher value = More isolated (likely Introvert)

