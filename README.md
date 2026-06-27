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

- Step 11 - Data Cleaning

    - Create a Cleaning Copy - Don't modify the original dataset directly.

        ```bash
        clean_df = df.copy()
        ```

    - Handle Duplicate Records

        ```bash
        duplicates = clean_df.duplicated().sum()

        print(f"Duplicate rows: {duplicates}")
        ```
    - If duplicates exists then remove duplicates

        ```bash
        clean_df = clean_df.drop_duplicates()

        # then again check 
        clean_df.duplicated().sum()
        ```

    - Handle Missing Values

        ```bash
        clean_df.isnull().sum()
        ```

    - Output

        ```bash
        Time_spent_Alone              61
        Stage_fear                    73
        Social_event_attendance       61
        Going_outside                 65
        Drained_after_socializing     51
        Friends_circle_size           75
        Post_frequency                63
        Personality                    0
        Social_Activity_Score        183
        Isolation_Index              121
        dtype: int64
        ```

    - Numerical Features - Fill missing values with the median

        ```bash
        from sklearn.impute import SimpleImputer
        numeric_columns = clean_df.select_dtypes(include=['int64', 'float64']).columns

        num_imputer = SimpleImputer(strategy="median")

        clean_df[numeric_columns] = num_imputer.fit_transform(
            clean_df[numeric_columns]
        )
        ```

    - Categorical Features - Fill missing values with the most frequent value

        ```bash
        categorical_columns = clean_df.select_dtypes(
            include=['object', 'string', 'category']
        ).columns

        cat_imputer = SimpleImputer(strategy="most_frequent")

        clean_df[categorical_columns] = cat_imputer.fit_transform(
            clean_df[categorical_columns]
        )
        ```

    - Then Verify Missing Values

        ```bash
        clean_df.isnull().sum()

        # expect 0 for all 
        ```
    - Verify Data Types

        ```bash
        clean_df.info()
        ```

    - Final Dataset Validation - Finally, verify that the cleaned dataset is ready for preprocessing.
    
        ```bash
        print("Shape:", clean_df.shape)

        print("Missing values:")
        print(clean_df.isnull().sum())

        print("Duplicate rows:", clean_df.duplicated().sum())
        ```
    - Output

        ```bash
        Shape: (2499, 10)
        Missing values:
        Time_spent_Alone             0
        Stage_fear                   0
        Social_event_attendance      0
        Going_outside                0
        Drained_after_socializing    0
        Friends_circle_size          0
        Post_frequency               0
        Personality                  0
        Social_Activity_Score        0
        Isolation_Index              0
        dtype: int64
        Duplicate rows: 0
        ```


- Step 12 - Data Preprocessing

    - The goal of preprocessing is to transform the cleaned dataset into a format that machine learning algorithms can understand and use effectively.

    - Prepare the cleaned dataset for machine learning algorithms

        ```bash
        Clean Dataset
            ↓
        Encode Categories
            ↓
        Split Features & Target
            ↓
        Train/Test Split
            ↓
        Scale Features (only when required)
            ↓
        Ready for Model Training
        ```

    - Create a Model Dataset

    - Even though clean_df is already clean, we don't preprocess it directly. instead of that we Create another copy

        ```bash
        model_df = clean_df.copy()
        ```

    - A separate copy of the cleaned dataset was created for machine learning preprocessing. This preserves the cleaned dataset while allowing transformations required for model training.

    - Inspect Data Types - to check Numeric columns, Object columns, Target column

        ```bash
        model_df.info()
        ```
    - Identify Feature Types

        ```bash
        categorical_columns = model_df.select_dtypes(include=["object", "string", "str"]).columns.tolist()

        numeric_columns = model_df.select_dtypes(include=["number"]).columns.tolist()

        print("Categorical Features:", categorical_columns)
        print("Numerical Features:", numeric_columns)
        ```

    - Output 

        ```bash
        Categorical Features: ['Stage_fear', 'Drained_after_socializing', 'Personality']

        Numerical Features: ['Time_spent_Alone', 'Social_event_attendance', 'Going_outside', 'Friends_circle_size', 'Post_frequency', 'Social_Activity_Score', 'Isolation_Index']
        ```

    - **Why do this?** Because Different preprocessing methods are applied to different data types.

    - For Example

        ```bash
        | Type               | Preprocessing       |
        | ------------------ | ------------------- |
        | Numeric            | Scaling (sometimes) |
        | Binary categorical | Mapping             |
        | Target             | Label Encoding      |

        ```

    - Encode Binary Features - Suppose your dataset contains:**Yes** and **No**, Machine learning models cannot process text values. So Convert them into numbers.

        ```bash
        binary_mapping = {
            "Yes": 1,
            "No": 0
        }

        binary_features = [
            "Stage_fear",
            "Drained_after_socializing"
        ]

        for column in binary_features:
            model_df[column] = model_df[column].map(binary_mapping)
        ```

    - **The reason not include "Personality" into binary feature or mapping is because Personality is not a feature. it is the target variable (label).**

    - Encode the Target Variable - target is **Introvert** and **Extrovert** So Convert it into numbers

        ```bash
        from sklearn.preprocessing import LabelEncoder

        target_encoder = LabelEncoder()

        model_df["Personality"] = target_encoder.fit_transform(
            model_df["Personality"]
        )
        ```
    - Inspect the encoding

        ```bash
        print(target_encoder.classes_)
        ```

    - Save the encoder for deployment.

        ```bash
        joblib.dump(target_encoder, "../model/target_encoder.pkl")
        ```

    - Verify Encoding

        ```bash
        model_df.head()

        # then

        model_df.info()
        ```

    - Separate Features and Target

    - Features : x and Target : y

        ```bash
        X = model_df.drop("Personality", axis=1)

        y = model_df["Personality"]
        ```

    - Check 

        ```bash
        print(X.head())
        print(y.head())
        ```

    - Verify the shapes

        ```bash
        print("Features:", X.shape)
        print("Target:", y.shape)
        ```

    - Train-Test Split - split the data into training and testing sets.

        ```bash
        from sklearn.model_selection import train_test_split

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.20,
            random_state=42,
            stratify=y
        )
        ```
    - Verify the split

        ```bash
        print("Training Features:", X_train.shape)
        print("Testing Features:", X_test.shape)

        print("Training Labels:", y_train.shape)
        print("Testing Labels:", y_test.shape)
        ```

    - Feature Scaling

    - Do all models require scaling?

        ```bash
        | Model               | Scaling Required?|
        | ------------------- | -----------------|
        | Logistic Regression |  Yes             |
        | Random Forest       |  No              |
        | XGBoost             |  No              |

        ```

    - Create scaled versions only for Logistic Regression

        ```bash
        from sklearn.preprocessing import StandardScaler

        scaler = StandardScaler()

        X_train_scaled = scaler.fit_transform(X_train)

        X_test_scaled = scaler.transform(X_test)
        ```

    - Save the scaler as well

        ```bash

        joblib.dump(scaler, "../model/scaler.pkl")

        ```

- Step 13 - Baseline Models

    - Train multiple models. (ex :- Logistic Regression, Random Forest, Gradient Boosting)

    - Why train multiple models?

    - We train multiple models to compare their performance and choose the best one for the problem, rather than blindly assuming one model will work well.

    - Main Reasons:

        - 1. Different models have different strengths
            - Some models are simple and easy to interpret (e.g., Logistic Regression).
            - Some are powerful at capturing complex patterns (e.g., Random Forest, XGBoost).
            - Some are more robust to noisy data.

        - 2. No single model is best for every dataset
            - Every dataset is different. What works well on one project may not work well on another.

        - 3. To find the best performance
            - By comparing models, we can select the one that gives the highest accuracy, AUC, or F1-score on our data.

    - A common comparison might look like:

    ```bash
    | Model                                     | Type     | Why include it?                                  |
    | ----------------------------------------- | -------- | ------------------------------------------------ |
    | Logistic Regression                       | Baseline | Simple, fast, interpretable                      |
    | Random Forest                             | Ensemble | Handles nonlinear relationships, robust to noise |
    | Gradient Boosting (or XGBoost if allowed) | Boosting | Often achieves higher predictive performance     |

    ```

    - **1. Baseline Model**

        - What it is: The simplest possible model.
        - Purpose: Acts as a reference point or "minimum score".
        - Example: Logistic Regression

        - **Simple Analogy**:
        - Imagine you are trying to guess whether a person is Introvert or Extrovert by just looking at one simple rule - for example, “If they spend more than 5 hours alone, they are Introvert, else Extrovert.”

        - This the **Baseline**. It’s a very basic and simple approach.

        - **Why we use it?**
        - If your advanced models are not much better than the baseline, then something is wrong with your data or features.

    - **2. Ensemble Model**

        - What it is: A model that combines many small models to make one strong model.
        - Main Idea: "Many heads are better than one."
        - Popular Example: Random Forest

        - **Simple Analogy**:
        Instead of trusting just one person’s opinion about whether someone is Introvert or Extrovert, you ask 100 different people (each looking at different behaviors like time spent alone, social events, friends circle, etc.) and take the majority vote.
        This is Random Forest — many small decision trees working together.

        - **Key Advantage**:
        - It reduces big mistakes and gives more stable and reliable predictions. It is less likely to get confused by noisy data (like someone who sometimes behaves like an introvert and sometimes like an extrovert).

    - **3. Boosting**

        - What it is: A special type of Ensemble where models are built one after another, learning from previous mistakes.
        - Popular Examples: XGBoost, LightGBM, Gradient Boosting

        - **Simple Analogy**:
        - Think of a group of students trying to identify Introverts and Extroverts:
        - The first student makes some wrong predictions.
        - The second student focuses especially on the cases where the first student was wrong.
        - The third student focuses on the remaining difficult cases.
        - Each new student learns from the previous students’ mistakes.

        - In the end, the whole group becomes very good at correctly identifying Introvert vs Extrovert.
        This is how Boosting (XGBoost) works.

        - **Key Advantage**:
        - It usually gives the highest accuracy, but it takes more time to train.