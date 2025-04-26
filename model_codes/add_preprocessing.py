# === IMPORTS ===
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
import joblib

# === FUNCTIONS ===

def load_and_prepare_data(filepath, delimiter=';', reference_date="2018-12-31"):
    df = pd.read_csv(filepath, delimiter=delimiter)

    # Convert date columns
    date_cols = ['Date_start_contract', 'Date_last_renewal', 'Date_next_renewal', 
                 'Date_birth', 'Date_driving_licence', 'Date_lapse']
    for col in date_cols:
        df[col] = pd.to_datetime(df[col], errors='coerce', dayfirst=True)

    # Create target
    df['Renewed'] = df['Lapse'].apply(lambda x: 1 if x == 0 else 0)

    # Feature engineering
    reference_date = pd.to_datetime(reference_date)
    df['Customer_age'] = (reference_date - df['Date_birth']).dt.days // 365
    df['Driving_experience'] = (reference_date - df['Date_driving_licence']).dt.days // 365
    df['Contract_duration'] = (df['Date_next_renewal'] - df['Date_start_contract']).dt.days
    df['Experience_ratio'] = df['Driving_experience'] / (df['Customer_age'] + 1)

    # Filter outliers
    df = df[(df['Customer_age'] >= 18) & (df['Customer_age'] <= 100)]
    df = df[(df['Driving_experience'] >= 0) & (df['Driving_experience'] <= 80)]

    # Drop unused columns
    df = df.drop(columns=['ID', 'Date_start_contract', 'Date_last_renewal', 'Date_next_renewal', 
                          'Date_birth', 'Date_driving_licence', 'Date_lapse', 'Lapse'])

    # Define features and target
    X = df.drop(columns=['Renewed'])
    y = df['Renewed']

    return X, y

def create_preprocessor(X):
    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
    numerical_cols = X.select_dtypes(include=['float64', 'int64']).columns.tolist()

    # Ensure all categoricals are strings
    for col in categorical_cols:
        X[col] = X[col].astype(str)

    numeric_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])
    categorical_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer([
        ('num', numeric_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])

    return preprocessor