import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Load dataset
df = pd.read_csv("california_housing.csv")

# Drop missing target values (house prices)
df = df.dropna(subset=["median_house_value"])

# Drop rows with missing features
df = df.dropna()

# Select features and target
X = df[["longitude", "latitude", "housing_median_age", "total_rooms",
        "total_bedrooms", "population", "households", "median_income", "ocean_proximity"]]
y = df["median_house_value"]

# Define column types
numeric_features = X.select_dtypes(include=["float64", "int64"]).columns.tolist()
categorical_features = ["ocean_proximity"]

# Preprocessing pipelines
numeric_transformer = SimpleImputer(strategy="mean")
categorical_transformer = OneHotEncoder(handle_unknown="ignore")

# Combine preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ]
)

# Create pipeline with model
pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", LinearRegression())
])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
pipeline.fit(X_train, y_train)

# Predict
y_pred = pipeline.predict(X_test)

# Evaluation
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("R-squared Score:", r2_score(y_test, y_pred))