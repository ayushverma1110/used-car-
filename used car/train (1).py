import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor

from preprocess import preprocess_data

# Load data
df = pd.read_csv(r"E:\Projects\Used Car Price\data\used_cars.csv")
df = preprocess_data(df)

# Split features & target
X = df.drop('price', axis=1)
y = df['price']

# Identify categorical & numerical columns
cat_cols = X.select_dtypes(include='object').columns
num_cols = X.select_dtypes(exclude='object').columns

# Preprocessing
preprocessor = ColumnTransformer([
    ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols),
    ('num', 'passthrough', num_cols)
])

# Model
from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor(
    n_estimators=50,        # reduced
    max_depth=12,           # reduced
    min_samples_split=5,
    random_state=42,
    n_jobs=-1
)


# Pipeline
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('model', model)
])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
pipeline.fit(X_train, y_train)

# Save model
with open(r"E:\Projects\Used Car Price\model\car_price_model.pkl", "wb") as f:
    pickle.dump(pipeline, f)

print("✅ Model trained and saved successfully")
