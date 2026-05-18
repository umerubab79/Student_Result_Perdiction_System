import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("       STUDENT PERFORMANCE PREDICTION SYSTEM (ANN)")
print("=" * 60)

df = pd.read_csv('student-mat.csv')
print(f"\nDataset loaded: {len(df)} students")
print(f"Features: {df.shape[1] - 1}")
print(f"Target: G3 (Final Grade)")

df = df.drop(['G1', 'G2'], axis=1)
df['Performance'] = df['G3']
df = df.drop('G3', axis=1)

print(f"\nTarget variable (Performance) range: {df['Performance'].min()} - {df['Performance'].max()}")

categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
numerical_cols.remove('Performance')

print(f"\nCategorical features: {len(categorical_cols)}")
print(f"Numerical features: {len(numerical_cols)}")

df_encoded = df.copy()
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df_encoded[col] = le.fit_transform(df[col])
    label_encoders[col] = le

X = df_encoded.drop('Performance', axis=1)
y = df_encoded['Performance']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = MLPRegressor(
    hidden_layer_sizes=(128, 64, 32, 16),
    activation='relu',
    solver='adam',
    max_iter=500,
    early_stopping=True,
    validation_fraction=0.2,
    random_state=42,
    verbose=True
)

print("\n" + "=" * 60)
print("Training ANN Model...")
print("=" * 60)

model.fit(X_train_scaled, y_train)

print("\n" + "=" * 60)
print("MODEL EVALUATION")
print("=" * 60)

predictions = model.predict(X_test_scaled)
mae = mean_absolute_error(y_test, predictions)
mse = mean_squared_error(y_test, predictions)
rmse = np.sqrt(mse)

print(f"\nTest Mean Absolute Error: {mae:.2f} marks")
print(f"Test RMSE: {rmse:.2f} marks")
print(f"Test MSE: {mse:.2f}")

print("\n" + "=" * 60)
print("SAMPLE PREDICTIONS vs ACTUAL")
print("=" * 60)
print(f"{'Actual':<10} {'Predicted':<10} {'Difference':<10}")
print("-" * 30)
for i in range(min(10, len(predictions))):
    actual = y_test.iloc[i]
    pred = predictions[i]
    diff = actual - pred
    print(f"{actual:<10} {pred:<10.2f} {diff:<10.2f}")

print("\n" + "=" * 60)
print("Model Training Complete!")
print("=" * 60)
