import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import mlflow
import mlflow.sklearn

df = pd.read_csv("games_processed.csv")

TARGET = "rating"

X = df.drop(columns=[TARGET])
y = df[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

mlflow.set_experiment("Game Rating Model")

with mlflow.start_run():

    model = RandomForestRegressor(random_state=42)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    mse = mean_squared_error(y_test, preds)

    mlflow.log_metric("mse", mse)
    mlflow.sklearn.log_model(model, "model")

    print("Training selesai")
    print("MSE:", mse)

