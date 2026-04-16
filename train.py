import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

# TEMP fake data example (replace with yours if needed)
data = pd.DataFrame(
    {
        "age": [20, 25, 30, 35],
        "gender": [0, 1, 1, 0],
        "country": [1, 2, 3, 4],
        "highest_deg": [1, 2, 3, 4],
        "coding_exp": [1, 2, 3, 4],
        "title": [1, 2, 3, 4],
        "company_size": [1, 2, 3, 4],
        "salary": [50, 60, 70, 80],  # target
    }
)

X = data.drop("salary", axis=1)
y = data["salary"]

model = DecisionTreeRegressor()
model.fit(X, y)

joblib.dump(model, "salary_predict_model.pkl")

print("Model saved!")
