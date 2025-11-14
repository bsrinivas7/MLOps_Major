# test.py
import joblib
from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import train_test_split

data = fetch_olivetti_faces()
X_train, X_test, y_train, y_test = train_test_split(
    data.data, data.target, test_size=0.3, random_state=42
)
model = joblib.load('savedmodel.pth')
accuracy = model.score(X_test, y_test)
print(f"Test Accuracy: {accuracy}")
