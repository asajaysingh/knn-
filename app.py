from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# Friends data
X = [
    [4,0],  # A
    [3,1],  # B
    [0,4],  # C
    [1,3]   # D
]

y = [
    "positive",
    "positive",
    "negative",
    "negative"
]

# Create model
model = KNeighborsClassifier(n_neighbors=3)

# Train
model.fit(X, y)

# New review
new_review = [[2,1]]

# Predict
result = model.predict(new_review)

print(result[0])