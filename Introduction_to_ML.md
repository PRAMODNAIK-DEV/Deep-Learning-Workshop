# 🌸 k-Nearest Neighbour (k-NN) Algorithm on Iris Dataset  

This project demonstrates how to implement the **k-Nearest Neighbour (k-NN) algorithm** on the famous **Iris dataset** using Python and scikit-learn.  

We will go step by step, so students can copy and run **chunk by chunk** instead of running the whole code at once.  

---

## Step 1: Setup — Virtual Environment & Installation

### Create Virtual Environment
```bash
# Create a virtual environment named "iris_env"
python -m venv venv
```

### Activate Virtual Environment
- **Windows (PowerShell):**
```bash
venv\Scripts\activate
```
### Install Required Library  
Make sure you have scikit-learn installed:  

```bash
pip install scikit-learn
```

---

## Step 2: Import Libraries

```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
```

---

## Step 3: Load the Iris Dataset  

```python
iris = load_iris()
X = iris.data          # Features: Sepal Length, Sepal Width, Petal Length, Petal Width
y = iris.target        # Target: 0 -> setosa, 1 -> versicolor, 2 -> virginica

print("Number of Samples:", len(X))
print("Target Labels:", y)
```

**Output:**
- The dataset has **150 rows and 4 features**.  
- Targets are encoded as numbers (0, 1, 2).  

---

## Step 4: Split into Training & Testing Data  

We split the data into **70% training** and **30% testing** using `train_test_split`:  

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)
```

- `test_size=0.3` → 30% data for testing, 70% for training.  
- `random_state=42` → ensures that every time we run the code, we get the same train and test split there won't be a shuffling. This is used to control the randomness of how the data is split. The number 42 is just a number — you could use 0, 1, 99, or any other integer. 

---

## Step 5: Create the k-NN Classifier  

We’ll use **k=3 neighbors**:  

```python
knn = KNeighborsClassifier(n_neighbors=3)
```

---

## Step 6: Train the Model  

```python
knn.fit(X_train, y_train)
```

---

## Step 7: Make Predictions  

```python
y_pred = knn.predict(X_test)
```

---

## Step 8: Evaluate the Model  

```python
print(f"\nModel Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%\n")
```

You should get accuracy around **95-97%**.  

---

## Step 9: Compare Predictions  

We print both correct and wrong predictions:  

```python
print("Prediction Results:")
for i in range(len(y_test)):
    actual = iris.target_names[y_test[i]]
    predicted = iris.target_names[y_pred[i]]
    status = "Correct" if y_test[i] == y_pred[i] else "Wrong"
    print(f"Sample {i + 1}: Predicted = {predicted}, Actual = {actual} -> {status}")
```

**Example Output:**  
```
Sample 1: Predicted = virginica, Actual = virginica -> Correct
Sample 2: Predicted = versicolor, Actual = versicolor -> Correct
Sample 3: Predicted = virginica, Actual = versicolor -> Wrong
...
```

---

## Summary  
👉 This demonstrates the **supervised learning workflow**: **Data → Split → Train → Predict → Evaluate**  

---