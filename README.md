<h3>Name:Danica Christa</h3>
<h3>212223240022</h3>
<h3>EX. NO.6</h3>
<h3>DATE:</h3>
<h1 align="center">Heart attack prediction using MLP</h1>

<h3>Aim:</h3>
To construct a Multi-Layer Perceptron to predict heart attack using Python

<h3>Algorithm:</h3>
Step 1: Import the required libraries...  
Step 2: Load the heart disease dataset...  
...  
Step 11: Plot the error convergence...

<h3>Program:</h3>

```python
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

data = pd.read_csv('C:/Users/admin/Documents/AIML/SEM 4/neural networks/heart.csv')
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

mlp = MLPClassifier(hidden_layer_sizes=(100, 100), max_iter=1000, random_state=42)
mlp.fit(X_train, y_train)

y_pred = mlp.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

plt.plot(mlp.loss_curve_)
plt.title("MLP Training Loss Convergence")
plt.xlabel("Iteration")
plt.ylabel("Training Loss")
plt.show()
```

<H3>Output:</H3>

![image](https://github.com/user-attachments/assets/25b08392-0c6c-4ea7-85cd-839e3ac0631f)

<H3>Results:</H3>
Thus, an ANN with MLP is constructed and trained to predict the heart attack using python.
