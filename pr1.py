# Apply LDA Algorithm on Iris Dataset and classify which species a given flower 
# belongs to. 
# Dataset Link:https://www.kaggle.com/datasets/uciml/iris
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

data = pd.read_csv("Iris.csv")

X = data.drop(['Id', 'Species'], axis=1)
y = data['Species']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearDiscriminantAnalysis()
model.fit(X_train, y_train)

print("Accuracy:", model.score(X_test, y_test))

sl = float(input("Enter Sepal Length: "))
sw = float(input("Enter Sepal Width: "))
pl = float(input("Enter Petal Length: "))
pw = float(input("Enter Petal Width: "))

flower = model.predict([[sl, sw, pl, pw]])

print("Predicted Species:", flower[0])