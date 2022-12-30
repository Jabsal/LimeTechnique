from lime import lime_tabular
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression

# Generate some synthetic data
X, y = make_regression(n_samples=1000, n_features=10)

# Train a linear regression model
model = LinearRegression()
model.fit(X, y)

# Define the local neighborhood around a specific instance
instance = X[0]
explainer = lime_tabular.LimeTabularExplainer(X, feature_names=['f1', 'f2', 'f3', 'f4', 'f5', 'f6', 'f7', 'f8', 'f9', 'f10'])

# Generate an explanation for the model's prediction on the instance
explanation = explainer.explain_instance(instance, model.predict, num_features=5)

# Print the explanation
print(explanation.as_list())
