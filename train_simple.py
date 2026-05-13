import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
import pickle

# Generate synthetic data (simulating credit card features)
X = np.random.randn(10000, 28)
y = (X.sum(axis=1) > 0).astype(int)

model = DecisionTreeClassifier(max_depth=10)
model.fit(X, y)

# Save model
with open("/home/ec2-user/modelserve/models/model.pkl", "wb") as f:
    pickle.dump(model, f)
print("Model trained and saved!")