import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Example dataset
data = {
    'Feature1': [1, 2, 3, 4, 5],
    'Feature2': [5, 4, 3, 2, 1],
    'Target': [2, 3, 4, 5, 6]
}
df = pd.DataFrame(data)

# Plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

x = df['Feature1']
y = df['Feature2']
z = df['Target']

ax.scatter(x, y, z, c='r', marker='o')

ax.set_xlabel('Feature 1')
ax.set_ylabel('Feature 2')
ax.set_zlabel('Target')

plt.show()
