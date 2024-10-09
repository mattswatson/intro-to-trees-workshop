import numpy as np
from sklearn.inspection import DecisionBoundaryDisplay
import matplotlib.pyplot as plt

def plot_tree_boundaries(model, x_train, y_train, feature_names, target_names):
    # Parameters
    n_classes = len(np.unique(y_train))
    plot_colors = "rb"
    plot_step = 0.02

    # Plot the decision boundary
    g = DecisionBoundaryDisplay.from_estimator(
        model,
        x_train,
        cmap=plt.cm.RdYlBu,
        response_method="predict",
        xlabel=feature_names[0],
        ylabel=feature_names[1],
    )

    # Plot the training points
    for i, color in zip(range(n_classes), plot_colors):
        idx = np.where(y_train == i)[0]
        plt.scatter(
            x_train.iloc[idx, 0],
            x_train.iloc[idx, 1],
            c=color,
            label=target_names[i],
            cmap=plt.cm.RdYlBu,
            edgecolor="black",
            s=15
        )
        
    return g