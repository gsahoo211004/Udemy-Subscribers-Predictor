import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def evaluate_model(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    r2 = r2_score(y_true, y_pred)
    print(f"MAE: {mae:.2f}, RMSE: {rmse:.2f}, R2: {r2:.2f}")
    return mae, rmse, r2

def plot_predictions(y_true, y_pred):
    plt.figure(figsize=(6,6))
    plt.scatter(y_true, y_pred, alpha=0.3)
    plt.xlabel("Actual Subscribers")
    plt.ylabel("Predicted Subscribers")
    lims = [0, np.percentile(np.concatenate([y_true,y_pred]), 99)]
    plt.xlim(lims); plt.ylim(lims)
    plt.plot(lims, lims, '--', color='k')
    plt.title("Actual vs Predicted")
    plt.show()
