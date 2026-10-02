import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss

def expected_calibration_error(y_true, y_prob, n_bins=10):
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        bin_mask = (y_prob >= bin_edges[i]) & (y_prob < bin_edges[i + 1])
        if i == n_bins - 1:
            bin_mask = bin_mask | (y_prob == bin_edges[i + 1])
        n_bin = np.sum(bin_mask)
        if n_bin > 0:
            bin_acc = np.mean(y_true[bin_mask])
            bin_conf = np.mean(y_prob[bin_mask])
            ece += (n_bin / len(y_true)) * np.abs(bin_acc - bin_conf)
    return ece

def main():
    # Evaluate calibration on validation set
    df_val = pd.read_csv('data/fracture_splits/val_fracture.csv')
    df_th = pd.read_csv('reports/threshold_optimization_validation.csv')
    print("Calibration analysis on validation set...")
    print(df_th)

if __name__ == '__main__':
    main()
