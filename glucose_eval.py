import numpy as np
import scipy as sp
from matplotlib import pyplot as plt
import pandas as pd

class AnalyteLabError():
    def __init__(self, filtered, analyte, true_vals=2):
        self.filt = filtered
        self.analyte = analyte
        self.true_vals = true_vals

    def V_mus(self, return_list=False):
        ll = []
        # This if and else statement may seem redundant, but this is an adaptation.
        if len(self.filt) > 1:
            for fil in self.filt:
                ll.append(fil[self.analyte].dropna())
        else:
            for fil in self.filt:
                ll.append(fil[self.analyte].dropna())
        n = len(ll)
        mus = np.array(1/self.true_vals)*np.array([sum([ref.iloc[i] for i in range(self.true_vals)]) for ref in ll])
        V = np.array(1/(2*n)) * np.array(sum([(ref.iloc[i] - mu)**2 for i in range(self.true_vals) for ref,mu in zip(ll, mus)]))
        
        if return_list != False:
            return mus, V, ll
        return mus, V

    def add_error(self, V, index=0):
        mus, V, ll = self.V_mus(return_list=True)
        ref = self.filt[index]
        new_list = []
        for val in ref[self.analyte].dropna():
            new_val = np.random.normal(val, V)
            new_list.append(new_val)
        return new_list

    def plot_draws(self, outlier_multiplier=1.5, outliers= False, max_iters=10000, index=0):
        ll = []
        ref = self.filt[index]
        mus, V = self.V_mus(return_list=False)
        for _ in range(max_iters):
            new = self.add_error(V=np.sqrt(V), index=index)
            ll.append(new)
            plt.scatter(np.arange(0,len(ref[self.analyte].dropna())), new, color='blue', s=1, alpha=0.002)

        if outliers != False:
            data = np.array(ll)
            num_cols = data.shape[1]
            conf_intervals = []
            num_outliers = []

            for col_idx in range(num_cols):
                #lower, upper = confidence_interval(data[:, col_idx], means[col_idx], interval_percentage)
                #conf_intervals.append((lower, upper))

                # Calculate outliers with 95% confidence
                iqr = np.percentile(data[:, col_idx], 75) - np.percentile(data[:, col_idx], 25)
                lower_bound = np.percentile(data[:, col_idx], 25) - outlier_multiplier * iqr
                upper_bound = np.percentile(data[:, col_idx], 75) + outlier_multiplier * iqr
                outliers = np.sum((data[:, col_idx] < lower_bound) | (data[:, col_idx] > upper_bound))
                num_outliers.append(outliers)


            print("Number of outliers in 95 percent:")
            for col_idx, outliers in enumerate(num_outliers):
                print(f"Glucose Value {col_idx + 1}: {outliers} out of 10000")

        plt.scatter(np.arange(0,len(ref[self.analyte].dropna())), ref[self.analyte].dropna(), color='blue', label=f'Date {index}')
        plt.title(f'Normal draws for {self.analyte} (Patient )')
        plt.legend()   
