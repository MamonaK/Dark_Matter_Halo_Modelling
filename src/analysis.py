# -*- coding: utf-8 -*-
"""
Created on Tue Jul 28 17:33:12 2026

@author: mamon
"""
import numpy as np


# --- Chi-squared ---
def chi2(y, y_fit, u):
    return np.sum((y - y_fit)**2 / u**2)

def chi2reduced(y, y_fit, u, nparams):
    return chi2(y, y_fit, u) / (len(y) - nparams)

"""These Chi square functions were taken from the University of Toronto PHY224 Python Tutorial"""


# --- Root Mean Square Error (RMSE)---

def rmse(y,y_fit):
    return np.sqrt(np.mean((y - y_fit)** 2))