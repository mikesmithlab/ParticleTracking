import numpy as np
import matplotlib.pyplot as plt
import Generic.fitting as fitting


number_of_balls = np.array([1, 5, 10, 20, 30, 40, 50, 60, 70])
mass = np.array([0.036, 0.193, 0.388, 0.773, 1.161, 1.545, 1.935, 2.316, 2.707])


fitter = fitting.Fit('linear', number_of_balls, mass)
fitter.add_params()
fitter.fit()
fitter.fit_errors()

