from yellowbrick.regressor import ResidualsPlot as rp
from sklearn.linear_model import LinearRegression as lr
import numpy as np

tot = [3389, 1101, 1131, 596, 896, 1767, 807, 1111, 645, 628, 1360, 652, 860, 500, 781, 1070, 1754]
gen = [1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1]
amt = [7500, 1975, 3600, 675, 750, 2500, 350, 1500, 375, 1050, 3000, 450, 1750, 2000, 4500, 1500, 3000]
pr = [220, 200, 205, 160, 185, 180, 154, 200, 137, 167, 180, 160, 135, 160, 180, 170, 180]
diap = [0, 0, 60, 60, 70, 60, 80, 70, 60, 60, 60, 64, 90, 60, 0, 90, 0]
qrs = [140, 100, 111, 120, 83, 80, 98, 93, 105, 74, 80, 60, 79, 80, 100, 120, 129]

X = np.array([gen, amt, pr, diap, qrs]).reshape(-1, 5)
y = np.array(tot).reshape(-1, 1)

linear = lr()
visualizer = rp(linear)
visualizer.fit(X, y)
g = visualizer.proof()
