from yellowbrick.regressor import ResidualsPlot as rp
from sklearn.linear_model import LinearRegression as lr
import numpy as np
import matplotlib.pyplot as plt 
from scipy.stats import t
from scipy.stats import probplot as pp
import math

def durbinWatson(residual):
    numerator = sum(residual[1:][:] * residual[:-1][:])
    denom = sum(residual * residual) 
    r = numerator / denom
    return 2*(1-r)

tot = np.array([3389, 1101, 1131, 596, 896, 1767, 807, 1111, 645, 628, 1360, 652, 860, 500, 781,
        1070, 1754]).reshape(-1, 1)
gen = np.array([1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1]).reshape(-1, 1)
amt = np.array([7500, 1975, 3600, 675, 750, 2500, 350, 1500, 375, 1050, 3000, 450, 1750, 2000, 4500,
        1500, 3000]).reshape(-1, 1)
pr = np.array([220, 200, 205, 160, 185, 180, 154, 200, 137, 167, 180, 160, 135, 160, 180, 170, 180]).reshape(-1, 1)
diap = np.array([0, 0, 60, 60, 70, 60, 80, 70, 60, 60, 60, 64, 90, 60, 0, 90, 0]).reshape(-1, 1)
qrs = np.array([140, 100, 111, 120, 83, 80, 98, 93, 105, 74, 80, 60, 79, 80, 100, 120,
        129]).reshape(-1, 1)
X = np.concatenate((gen, amt, pr, diap, qrs), axis=1)
y = tot
sampleSize = np.size(X, 0)
dataDim = np.size(X, 1)
Z = np.concatenate((np.ones((sampleSize, 1)), X), axis=1)
z0 = np.array([1, 1, 1200, 140, 70, 85])


linear = lr()
model = rp(linear)
model.fit(X, y)
predicted = model.predict(X)
residual = predicted - y
df = sampleSize-dataDim-1
sampleCovar = residual.T.dot(residual) / df
tStats = t.ppf(0.025, df)
interLen = math.sqrt(sampleCovar*(1+z0.T.dot(np.linalg.inv(Z.T.dot(Z))).dot(z0)))
upper = predicted + tStats*interLen
lower = predicted - tStats*interLen
print upper
print "\n"
print lower

durbin_watson = durbinWatson(residual)
print "Durbin-Watson test stats is given by: "
print durbin_watson

plt.cla()
plt.figure(1)
# autocorrelation test of residual to "time"
sampleNumber = range(1, sampleSize+1)
plt.scatter(sampleNumber, residual)
plt.title("residual plot against sample number")
plt.xlabel("sample number")
plt.ylabel("residual")
plt.savefig("./time.png")

plt.figure(2)
# residual against product of predictors
indepen = amt * pr * qrs
plt.scatter(indepen, residual)
plt.title("residual against product of predictors")
plt.xlabel("product of predictors")
plt.ylabel("residual")
plt.savefig("./indep.png")

plt.figure(3)
# residual vs predicted value
plt.scatter(predicted, residual)
plt.title("residual against predicted values")
plt.xlabel("predicted values")
plt.ylabel("residual")
plt.savefig("./predicted.png")

plt.figure(4)
# Q-Q plot
pp(np.squeeze(residual), plot=plt)
plt.savefig("./qq.png")


