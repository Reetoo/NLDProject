#include the csv file as required
import numpy as np
from scipy.integrate import odeint
from scipy.optimize import curve_fit
from numpy import genfromtxt
def f(y, t, a, b, g): # rhs for ODE solver
    S, I = y
    Sdot = -a * S * I
    Idot = (a - b) * S * I + (b - g - b * I) * I
    dydt = [Sdot, Idot]
    return dydt

def y(t, a, b, g, S0, I0): # solving the ODE
    y0 = [S0, I0]
    y = odeint(f, y0, t, args=(a, b, g)) # solver
    S = y[:, 0]
    I = y[:, 1]
    return I.ravel() # return solution for fitting

file = open('./blog.csv')
data = genfromtxt(file, delimiter=',', names=['month','rating'])
I_data = data['rating']/100 # scaling down to 0-1 range
data_t = range(len(I_data)) # time range

popt, cov = curve_fit(y, data_t, I_data, [.05, 0.02, 0.01, 0.99, 0.01]) # extract fit results
a_opt, b_opt, g_opt, S0_opt, I0_opt = popt

print("a = %g" % a_opt)
print("b = %g" % b_opt)
print("g = %g" % g_opt)
print("S0 = %g \n I0 = %g" % (S0_opt, I0_opt))

import matplotlib.pyplot as plt
t = np.linspace(0, len(I_data), 2000)
plt.plot(data_t, I_data, '.',
         t, y(t, a_opt, b_opt, g_opt, S0_opt, I0_opt), '-')
plt.gcf().set_size_inches(6, 4)
#plt.savefig('out.png', dpi=96) #to save the fit result
plt.show()
