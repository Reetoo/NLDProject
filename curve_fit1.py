import numpy as np
from scipy.integrate import odeint
from scipy.optimize import curve_fit
import sys    

def f(y, t, a, b):
    return a*y**2 + b

def y(t, a, b, y0):
    """
    Solution to the ODE y'(t) = f(t,y,a,b) with initial condition y(0) = y0
    """
    y = odeint(f, y0, t, args=(a, b))
    return y.ravel()

data_y =[]
file = open('/Users/sankalpgaur/Desktop/Betu/Programming/MATLAB/multiTImeline_1.numbers')
for e_raw in file.read().split('\r\n'):
	e=float(e_raw); data_y.append(e)
data_t = range(len(data_y))
popt, cov = curve_fit(y, data_t, data_y, [-1.2, 0.1, 0])
a_opt, b_opt, y0_opt = popt

print("a = %g" % a_opt)
print("b = %g" % b_opt)
print("y0 = %g" % y0_opt)

import matplotlib.pyplot as plt
t = np.linspace(0, 10, 2000)
plt.plot(data_t, data_y, '.',
         t, y(t, a_opt, b_opt, y0_opt), '-')
plt.gcf().set_size_inches(6, 4)
plt.savefig('out.png', dpi=96)
plt.show()
