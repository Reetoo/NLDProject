#include the csv file as required
import numpy as np
from scipy.integrate import odeint
from scipy.optimize import curve_fit
from numpy import genfromtxt

scale_factor1 = 80
scale_factor2 = 100
meme1 = 'chemistry cat'
meme2 = 'dwight'

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

file = open('./trendsData/gregSteve.csv') # replace with filename, as required
data = genfromtxt(file, delimiter=',', names=['month','meme1','meme2'])

I1_data = data['meme1']/scale_factor1 # scaling down to suitable range, NEED TO FIGURE THIS OUT
I2_data = data['meme2']/scale_factor2 # scaling down to suitable range, NEED TO FIGURE THIS OUT
data_t = range(len(I1_data)) # time range
#-------------------------meme 1------------------------------------------------
popt, cov = curve_fit(y, data_t, I1_data, [.05, 0.02, 0.01, 0.99, 0.01]) # extract fit results
a_opt, b_opt, g_opt, S0_opt, I0_opt= popt

print("a = %g" % a_opt)
print("b = %g" % b_opt)
print("g = %g" % g_opt)
print("S0 = %g \nI0 = %g" % (S0_opt, I0_opt))

import matplotlib.pyplot as plt
t = np.linspace(0, len(I1_data), 2000)
axes = plt.gca()
axes.set_ylim([0,1])

plt.xlabel("Days")
plt.ylabel("Search Volume Index")
plt.plot(data_t, I1_data, 'g.', label='Google Trends (%s)'%(meme1))
plt.plot(t, y(t, a_opt, b_opt, g_opt, S0_opt, I0_opt), 'g-', label='Viral model(%s)'%(meme1))
#plt.legend(loc='upper right')
#plt.gcf().set_size_inches(6, 4)
#plt.savefig('out.png', dpi=96) #to save the fit result
#plt.show()
#-------------------------meme 2------------------------------------------------
popt, cov = curve_fit(y, data_t, I2_data, [.05, 0.02, 0.01, 0.99, 0.01]) # extract fit results
a_opt, b_opt, g_opt, S0_opt, I0_opt= popt

print("a = %g" % a_opt)
print("b = %g" % b_opt)
print("g = %g" % g_opt)
print("S0 = %g \nI0 = %g" % (S0_opt, I0_opt))

t = np.linspace(0, len(I2_data), 2000)

#plt.xlabel("Days")
#plt.ylabel("Search Volume Index")
plt.plot(data_t, I2_data, 'r.', label='Google Trends (%s)'%(meme2))
plt.plot(t, y(t, a_opt, b_opt, g_opt, S0_opt, I0_opt), 'r-', label='Viral model(%s)'%(meme2))
plt.legend(loc='upper right')
plt.gcf().set_size_inches(8, 6)
#plt.savefig('out.png', dpi=96) #to save the fit result
plt.show()
