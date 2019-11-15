import numpy as np
from scipy.special import sici

def Si(t,Omega):
    return sici(Omega*t)[0]/np.pi

def sinc(t, Omega):
    return np.sinc(Omega/np.pi*t)*Omega/np.pi