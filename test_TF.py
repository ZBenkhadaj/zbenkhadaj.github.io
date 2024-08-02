import numpy as np
import matplotlib.pyplot as plt

print(" first time using Nano")
a = 2.3

def gaussienne(t):
	return np.exp(-a*t**2)

na = 1134   # parameter for dt, meaning a times smaller than the scale of the function's variable
nb = 1054  # parameter for Tmax, ...............bigger...........................................
nc = 8  # this is a special parameter to choose to what extent show the fourier transform, c times more than the scale of the fourier transform 
nk = int(nb*nc)

scale_time = a**(-0.5)
dt = scale_time/na
Tmax = scale_time*nb
N = int(Tmax/dt)+1

t = np.linspace(-Tmax/2, Tmax/2, N)
gauss_t = gaussienne(t) # this is centered around 0
TF_gauss_t = dt*np.fft.fft( np.fft.ifftshift(gauss_t) ) # this is not centered around 0 
freq = np.fft.fftfreq(N, dt) 

TF_gauss_t = np.fft.fftshift(TF_gauss_t) # in order to center around frequency 0
TF_gauss_t = np.abs(TF_gauss_t)  # taking the module of the complex function
freq = np.fft.fftshift(freq)

half = N//2
cst = np.pi/np.sqrt(np.log(10)*a)
plt.plot(cst*freq[half -nk: half +nk], np.sqrt(a/np.pi) * TF_gauss_t[half -nk: half +nk], ".", ms = 1, label =" fourier transform of une gaussienne ")
plt.xlabel("normalized frequency (so that at frequency 1, value is 0.1)")
plt.ylabel("normalized fourier transform (maxing at 1)")
plt.legend(loc= "best")
plt.show()

print(" 1/Tmax = ", a**0.5 / nb)

print(" N = ",N)
print(" dt = ",dt)
print(" N*dt = ", N*dt)
print(" Tmax = ",Tmax)
print(" df = ",freq[1] - freq[0])
print(" 1/Tmax = ",1/Tmax)
