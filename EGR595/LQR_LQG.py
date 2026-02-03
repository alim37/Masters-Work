'''
Linear Quadratic Regulator (LQR) & Linear Quadratic Gaussian (LQG) Filter on Inverted Pendulum
'''

import numpy as np
import control
from control import ss, lsim
import scipy as sp
import matplotlib.pyplot as plt

def pendcart(x,t,m,M,L,g,d,uf):
    u = uf(x) # evaluate anonymous function at x
    Sx = np.sin(x[2])
    Cx = np.cos(x[2])
    D = m*L*L*(M+m*(1-Cx**2))
    dx = np.zeros(4)
    dx[0] = x[1]
    dx[1] = (1/D)*(-(m**2)*(L**2)*g*Cx*Sx + m*(L**2)*(m*L*(x
    [3]**2)*Sx - d*x[1])) + m*L*L*(1/D)*u
    dx[2] = x[3]
    dx[3] = (1/D)*((m+M)*m*g*L*Sx - m*L*Cx*(m*L*(x[3]**2)*Sx
    - d*x[1])) - m*L*Cx*(1/D)*u;
    return dx


m=1; M=5; L=2; g=-10; d=1;
b = 1 # pendulum up (b=1)
A = np.array([[0,1,0,0], [0,-d/M,b*m*g/M,0], [0,0,0,1], [0,-
b*d/(M*L),-b*(m+M)*g/(M*L),0]])
B = np.array([0,1/M,0,b/(M*L)]).reshape((4,1))

Q = np.eye(4) # state cost, 4x4 identity matrix
R = 0.0001 # control cost
K = control.lqr(A,B,Q,R)[0]


tspan = np.arange(0,10,0.001)
x0 = np.array([-1,0,np.pi+0.1,0]) # Initial condition
wr = np.array([1,0,np.pi,0]) # Reference position
u = lambda x: -K@(x-wr) # Control law
x = sp.integrate.odeint(pendcart,x0,tspan,args=(m,M,L,g,d,u))


## LQG
Vd = np.eye(4) # disturbance covariance
Vn = 1 # noise covariance
# Build Kalman filter
Kf, P, E = control.lqe(A,np.eye(4),C,Vd,Vn)

Baug = np.concatenate((B, np.eye(4),np.zeros_like(B)),axis=1) # [u I*wd 0*wn]
Daug = np.array([0,0,0,0,0,1]) # D matrix passes noise through
sysC = ss(A,Baug,C,Daug) # Single-measurement system
# "True" system w/ full-state output, disturbance, no noise
sysTruth = ss(A,Baug,np.eye(4),np.zeros((4,Baug.shape[1])))
BKf = np.concatenate((B,np.atleast_2d(Kf).T),axis=1)
sysKF = ss(A-np.outer(Kf,C),BKf,np.eye(4),np.zeros_like(BKf))

## Estimate linearized system in down position: Gantry crane
dt = 0.01
t = np.arange(0,50,dt)
uDIST = np.sqrt(Vd) @ np.random.randn(4,len(t)) # random disturbance
uNOISE = np.sqrt(Vn) * np.random.randn(len(t)) # random noise
u = np.zeros_like(t)
u[100] = 20/dt # positive impulse
u[1500] = -20/dt # negative impulse
# input w/ disturbance and noise:
uAUG = np.concatenate((u.reshape((1,len(u))),uDIST,uNOISE.
reshape((1,len(uNOISE))))).T
y,t,_ = lsim(sysC,uAUG,t) # noisy measurement
xtrue,t,_ = lsim(sysTruth,uAUG,t) # true state
xhat,t,_ = lsim(sysKF,np.row_stack((u,y)).T,t) # estimate
