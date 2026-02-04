import numpy as np
import control
from control.matlab import * 
import scipy as sp
from scipy import integrate 
import matplotlib.pyplot as plt

def pendcart(x,t,m,M,L,g,d,uf):
    u = float(uf(x))

    Sx = np.sin(x[2])
    Cx = np.cos(x[2])
    D = m*L*L*(M+m*(1-Cx**2))

    dx = np.zeros(4)
    dx[0] = x[1]
    dx[1] = (1/D)*(-(m**2)*(L**2)*g*Cx*Sx + m*(L**2)*(m*L*(x[3]**2)*Sx - d*x[1])) + m*L*L*(1/D)*u
    dx[2] = x[3]
    dx[3] = (1/D)*((m+M)*m*g*L*Sx - m*L*Cx*(m*L*(x[3]**2)*Sx - d*x[1])) - m*L*Cx*(1/D)*u
    return dx

m=1; M=5; L=2; g=-10; d=1;

b = 1 
A = np.array([
    [0, 1, 0, 0],
    [0, -d/M, b*m*g/M, 0],
    [0, 0, 0, 1],
    [0, -b*d/(M*L), -b*(m+M)*g/(M*L), 0]
], dtype=float)

B = np.array([
    [0],
    [1/M],
    [0],
    [b/(M*L)]
], dtype=float)
C = np.array([[1, 0, 0, 0]], dtype=float)
D = np.array([[0]], dtype=float)

Q = np.eye(4) 
R = 0.0001 
K = lqr(A,B,Q,R)[0] 

tspan = np.arange(0,10,0.001)
x0 = np.array([-1,0,np.pi+0.1,0]) 
wr = np.array([1,0,np.pi,0]) 
u_law = lambda x: -K@(x-wr) 
x_lqr = sp.integrate.odeint(pendcart, x0, tspan, args=(m,M,L,g,d,u_law))


b = -1 
A = np.array([
    [0, 1, 0, 0],
    [0, -d/M, (b*m*g)/M, 0],
    [0, 0, 0, 1],
    [0, -(b*d)/(M*L), -(b*(m+M)*g)/(M*L), 0]
], dtype=float)
B = np.array([
    [0],
    [1/M],
    [0],
    [b/(M*L)]
], dtype=float)
Vd = np.eye(4) 
Vn = 1 

Kf, P, E = lqe(A, np.eye(4), C, Vd, Vn) 

Baug = np.concatenate((B, np.eye(4), np.zeros_like(B)), axis=1) 
Daug = np.zeros((1, Baug.shape[1]))
Daug[0,-1] = 1
sysC = ss(A, Baug, C, Daug) 
sysTruth = ss(A, Baug, np.eye(4), np.zeros((4, Baug.shape[1])))

BKf = np.concatenate((B, Kf.reshape(4,1)), axis=1) 
sysKF = ss(A - np.outer(Kf, C), BKf, np.eye(4), np.zeros_like(BKf))

dt = 0.01
t = np.arange(0, 50, dt)
uDIST = np.sqrt(Vd) @ np.random.randn(4, len(t)) 
uNOISE = np.sqrt(Vn) * np.random.randn(len(t)) 
u_impulse = np.zeros_like(t)
u_impulse[100] = 20/dt
u_impulse[1500] = -20/dt

uAUG = np.concatenate((u_impulse.reshape((1, len(t))), uDIST, uNOISE.reshape((1, len(t))))).T
y, t, _ = lsim(sysC, uAUG, t) 
xtrue, t, _ = lsim(sysTruth, uAUG, t) 
xhat, t, _ = lsim(sysKF, np.hstack((u_impulse.reshape(-1,1), y)), t)

y_noiseless = (C @ xtrue.T).T       
# y_measured  = y_noiseless + uNOISE.reshape(-1,1)  
y_measured = y

y_kf = (C @ xhat.T).T

plt.figure()
plt.plot(t, y_measured, color='0.7', linewidth=1.0, label='y (measured)')
plt.plot(t, y_noiseless, 'k', linewidth=2.0, label='y (no noise)')
plt.plot(t, y_kf, 'b--', linewidth=2.0, label='y (KF estimate)')
plt.xlabel('Time')
plt.ylabel('Measurement')
plt.legend()
plt.grid(True)

plt.figure()
plt.plot(t, xtrue[:,0], 'b',  label='x')
plt.plot(t, xhat[:,0],  'b--', label=r'$\hat{x}$')

plt.plot(t, xtrue[:,1], 'r',  label=r'$\dot{x}$')
plt.plot(t, xhat[:,1],  'r--', label=r'$\dot{\hat{x}}$')

plt.plot(t, xtrue[:,2], 'g',  label=r'$\theta$')
plt.plot(t, xhat[:,2],  'g--', label=r'$\hat{\theta}$')

plt.plot(t, xtrue[:,3], 'm',  label=r'$\dot{\theta}$')
plt.plot(t, xhat[:,3],  'm--', label=r'$\dot{\hat{\theta}}$')

plt.xlabel('Time')
plt.ylabel('State')
plt.legend()
plt.grid(True)

plt.show()
