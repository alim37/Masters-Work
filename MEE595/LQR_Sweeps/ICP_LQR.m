clear all;close all;clc

m = 1; %mass of the pendulum (kg)
M = 5; % Mass of the cart (kg)
l = 2; %length of the pendulum (m)
g = -10; % Acceleration due to gravity
d = 1;
% Initial Conditions
x = -3; theta = pi+0.1; x_dot = 0; theta_dot = 0;
X0 = [x; theta; x_dot; theta_dot; 0];  % add u
A = [0 0 1 0;
    0 0 0 1;
    0 -m*g/M -d/M 0;
    0 -(m+M)*g/(M*l) -d/(M*l) 0];

B = [0; 0; 1/M; 1/(M*l)];

Q = [1 0 0 0; 0 1 0 0; 0 0 10 0; 0 0 0 100];
R = 0.001;
K = lqr(A,B,Q,R);
X_desired = [1;pi;0;0];
% 
tspan = 0:0.1:25;
[t,X] = ode45(@(t,X) icp_dynamics(m,M,l,g,d,X,K,X_desired), tspan, X0);

%%
figure;
subplot(221);plot(t,X(:,1))
xlabel('Time [s]');ylabel('x [m]')
subplot(222);plot(t,X(:,2))
xlabel('Time [s]');ylabel('\theta [rad]')
subplot(2,2,[3,4]);plot(t,X(:,5))
xlabel('Time [s]');ylabel('u [N]')
grid on
