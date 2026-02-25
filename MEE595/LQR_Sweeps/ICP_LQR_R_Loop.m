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
X_desired = [1;pi;0;0];
% Time parameters
dt = 0.1;
tf = 20;
tspan = 0:dt:tf;
% 
R_list = [1e-4 1e-3 1e-2 1e-1];

figure;

ax1 = subplot(2,2,1); hold(ax1,'on'); grid(ax1,'on'); title(ax1,'x(t)'); xlabel(ax1,'t'); ylabel(ax1,'x [m]');
ax2 = subplot(2,2,2); hold(ax2,'on'); grid(ax2,'on'); title(ax2,'\theta(t)'); xlabel(ax2,'t'); ylabel(ax2,'\theta [rad]');
ax3 = subplot(2,2,[3 4]); hold(ax3,'on'); grid(ax3,'on'); title(ax3,'u(t)'); xlabel(ax3,'t'); ylabel(ax3,'u [N]');

legtxt = strings(numel(R_list),1);

for kidx = 1:numel(R_list)
    R = R_list(kidx);

    % LQR gain for this R
    K = lqr(A,B,Q,R);

    % simulate
    [t,X] = ode45(@(t,X) icp_dynamics(m,M,l,g,d,X,K,X_desired), tspan, X0);

    % plot (hold on already enabled)
    plot(ax1,t,X(:,1));
    plot(ax2,t,X(:,2));
    plot(ax3,t,X(:,5));

    legtxt(kidx) = "R = " + num2str(R);
end

legend(ax1, legtxt, 'Location','best');
legend(ax2, legtxt, 'Location','best');
legend(ax3, legtxt, 'Location','best');
