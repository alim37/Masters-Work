function dX = icp_dynamics_disturb(m,M,l,g,d,X,K,X_desired,w)
% Adds an input disturbance w to the plant input.
% Controller is still u = -K*(X(1:4) - X_desired) (or noisy version in caller).
%
% State X = [x; theta; xdot; thetadot; u_store] (5x1)
% dX(5) stores the applied plant input (u_total)

u = -K*(X(1:4) - X_desired);   % controller output
u_total = u + w;               % plant sees disturbance-added input

dX = zeros(5,1);

Sx = sin(X(2));
Cx = cos(X(2));
Kk = (M + m*(Sx^2));

dX(1) = X(3);
dX(2) = X(4);

dX(3) = (1/Kk)*((-m*g*Sx*Cx) + (m*l*(X(4)^2)*Sx) - (d*X(3)) + u_total);
dX(4) = (1/Kk)*((M+m)*g*Sx - m*l*X(4)^2*Sx*Cx + d*X(3)*Cx - Cx*u_total);

dX(5) = u_total;   % store applied input (for reference)
end