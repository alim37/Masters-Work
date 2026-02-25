% function dX = icp_dynamics(m,M,l,g,d,X,u)
function dX = icp_dynamics(m,M,l,g,d,X,K,X_desired)

u = -K*(X(1:4) - X_desired);

dX = zeros(5,1);

Sx = sin(X(2));
Cx = cos(X(2));
Kk = (M+m*(Sx^2));

dX(1,1) = X(3);

dX(2,1) = X(4);

dX(3,1) = (1/Kk)*((-m*g*Sx*Cx)+(m*l*(X(4)^2)*Sx)-(d*X(3))+u);

dX(4,1) = (1/Kk)*((M+m)*g*Sx-m*l*X(4)^2*Sx*Cx+d*X(3)*Cx-Cx*u);

dX(5,1) = u;







% Sx = sin(X(3));
% Cx = cos(X(3));
% Kk = m*l*l*(M+m*(Sx^2));
% 
% dX(1,1) = X(2);
% 
% dX(2,1) = (1/Kk)*(-m^2*l^2*g*Cx*Sx+m*l^2*(m*l*X(4)^2*Sx-d*X(2))+m*l^2*u);
% 
% dX(3,1) = X(4);
% 
% dX(4,1) = (1/Kk)*((m+M)*m*g*l*Sx-m*l*Cx*(m*l*X(4)^2*Sx-d*X(2))-m*l*Cx*u)+.01*randn;

