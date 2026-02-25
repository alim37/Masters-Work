clear; close all; clc

m0 = 1;     % pendulum mass
M0 = 5;     % cart mass
l  = 2;
g  = -10;
d  = 1;

Q = diag([1 1 10 100]);
%R = 1e-3;
R = 1.0 

% Base time settings (Sweeps 1-3)
dt = 0.1;
tf = 20;
tspan = 0:dt:tf;

% Initial condition: [x; theta; xdot; thetadot; u_store]
X0 = [-3; pi+0.1; 0; 0; 0];

% Helper to build A,B from (m,M)
AB = @(m,M) deal( ...
    [0 0 1 0;
     0 0 0 1;
     0 -m*g/M      -d/M     0;
     0 -(m+M)*g/(M*l)  -d/(M*l) 0], ...
    [0;0;1/M;1/(M*l)] );

outdir = "figs";
if ~exist(outdir, "dir"), mkdir(outdir); end
dpi = 300;

% Figure look
FIG_POS = [100 100 1600 1000];
FONT_SZ = 8;
LINE_W  = 1.6;

% Controller designed at nominal masses
[A0,B0] = AB(m0,M0);
K0 = lqr(A0,B0,Q,R);

%  SWEEP 1: Change reference (setpoint)
x_ref_list = [0 1 2 3];
theta_ref  = pi;

[fig1,ax1,ax2,ax3] = makeBigFigure("Sweep 1: Reference Change", FIG_POS);

leg = strings(numel(x_ref_list),1);

for i = 1:numel(x_ref_list)
    X_des = [x_ref_list(i); theta_ref; 0; 0];

    [t,X] = ode45(@(t,X) icp_dynamics(m0,M0,l,g,d,X,K0,X_des), tspan, X0);

    plot(ax1,t,X(:,1),'LineWidth',LINE_W);
    plot(ax2,t,X(:,2),'LineWidth',LINE_W);
    plot(ax3,t,X(:,5),'LineWidth',LINE_W);

    leg(i) = sprintf("x_{ref} = %.2f, \\theta_{ref}=\\pi", x_ref_list(i));
end

legend(ax1,leg,'Location','best');
legend(ax2,leg,'Location','best');
legend(ax3,leg,'Location','best');

applyPretty(fig1, FONT_SZ);
saveFig(fig1, fullfile(outdir,"sweep1_reference.png"), ...
    "Reference Change (x_{ref})", dpi);

%  SWEEP 2: Change mass (vary pendulum mass m)
m_list = [0.5 1.0 1.5 2.0];

[fig2,ax1,ax2,ax3] = makeBigFigure("Sweep 2: Mass Change", FIG_POS);

X_des = [1; pi; 0; 0];
leg = strings(numel(m_list),1);

for i = 1:numel(m_list)
    mi = m_list(i);

    [t,X] = ode45(@(t,X) icp_dynamics(mi,M0,l,g,d,X,K0,X_des), tspan, X0);

    plot(ax1,t,X(:,1),'LineWidth',LINE_W);
    plot(ax2,t,X(:,2),'LineWidth',LINE_W);
    plot(ax3,t,X(:,5),'LineWidth',LINE_W);

    leg(i) = sprintf("m = %.2f", mi);
end

legend(ax1,leg,'Location','best');
legend(ax2,leg,'Location','best');
legend(ax3,leg,'Location','best');

applyPretty(fig2, FONT_SZ);
saveFig(fig2, fullfile(outdir,"sweep2_mass_m.png"), ...
    "Mass Change (vary m, fixed R)", dpi);

%  SWEEP 3: Mass uncertainty (m_true = m0*(1+delta))
delta_list = [-0.3 -0.15 0 0.15 0.3];

[fig3,ax1,ax2,ax3] = makeBigFigure("Sweep 3: Mass Uncertainty", FIG_POS);

leg = strings(numel(delta_list),1);

for i = 1:numel(delta_list)
    delta = delta_list(i);
    m_true = m0*(1+delta);

    [t,X] = ode45(@(t,X) icp_dynamics(m_true,M0,l,g,d,X,K0,X_des), tspan, X0);

    plot(ax1,t,X(:,1),'LineWidth',LINE_W);
    plot(ax2,t,X(:,2),'LineWidth',LINE_W);
    plot(ax3,t,X(:,5),'LineWidth',LINE_W);

    leg(i) = sprintf("\\delta = %+0.2f  =>  m_{true}=%.2f", delta, m_true);
end

legend(ax1,leg,'Location','best');
legend(ax2,leg,'Location','best');
legend(ax3,leg,'Location','best');

applyPretty(fig3, FONT_SZ);
saveFig(fig3, fullfile(outdir,"sweep3_mass_uncertainty.png"), ...
    "Mass Uncertainty (m_{true}=m_0(1+\\delta))", dpi);


%  SWEEP 4: Noise + disturbance sweep over 100 seconds (RK4)
%  Correct control effort plotting: u_hist
% ============================================================

tf4  = 100;
dt4  = 0.05;
tspan4 = 0:dt4:tf4;

% ---- Noise/disturbance sweep lists (edit as you want) ----
sigma_x_list     = [0.005 0.010 0.015 0.020 0.030 0.040 0.050 0.060 0.070 0.080 0.090 0.100];
sigma_theta_list = [0.005 0.010 0.015 0.020 0.025 0.030 0.035 0.040 0.045 0.050 0.055 0.060];
sigma_w_list     = [0.05  0.10  0.20  0.30  0.40  0.50  0.70  0.90  1.10  1.30  1.60  2.00];

Nsweep = numel(sigma_x_list);

% If you want EXACTLY 12 like your friend's plot, keep as-is.
% If you want fewer, shorten the lists together.

% ---- Saturation options (pick one) ----
SATURATE_U = false;      % keep controller "pure" LQR command u = -Kx_meas
u_max = 50;

SATURATE_PLANT = false;   % recommended to avoid blow-ups from u+w under high noise
uPlant_max = 60;

% ---- Figure ----
[fig4,ax1,ax2,ax3] = makeBigFigure("Sweep 4: Noise + Disturbance sweep (100s, RK4)", FIG_POS);

leg = strings(Nsweep,1);

for j = 1:Nsweep

    sigma_x     = sigma_x_list(j);
    sigma_theta = sigma_theta_list(j);
    sigma_w     = sigma_w_list(j);

    % State history (4-state plant)
    X = zeros(numel(tspan4),4);
    X(1,:) = X0(1:4).';

    % Control command history (THIS is what you should plot)
    u_hist = zeros(numel(tspan4),1);

    for k = 1:numel(tspan4)-1

        x_true = X(k,:).';

        % measurement noise (mean 0)
        meas_noise = [sigma_x*randn;
                      sigma_theta*randn;
                      0;
                      0];
        x_meas = x_true + meas_noise;

        % controller command from noisy measurement
        u = -K0*(x_meas - X_des);

        % optional: saturate controller command
        if SATURATE_U
            u = max(min(u,u_max), -u_max);
        end

        % disturbance added to plant input
        w = sigma_w*randn;     % mean 0 (no bias)
        u_total = u + w;

        % recommended: saturate plant input to keep integration sane
        if SATURATE_PLANT
            u_total = max(min(u_total,uPlant_max), -uPlant_max);
        end

        % RK4 step with constant u_total
        f = @(x) icp_dynamics_uinput4(m0,M0,l,g,d,x,u_total);

        k1 = f(x_true);
        k2 = f(x_true + 0.5*dt4*k1);
        k3 = f(x_true + 0.5*dt4*k2);
        k4 = f(x_true + dt4*k3);

        x_next = x_true + (dt4/6)*(k1 + 2*k2 + 2*k3 + k4);

        X(k+1,:) = x_next.';
        u_hist(k) = u;  % store controller command (correct control effort)
    end

    % Plot each sweep run
    plot(ax1,tspan4,X(:,1),'LineWidth',LINE_W);
    plot(ax2,tspan4,X(:,2),'LineWidth',LINE_W);
    plot(ax3,tspan4,u_hist,'LineWidth',LINE_W);

    leg(j) = sprintf("run %d: \\sigma_x=%.3f, \\sigma_\\theta=%.3f, \\sigma_w=%.2f", ...
                     j, sigma_x, sigma_theta, sigma_w);
end

lgd = legend(ax1, leg, 'Location','eastoutside');
lgd.FontSize = 8;
lgd.Box = 'on';
lgd.ItemTokenSize = [10 8];

legend(ax2,'off');
legend(ax3,'off');

applyPretty(fig4, FONT_SZ);

saveFig(fig4, fullfile(outdir,"sweep4_noise_disturbance_sweep_RK4.png"), ...
    "Noise + Disturbance sweep, R = 1.0", dpi);

% Local helper functions
% function [fig,ax1,ax2,ax3] = makeBigFigure(bigtitle, figPos)
%     fig = figure('Name', bigtitle, 'Units','pixels', 'Position', figPos);
% 
%     tl = tiledlayout(2,2,'TileSpacing','compact','Padding','compact');
% 
%     ax1 = nexttile(tl,1); hold(ax1,'on'); grid(ax1,'on');
%     title(ax1,'x(t)'); xlabel(ax1,'t [s]'); ylabel(ax1,'x [m]');
% 
%     ax2 = nexttile(tl,2); hold(ax2,'on'); grid(ax2,'on');
%     title(ax2,'\theta(t)'); xlabel(ax2,'t [s]'); ylabel(ax2,'\theta [rad]');
% 
%     ax3 = nexttile(tl,[1 2]); hold(ax3,'on'); grid(ax3,'on');
%     title(ax3,'u(t)'); xlabel(ax3,'t [s]'); ylabel(ax3,'u [N]');
% 
%     if exist('sgtitle','file')
%         sgtitle(tl, bigtitle, 'FontWeight','bold');
%     else
%         annotation(fig,'textbox',[0 0.95 1 0.04], ...
%             'String', bigtitle, 'EdgeColor','none', ...
%             'HorizontalAlignment','center', 'FontWeight','bold');
%     end
% end

function [fig,ax1,ax2,ax3] = makeBigFigure(bigtitle, figPos)

    fig = figure('Name', bigtitle, 'Units','pixels', 'Position', figPos);

    % Use subplot to match the look in your screenshot
    ax1 = subplot(2,2,1); hold(ax1,'on'); grid(ax1,'on');
    title(ax1,'x(t)'); xlabel(ax1,'t [s]'); ylabel(ax1,'x [m]');

    ax2 = subplot(2,2,2); hold(ax2,'on'); grid(ax2,'on');
    title(ax2,'\theta(t)'); xlabel(ax2,'t [s]'); ylabel(ax2,'\theta [rad]');

    ax3 = subplot(2,2,[3 4]); hold(ax3,'on'); grid(ax3,'on');
    title(ax3,'u(t)'); xlabel(ax3,'t [s]'); ylabel(ax3,'u [N]');

    % Big title at the top (like your screenshot)
    sgtitle(bigtitle,'FontWeight','bold');
end

function applyPretty(fig, fontSz)
    set(findall(fig,'-property','FontSize'),'FontSize',fontSz);
end

function saveFig(fig, filepath, bigtitle, dpi)
    % ensure a top title exists (in case older MATLAB)
    if exist('sgtitle','file')
        sgtitle(fig, bigtitle, 'FontWeight','bold');
    else
        annotation(fig,'textbox',[0 0.95 1 0.04], ...
            'String', bigtitle, 'EdgeColor','none', ...
            'HorizontalAlignment','center', 'FontWeight','bold');
    end

    drawnow;
    if exist('exportgraphics','file')
        exportgraphics(fig, filepath, 'Resolution', dpi);
    else
        print(fig, filepath, '-dpng', ['-r' num2str(dpi)]);
    end
end

function dx = icp_dynamics_uinput4(m,M,l,g,d,x,u_total)
    % x = [pos; theta; posdot; thetadot]
    dx = zeros(4,1);

    Sx = sin(x(2));
    Cx = cos(x(2));
    Kk = (M + m*(Sx^2));

    dx(1) = x(3);
    dx(2) = x(4);

    dx(3) = (1/Kk)*((-m*g*Sx*Cx) + (m*l*(x(4)^2)*Sx) - (d*x(3)) + u_total);
    dx(4) = (1/Kk)*((M+m)*g*Sx - m*l*x(4)^2*Sx*Cx + d*x(3)*Cx - Cx*u_total);
end