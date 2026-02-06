%/*******************************************************
% * Copyright (C) 2022 Ruixu Liu <liur05@udayton.edu>
% * 
% * This file is part of Visual Perception for Autonomy.
% * Approach for nonlinear enhancement (AINDANE)
% * MIT License
% *******************************************************/
% 
function S = AINDANE_ACE(I,I_prime, multi_scale)

S = zeros(size(I));
for i = 1: length(multi_scale)
    c = multi_scale(i);
    %% section 3.2 Adaptive Contrast Enhancement
    % Implementation of the formula: S = f(I_prime,E^p)
    %% 2D gaussian convolution
    gaussian_sigma = sqrt(c^2/2);
    G = fspecial('gaussian',c,gaussian_sigma); % Gaussian filter Equation(9)
    I_conv = imfilter(I,G); % Equation(10)

    %% obtain E
    sigma = std(I(:));
    if sigma <= 3
        p = 3;
    elseif sigma >= 10
        p = 1;
    else
        p = (27 - 2*sigma)/7;
    end

    gama = I_conv./(I+0.0001);% add a small number to avoide divide 0

    E = gama.^ p; % Equation(11)

    %% Compute S(x,y)
    S_out = 255* I_prime .^ E; % Equation(12)
    
    S = S + S_out * 1/length(multi_scale); % Equation(13)
end

