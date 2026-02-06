%/*******************************************************
% * Copyright (C) 2022 Ruixu Liu <liur05@udayton.edu>
% * 
% * This file is part of Visual Perception for Autonomy.
% * Approach for nonlinear enhancement (AINDANE)
% * MIT License
% *******************************************************/
% 
function [I,In_prime] = AINDANE_ALE(Image)
%% section 3.1 Adaptive Luminance Enhancement
% Implementation of the formula: In_prime = f(In,z)

%%
% get the image size
I_size = size(Image);
gray_or_color = I_size(3);
% if it is RGB image transfer it to gray-scale
if gray_or_color == 3
    Ir = double(Image(:,:,1));
    Ig = double(Image(:,:,2));
    Ib = double(Image(:,:,3));
    I = (76.245*Ir + 149.685*Ig + 29.07*Ib)/255; % NTSC Equation(1)
else
    I = double(Image);
end

%% Normalization
In = I/255; % Equation(2)

%% Compute the CDF
% histogram the gray scale value
y = histcounts(reshape(I,1,[]),255); 
total_pixles = sum(y);
% creat the cdf space
cdf_image = zeros(1,255); 
for i = 1:255
    sum_before = sum(y(1:i));
    % get the cdf
    cdf_image(i) = sum_before/total_pixles; 
end
% find the L value for cdf = 0.1
[~ , L] = min(abs(cdf_image-0.1)); 

%% Compute z
if L <= 50
    z = 0;
elseif L>150
    z = 1;
else
    z = (L-50)/100;
end

%% Compute In'
In_prime = 0.5*In.^(0.75*z +0.25)+0.2*(1-In)*(1-z)+0.5*In.^(2-z); % Equation(3)


