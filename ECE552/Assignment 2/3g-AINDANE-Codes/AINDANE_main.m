%/*******************************************************
% * Copyright (C) 2022 Ruixu Liu <liur05@udayton.edu>
% * 
% * This file is part of Visual Perception for Autonomy.
% * Approach for nonlinear enhancement (AINDANE)
% * MIT License
% *******************************************************/
% 
clear;
close all;
clc;

%% Load Image
I_input = imread('image.bmp');

%% Section 3.1 Adaptive Luminance Enhancement
% Formula: I_prime = f(In,z)
[I_grayscale, I_prime] = AINDANE_ALE(I_input);


%% Section 3.2 Adaptive Contrast Enhancement
% Formula: S = f(I_prime,E^p)

% for different sclaes 
c1 = 5;
c2 = 20;
c3 = 120;


% compute S(x,y)
S = AINDANE_ACE(I_grayscale,I_prime,[c1,c2,c3]);
% S = AINDANE_ACE(I_grayscale,I_prime, c2);

%% section 3.3 Color Restortion
% Implementation of the formula: Sj = S* Ij/I * lambda 
lambda = 1;
S_f = AINDANE_CR(S, I_input, I_grayscale, lambda);

%% Quantitative Evaluation
I_zonal = I_grayscale(1:end,1:end); % choose a zonal size
S_zonal = S(1:end,1:end);
I_z_size = size(I_zonal);
I_z_pixel = I_z_size(1)*I_z_size(2); % compute the total zonal pixels 

original_mean = mean(I_grayscale(:));
o_m = mean(I_zonal(:));
original_variance = sum(sum((o_m - I_zonal).^2))/I_z_pixel;
original_sd = sqrt(original_variance);

enhancement_mean = mean(S(:));
e_m = mean(S_zonal(:));
enhancement_variance = sum(sum((e_m - S_zonal).^2))/I_z_pixel;
enhancement_sd = sqrt(enhancement_variance);

figure;
hold on
plot(original_sd,original_mean,'b+',enhancement_sd,enhancement_mean,'r*');
plot([original_sd,enhancement_sd],[original_mean,enhancement_mean],'y');
plot([0,91],[100,100],'--g');
plot([35,35],[0,251],'--g');
axis([0 90 0 250]);
text(5,170,'Insufficient Contrast');
text(5,85,'Insufficient Contrast');
text(8,75,'and Lightness');
text(50,60,'Insufficient Lightness');
text(38,230,'Vissually Optimal');
legend('original','enhancement');
title('Quantitative Evaluation');
xlabel('Mean of zonal standard deviation');
ylabel('Image mean');
hold off

%% using AHE to get the enhancement image 
 img_adjusted = zeros(size(I_input),'uint8');
 for ch=1:3
       img_adjusted(:,:,ch) = adapthisteq(I_input(:,:,ch));
 end

%% Visualization Result
figure;
sgtitle('Orignal Image')
imshow(I_input)

figure;
sgtitle('Enhanced by AHE')
imshow(img_adjusted)

figure;
sgtitle('Enhanced by AINDANE');
imshow(S_f)












