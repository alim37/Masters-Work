%/*******************************************************
% * Copyright (C) 2022 Ruixu Liu <liur05@udayton.edu>
% * 
% * This file is part of Visual Perception for Autonomy.
% * Approach for nonlinear enhancement (AINDANE)
% * MIT License
% *******************************************************/
% 
function S_shift = AINDANE_CR(S, I_input, I, lambda)
%% section 3.3 Color Restortion
% Implementation of the formula: Sj = S* Ij/I * lambda
lambda_r = 0.99*lambda;
lambda_g = 0.99*lambda;
lambda_b = 0.99*lambda;

% get the image size
I_size = size(I_input);
gray_or_color = I_size(3);

if gray_or_color == 3
    Ir = double(I_input(:,:,1));
    Ig = double(I_input(:,:,2));
    Ib = double(I_input(:,:,3));
    Sr = S .* Ir ./ (I+0.0001) *lambda_r ;
    Sg = S .* Ig ./ (I+0.0001) *lambda_g ;
    Sb = S .* Ib ./ (I+0.0001) *lambda_b ;
    S_t(:,:,1) = Sr;
    S_t(:,:,2) = Sg;
    S_t(:,:,3) = Sb;
    S_shift = uint8(S_t);
else
    S_shift = uint8(S)*lambda;
end


