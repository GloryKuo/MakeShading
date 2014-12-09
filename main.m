global O;
global grad_O;
global clipped_O;
global pixelweight;


img = imread('input2.png');
img = imresize(img, 0.1);
img = rgb2gray(img);
img_d = im2double(img);
O = img_d(:);

grad_O = abs(gradient(O));
clipped_O = zeros(size(O));
clipped_O = grad_O(grad_O < 0.5);

G = fspecial('gaussian', [100, 100]);
pixelweight = double(ones(size(O))) - abs(O - imfilter(O, G));

img_opt = fmincon(@objFunc, img_d(:), ones(96520), zeros(96520));
% opts=optimset('fminsearch');
% opts.Display = 'iter';
% opts.MaxFunEvals=1000;
% opts.MaxIter=10000000000;
% opts.TolFun=1e-1000000000000000;
% opts.TolX=1e-100000000000000000;
% result = fminsearch(@objFunc, img_d(:), opts);