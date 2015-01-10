
img = imread('MakeShading/output/shading_14.jpg');
figure;imshow(img);
grad = abs(gradient(im2double(img)));
grad_eq = histeq(grad);
% grad_eq = histeq(grad_eq);
show = uint8(round(grad_eq*255));
figure;imshow(show);



