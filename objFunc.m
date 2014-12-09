function [ cost ] = objFunc( I )

global O;
global grad_O;
global clipped_O;
global pixelweight;

grad_I = gradient(I);

E = (grad_I - clipped_O).^2 + 0.1.*pixelweight(:).*((I-O).^2);
cost = sum(sum(E));

end

