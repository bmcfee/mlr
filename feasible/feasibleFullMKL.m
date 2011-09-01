function W = feasibleFullMKL(W)
%
% W = feasibleFullMKL(W)
%
% Projects a single d*d matrix onto the PSD cone
%
    m       = size(W,3);

    for i = 1:m
        [v,d]       = eig(0.5 * (W(:,:,i) + W(:,:,i)'));
        W(:,:,i)    = v * bsxfun(@times, max(real(diag(d)),0), v');
    end
end

