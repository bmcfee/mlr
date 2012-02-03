function W = feasibleFullMKL(W)
%
% W = feasibleFullMKL(W)
%
% Projects a single d*d matrix onto the PSD cone
%
    global FEASIBLE_COUNT;
    m       = size(W,3);
    FEASIBLE_COUNT = FEASIBLE_COUNT + m;

    parfor i = 1:m
        [v,d]       = eig(0.5 * (W(:,:,i) + W(:,:,i)'));
        W(:,:,i)    = v * bsxfun(@times, max(real(diag(d)),0), v');
    end
end

