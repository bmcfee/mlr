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
        W(:,:,i)    = psd_sparse(W(:,:,i));
    end
end

