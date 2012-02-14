function W = dualWDiagMKL(alpha, Z, U, RHO, K)

    global PsiR;
    d = size(K, 1);
    m = length(alpha);

    nKernels = size(K,3);

    W = Z - U;
    for p = 1:nKernels
        W(:,p) = W(:,p) - diag(K(:,:,p)) / RHO;
    end

    for i = 1:m
        W = W + alpha(i) * PsiR{i} / RHO;
    end

end
