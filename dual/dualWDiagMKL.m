function W = dualWDiagMKL(alpha, Z, U, RHO, K)

    global PsiR;
    d = size(K, 1);
    m = length(alpha);

    nKernels = size(K,3);

    W = Z - U;
    for p = 1:nKernels
        W(:,p) = W(:,p) - diag(K(:,:,p)) / RHO;
        for i = 1:m
            W(:,p) = W(:,p) + alpha(i) * PsiR{i}(:,:,p) / RHO;
        end
    end

end
