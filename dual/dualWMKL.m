function W = dualWMKL(alpha, Z, U, RHO, K)

    global PsiR;
    d = size(K, 1);
    m = length(alpha);

    nKernels = size(K,3);

    W = Z - U - K / RHO;
    for i = 1:m
        W = W + alpha(i) * PsiR{i} / RHO;
    end
%     for p = 1:nKernels
%         W(:,:,p) = W(:,:,p) - K(:,:,p) / RHO;
%         for i = 1:m
%             W(:,:,p) = W(:,:,p) + alpha(i) * PsiR{i}(:,:,p) / RHO;
%         end
%     end

end
