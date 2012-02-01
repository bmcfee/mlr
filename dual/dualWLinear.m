function W = dualWlinear(alpha, Z, U, RHO, K)

    global PsiR;
    d = size(K, 1);
    m = length(alpha);

    W = Z - U - eye(d) /(d * RHO);
    for i = 1:m
        W = W + alpha(i) * PsiR{i} / RHO;
    end

end
