function D = distToFrom(n, Vto, Vfrom, Ito, Ifrom)
    Cross           = -2 * Vto' * Vfrom;
    Tonorm          = sum(Vto .^2, 1)';
    Fromnorm        = sum(Vfrom .^2, 1);

    D               = zeros(n);
    D(Ito,Ifrom)    = bsxfun(@plus, bsxfun(@plus, Cross, Tonorm), Fromnorm);
end
