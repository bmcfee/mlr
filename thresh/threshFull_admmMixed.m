function result = threshFull_admmMixed(R,lam);
% return argmin_V lam*||V||_2,1 + 0.5 ||V - A||_F^2
% symmetry constraint on V


   
    W = R;
    V = eye(length(R));
    U = zeros(size(R));
    q = 1;
    mu = 100;
    mult = 2;
    
    p = numel(R);
    n = numel(U);
    tol = 1e-4;
    ptol = tol * sqrt(p);
    dtol = tol * sqrt(n);
    
    iter = 0;
    while 1
        iter = iter + 1;
        Vold = V;
        W = threshFull_mixed(((1*R+q*(V-U))/(1+q)),lam/(1+q));
        V = feasible_symm(W+U);
        U = U + W - V;
        
        pr = q * norm( V(:)-Vold(:));
        dr = norm((W(:) - V(:)));
        
        if pr < ptol && dr < dtol,break,end
        
        if pr/dr > mu
            q = 1/mult * q;
            U = mult * U;
            %         disp(['q = ' num2str(q)]);
        elseif dr/pr > mu
            q = mult * q;
            U = 1/mult * U;
            %         disp(['q = ' num2str(q)]);
            
        end
    end
    %disp(['Iterations: ' num2str(iter)]);
    %disp(['PR: ' num2str(pr) 'DR: ' num2str(dr)])
    result = V;
end



function result = feasible_symm(A)
result = 0.5*(A+A');
end


