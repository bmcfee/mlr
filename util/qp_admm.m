function alpha = qp_admm(H,b,C);

x = b;
z = b;
u = b;
q = 1;
mu = 100;
mult = 2;

p = numel(b);
n = numel(b);
tol = 1e-4;
ptol = tol * sqrt(p);
dtol = tol * sqrt(n);
% f = zeros(10000,1);
iter = 0;
qchange = 1;
while 1
    iter = iter + 1;
    zold = z;
%     f(iter) = 0.5*x'*H*x + b'*x;
    if qchange == 1
        Hinv = (H + q*eye(size(H)))^-1; %recompute factor matrix if q rescaled
        qchange = 0;
    end
    
    %     x = (H + q*eye(size(H))) \ (q*(z-u)-b);
    x = Hinv * (q*(z-u)-b);
    
    z = (x+u).*((x+u)>0); % project onto nonneg orthant
    if sum(z) > C
        z = proj_simplex(x+u,C); %project onto simplex
    end
    
    u = u + x - z;
    
    pr = q * norm( z(:)-zold(:));
    dr = norm((x(:) - z(:)));
    
    if pr < ptol && dr < dtol,break,end
    
%     if mod(iter,5) == 0
        if pr/dr > mu
            qchange = 1;
            q = 1/mult * q;
            u = mult * u;
%             disp(['q = ' num2str(q)]);
        elseif dr/pr > mu
            qchange = 1;
            q = mult * q;
            u = 1/mult * u;
%             disp(['q = ' num2str(q)]);
            
        end
%     end
end
% disp(['Iterations: ' num2str(iter)]);
% disp(['PR: ' num2str(pr) 'DR: ' num2str(dr)])
% figure; plot(f(1:iter))
alpha = z;