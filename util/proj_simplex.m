function w = proj_simplex(v,C);
%implementing simplex projection from 
% John C. Duchi, Shai Shalev-Shwartz, Yoram Singer, Tushar Chandra: 
% Efficient projections onto the l1-ball for learning in high dimensions.

n = length(v);
u = sort(v,'descend');

theta = 0;
for i = 1:n
    test = u(i)- 1/i*(sum(u(1:i))-C);
    if test < 0
        rho = i-1;
        %             disp(['rho is ' num2str(i-1)]);
        break
    end
    rho = n;
end
theta = 1/rho*(sum(u(1:rho))-C);
w = v - theta*ones(n,1);
w = w .* (w>0);
end
