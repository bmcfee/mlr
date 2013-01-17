function result = threshFull_mixed(A, lam)
%return argmin_V lam*||V||_2,1 + 0.5 ||V - A||_F^2
% no symmetry constraint on V

norm = sqrt(sum(A.^2));
result = bsxfun(@times, A, max(1 - (lam ./ norm),0));
	
end
