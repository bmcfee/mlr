function r = regularizeNone(W, X, gradient)
%
%   r = regularizeNone(W, X, gradient)
%   
%   Always returns 0

    if gradient    
    	    r = zeros(size(W));
    else
	    r = 0;
    end
end
