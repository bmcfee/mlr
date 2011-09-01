function Xi = lossHingeRCPFull(W, Psi, M, gradient)
%
%   Xi = lossHinge(W, Psi, M, gradient)
%   
%   W:          d*d metric
%   Psi:        1x2 cell:
%                   d*d feature matrix (actual Psi)
%                   d*d matrix L s.t. L'L = Psi
%   M:          the desired margin
%   gradient:   if 0, returns the loss value
%               if 1, returns the gradient of the loss WRT W
    
    global C;

    Xi = max(0, M - sum(sum(W .* Psi(:,:,1))));

    if gradient & Xi > 0
        %Xi = -Psi;

        %   CHANGED:2011-06-06 10:24:35 by Brian McFee <bmcfee@cs.ucsd.edu>
        %  draw a random rank-1 matrix according to the random conic projection rule 
        d = length(W);
        v = Psi(:,:,2) * randn(d,1);

        % divide the whole thing out by C
        Xi = -v * v' / C;
    end
end
