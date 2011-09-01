function r = regularizeMKLDOD(W, X, gradient)
%
% r = regularizeMKLDOD(W, X, gradient)
%
%

    [d,n,m] = size(X);

    if gradient
        r = zeros(m,m,d);
        for i = 1:m
            r(i,i,:) = diag(X(:,:,i));
            for j = (i+1):m
                r(i,j,:) = diag(X(:,:,i)) + diag(X(:,:,j));
            end
        end
    else
        r = 0;
        for i = 1:m
            r = r + squeeze(W(i,i,:))' * diag(X(:,:,i));
            for j = (i+1):m
                r = r + squeeze(W(i,j,:))' * (diag(X(:,:,i)) + diag(X(:,:,j)));
            end
        end
    end
end
