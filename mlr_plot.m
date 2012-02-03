function mlr_plot(X, Y, W, D)

    [d, n] = size(X);
    %%%
    % Color-coding
    CODES = {   'b.', 'r.', 'g.', 'c.', 'k.', 'm.', 'y.', ...
                'b+', 'r+', 'g+', 'c+', 'k+', 'm+', 'y+', ...
                'b^', 'r^', 'g^', 'c^', 'k^', 'm^', 'y^', ...
                'bx', 'rx', 'gx', 'cx', 'kx', 'mx', 'yx', ...
                'bo', 'ro', 'go', 'co', 'ko', 'mo', 'yo'};

    %%%
    % First, PCA-plot of X

    figure;

    z = sum(X,3);
    subplot(3,2,[1 3]), pcaplot(z, eye(d), Y, CODES), title('Native');
    subplot(3,2,[2 4]), pcaplot(X, W, Y, CODES), title('Learned');

    [vecs, vals] = eig(z * z');
    subplot(3,2,5), bar(sort(real(diag(vals)), 'descend')), title('X*X'' spectrum'), axis tight;

    if size(X,3) == 1
        [vecs, vals] = eig(W);
        vals = real(diag(vals));
    else
        if size(W,3) == 1
            vals = real(W(:));
        else
            vals = [];
            for i = 1:size(W,3)
                [vecs, vals2] = eig(W(:,:,i));
                vals = [vals ; real(diag(vals2))];
            end
        end
    end
    subplot(3,2,6), bar(sort(vals, 'descend')), title('W spectrum'), axis tight;

    if nargin < 4
        return;
    end

    %%%
    % Now show some diagnostics

    figure;
    subplot(2,1,1), plot(D.f), title('Objective');
    subplot(2,1,2), barh([D.time_SO, D.time_solver D.time_total]), ...
                    title('Time% in SO/solver/total');


function pcaplot(X, W, Y, CODES)
    if size(X,3) == 1
        A = X' * W * X;
    else
        A = 0;
        if size(W,3) == 1
            for i = 1:size(X,3)
                A = A + X(:,:,i)' * bsxfun(@times, W(:,i), X(:,:,i));
            end
        else
            for i = 1:size(X,3)
                A = A + X(:,:,i)' * W(:,:,i) * X(:,:,i);
            end
        end
    end
    [v,d] = eigs(A, 3);
    X2 = d.^0.5 * v';
    hold on;
    for y = 1:max(Y)
        z = Y == y;
        scatter3(X2(1, z), X2(2, z), X2(3,z), CODES{y});
    end

    axis equal;
