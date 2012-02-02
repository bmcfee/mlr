function [dPsi, M, SO_time] = cuttingPlaneFull(k, X, W, Ypos, Yneg, batchSize, SAMPLES, ClassScores)
%
% [dPsi, M, SO_time] = cuttingPlaneFull(k, X, W, Yp, Yn, batchSize, SAMPLES, ClassScores)
%
%   k           = k parameter for the SO
%   X           = d*n data matrix
%   W           = d*d PSD metric
%   Yp          = cell-array of relevant results for each point
%   Yn          = cell-array of irrelevant results for each point
%   batchSize   = number of points to use in the constraint batch
%   SAMPLES     = indices of valid points to include in the batch
%   ClassScores = structure for synthetic constraints
%
%   dPsi        = dPsi vector for this batch
%   M           = mean loss on this batch
%   SO_time     = time spent in separation oracle

    global SO PSI DISTANCE CPGRADIENT;

    [d,n,m] = size(X);
    D       = DISTANCE(W, X);

    M       = 0;
    S       = zeros(n);
    dIndex  = sub2ind([n n], 1:n, 1:n);

    SO_time = 0;

    if isempty(ClassScores)
        for i = 1:batchSize
            if i > length(SAMPLES)
                break;
            end
            i = SAMPLES(i);

            if isempty(Ypos{i})
                continue;
            end
            if isempty(Yneg)
                % Construct a negative set 
                Ynegative = setdiff((1:n)', [i ; Ypos{i}]);
            else
                Ynegative = Yneg{i};
            end
            SO_start        = tic();
                [yi, li]    =   SO(i, D, Ypos{i}, Ynegative, k);
            SO_time         = SO_time + toc(SO_start);

            M               = M + li /batchSize;
            snew            = PSI(i, yi', n, Ypos{i}, Ynegative);
            S(i,:)          = S(i,:) + snew';
            S(:,i)          = S(:,i) + snew;
            S(dIndex)       = S(dIndex) - snew';
        end
    else

        % Do it class-wise for efficiency
        for j = 1:length(ClassScores.classes)
            c       = ClassScores.classes(j);
            points  = find(ClassScores.Y == c);

            Yneg    = find(ClassScores.Yneg{j});
            yp      = ClassScores.Ypos{j};
            
            if length(points) <= 1
                continue;
            end
            for x = 1:length(points)
                i           = points(x);
                yp(i)       = 0;
                Ypos        = find(yp);
                SO_start    = tic();
                    [yi, li]    = SO(i, D, Ypos, Yneg, k);
                SO_time     = SO_time + toc(SO_start);

                M           = M + li /batchSize;

                snew        = PSI(i, yi', n, Ypos, Yneg);
                S(i,:)      = S(i,:) + snew';
                S(:,i)      = S(:,i) + snew;
                S(dIndex)   = S(dIndex) - snew';

                yp(i)       = 1;
            end
        end
    end

    dPsi    = CPGRADIENT(X, S, batchSize);

end
