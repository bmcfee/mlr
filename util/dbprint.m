function dbprint(l, varargin)

    global DEBUG;

    if DEBUG >= l
        display(sprintf(varargin{:}));
    end
end

