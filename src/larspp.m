% There are three main ways to use this function:
%
% (1) ** beta = larspp(X, y) **
% 
% Finds the BETA that minimize norm(X*BETA-y,2) + lambda*norm(BETA,1)
% for various lambda, and for the given vector y.  The matrix X is N
% by p and the vector y is N by 1.  This function returns a matrix
% BETA, where each column of BETA is a solution along the
% regularization path, for decreasing values of lambda.
%
% Alternatively, we may consider it as finding the BETA that
% minimize norm(X*BETA-y,2) subject to the constraint that
% norm(BETA,1) <= t, for various t, and for the given vector y.
% Again, BETA is a matrix whose columns are each a solution for
% increasing values of t.
%
%
% (2) ** beta = larspp(X, y, method = 'lar',                        **
%     **               stop_type = 'none', stop_val = 0, spar = 1   **
%     **               whole_path = 1, least_sqrs = 1, verbose = 1, **
%     **               kernel = 'auto', precomputed = 0)            **
%
% This solves the same problem, but with different options.  The user
% may provide any of these optional parameters.  They are:
%
% - METHOD: may be any of the following ('lar' is default):
%     'lar':       least angle regression approximate solution
%     'lasso':     strict L1-regularized solution
%     'lasso_pos': same, but with BETA >= 0 constraint added
%
% - STOP_TYPE: may be any of the following ('none' is default):
%     'none':     no stop condition
%     'norm':     stop when norm(BETA,1) >= STOP_VAL
%     'lambda':   stop when lambda(BETA) <= STOP_VAL
%     'num_iter': stop after STOP_VAL iterations
%     'num_beta': stop after STOP_VAL non-zero BETA found
%
% - STOP_VAL: meaning indicated in STOP_TYPE description.
%  
% - SPAR: return a sparse matrix if true
%
% - WHOLE_PATH: return entire regression path if true, otherwise
%               only the target value, as defined by STOP_TYPE
%               and STOP_VAL
%
% - LEAST_SQRS: if true, return non-regularized least-squares BETA
%               given the subset selected by regularized solution.
%
% - VERBOSE: print various information.
%
% - KERNEL: whether to use the kernelized version ('auto' is default)
%     'auto':    automatically select based on dimensions
%     'no_kern': do not use kernel (best for p>N)
%     'kern':    use kernel (best for N>p)
%
% - PRECOMPUTED: set to 1 if X <- X'*X and y <- X'*y (0 by default)
% 
% (3) ** beta = larspp(X, Y, method = 'lar',                        **
%     **               stop_type = 'none', stop_val = 0, spar = 1   **
%     **               whole_path = 1, least_sqrs = 1, verbose = 1  **
%     **               kernel = 'auto', precomputed = 0)            **
%
% Solves the same problem, but for multiple right hand sides, where
% Y is a matrix of right hand sides (one per column of Y).  Here,
% only a single BETA is returned for each column of Y.  If no stop
% condition is given, then it returns the last BETA in the
% regularization path.  The WHOLE_PATH flag is always ignored,
% since only a single BETA is returned for each right-hand-side.
%
% History:
% 1/3/2007: First completed version - James Diebel
% 1/10/2007: Added support for kernelized LARS
%
% Copyright (c) 2006 Varun Ganapathi, James Diebel, Stanford University
%

% LARS++, Copyright (C) 2007 Varun Ganapathi, David Vickery, James
% Diebel, Stanford University
%
% This program is free software; you can redistribute it and/or
% modify it under the terms of the GNU General Public License as
% published by the Free Software Foundation; either version 2 of the
% License, or (at your option) any later version.
%
% This program is distributed in the hope that it will be useful, but
% WITHOUT ANY WARRANTY; without even the implied warranty of
% MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
% General Public License for more details.
%
% You should have received a copy of the GNU General Public License
% along with this program; if not, write to the Free Software
% Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA
% 02110-1301 USA.


function beta = larspp(X,Y,varargin)
%% define default call args
method = -1;
stop_type = -1;
stop_val = 0;
spar = 1;
whole_path = 1;
least_sqrs = 0;
verbose = 0;
kernel = -1;
precomputed = 0;

%% extract optional args
n = length(varargin);
if n >= 1
    if strcmp(varargin{1},'lasso') == 1; method = 0; end
end
if n >= 2
   if strcmp(varargin{2},'norm') == 1; stop_type = 0; end
   if strcmp(varargin{2},'lambda') == 1; stop_type = 1; end
   if strcmp(varargin{2},'num_iter') == 1; stop_type = 2; end
   if strcmp(varargin{2},'num_beta') == 1; stop_type = 3; end
end 
if n >= 3
    stop_val = varargin{3};
end
if n >= 4
    spar = varargin{4};
end
if n >= 5
    whole_path = varargin{5};
end
if n >= 6
    least_sqrs = varargin{6};
end
if n >= 7
    verbose = varargin{7};
end
if n >= 8
   if strcmp(varargin{8},'no_kern') == 1; kernel = 0; end
   if strcmp(varargin{8},'kern') == 1; kernel = 1; end
end 
if n >= 9
    precomputed = varargin{9};
end

%% Check for sparse X and Y
if issparse(Y)
    Y = full(Y);
end
if issparse(X)
    % Call the C++ library MEX routine for sparse X
    [indx,jndx,x_val] = find(X);
    indx = int32(indx)-1; jndx = int32(jndx)-1;
    nz = int32(length(x_val));
    
    if ~spar
        beta = mexlars(X, Y, method, stop_type, stop_val,...
            whole_path, least_sqrs, verbose, kernel, precomputed,...
            x_val,indx,jndx,nz);
    else
        [i,j,s,r] = mexlars(X, Y, method, stop_type, stop_val,...
            whole_path, least_sqrs, verbose, kernel, precomputed,...
            x_val,indx,jndx,nz);
        beta = sparse(i,j,s,size(X,2),r);
    end
else
    % Call the C++ library MEX routine for full X
    if ~spar
        beta = mexlars(X, Y, method, stop_type, stop_val,...
            whole_path, least_sqrs, verbose, kernel, precomputed);
    else
        [i,j,s,r] = mexlars(X, Y, method, stop_type, stop_val,...
            whole_path, least_sqrs, verbose, kernel, precomputed);
        beta = sparse(i,j,s,size(X,2),r);
    end
end