% ***********************************************************************************************
%   Gaussian Mixture Diffusion Model (GMDM) used to simulate surface subsidence in mining areas.
%   Programmed by YANSUO ZHANG at IEGO (CSIC-UCM), Spain
% ***********************************************************************************************

function [result_GMDM] = Gaussian_Mixture_Diffusion_Model(samp_size)

[X1, X2] = meshgrid(1:samp_size, 1:samp_size);
X = [X1(:) X2(:)];

% Set the Gaussian component
num_components = randi([3, 7]);

% Set the central coordinate of the deformation field, i.e., the mean of the Gaussian distribution
mu_points = [];

rowLocationRange = [0.4, 0.6]; 
colLocationRange = [0.4, 0.6];
mu_points(1, :) = [(rowLocationRange(1) + (rowLocationRange(2) - rowLocationRange(1)) * rand(1,1))*samp_size, ...
                   (colLocationRange(1) + (colLocationRange(2) - colLocationRange(1)) * rand(1,1))*samp_size];

% Loop to generate the remaining points.
% Calculate the Euclidean distance between the new point and existing points.
% Add the new point only if all distances are less than the distance threshold!
while size(mu_points, 1) < num_components
    
    new_mu_point = [(rowLocationRange(1) + (rowLocationRange(2) - rowLocationRange(1)) * rand(1,1))*samp_size, ...
                    (colLocationRange(1) + (colLocationRange(2) - colLocationRange(1)) * rand(1,1))*samp_size];
    
    distances = sqrt(sum((mu_points - new_mu_point).^2, 2));    
    
    if all(distances < 45)
        mu_points = [mu_points; new_mu_point];
    end

end
mu = mat2cell(mu_points, ones(1, size(mu_points, 1)), size(mu_points, 2));

% Set the spatial extent of the deformation region, i.e., the covariance of the Gaussian distribution
N = 2;
for i = 1:num_components
    D = diag(rand(N, 1));
    U = orth(rand(N, N));
    sigma = U' * D * U;
    Sigma{i} = sigma * ((samp_size + samp_size) * 2);
    % Sigma{i} = sigma * samp_size * 2;
end

weights = rand(1, num_components);
gmm_weights = weights / sum(weights);

% Generate the initial mixture of Gaussian distributions
for i = 1:num_components
    p{i} = elli_gaus(X, mu{i}, Sigma{i});
end
p_mat = cell2mat(p);
gmm = gmm_weights * p_mat';
origin_gmm = reshape(gmm', samp_size, samp_size);
result_GMM_ori = origin_gmm / max(max(origin_gmm));

% Diffusion Process
gmm_mus = mu;
gmm_covs = Sigma;

% Sample initial points from the initial mixture of Gaussian distributions
num_samples = 1000;
x0 = sample_gmm(num_samples, num_components, gmm_weights, gmm_mus, gmm_covs);

% Set the diffusion coefficient and the number of diffusion steps
dif_coef = 5;
nsteps = 50;
dt = 1 / nsteps;

x_traj = zeros(size(x0, 1), size(x0, 2), nsteps);
x_traj(:, :, 1) = x0;

% Iterate to simulate the diffusion process
for i = 2:nsteps
    t = (i - 1) * dt;

    [gmm_covs] = diffusion_gmm(t, dif_coef, gmm_covs, gmm_weights);
    % gmm_mus_new = gmm_mus;
    gmm_covs_new = gmm_covs;
    x = sample_gmm(num_samples, num_components, gmm_weights, gmm_mus, gmm_covs);
    x_traj(:, :, i-1) = x;

    eps_z = randn(size(x0));
    x_traj(:, :, i) = x_traj(:, :, i-1) + eps_z * (dif_coef^t) * sqrt(dt);
end

% Generate the final mixture of Gaussian distributions
for i = 1:num_components
    p{i} = elli_gaus(X, gmm_mus{i}, gmm_covs_new{i});
end
p_mat = cell2mat(p);
end_gmm = gmm_weights * p_mat';
end_gmm = reshape(end_gmm', samp_size, samp_size);
result_GMDM = end_gmm / max(max(end_gmm));

% figure;
% subplot(1, 2, 1);
% scatter(x_traj(:, 1, 1), x_traj(:, 2, 1), 5, 'filled');
% title('Density of Target distribution $x_0$', 'Interpreter', 'latex');
% subplot(1, 2, 2);
% scatter(x_traj(:, 1, end), x_traj(:, 2, end), 5, 'filled');
% title(['Density of $x_T$ samples after ' num2str(nsteps) ' step diffusion'], 'Interpreter', 'latex');

% figure;
% hold on;
% for i = 1:size(x_traj, 1)
%     plot(squeeze(x_traj(i, 1, :)), squeeze(x_traj(i, 2, :)), 'Color', [0, 0, 1, 0.1]);
% end
% title('Diffusion trajectories', 'Interpreter', 'latex');
% hold off;

end


function x = sample_gmm(num_samples, num_components, gmm_weights, gmm_mus, gmm_covs)
x = [];
for i = 1:num_components
    n = round(num_samples * gmm_weights(i));
    x_i = mvnrnd(gmm_mus{i}, gmm_covs{i}, n);
    x = [x; x_i];
end
end


function [covs_dif] = diffusion_gmm(t, dif_coef, Sigma, gmm_weights)

num_components = length(gmm_weights);
lambda_t = (dif_coef^(2 * t) - 1) / (2 * log(dif_coef));
noise_cov = eye(2) .* lambda_t;
for i = 1:num_components
    % mus_dif{i} = mu{i} + sqrt(1 - lambda_t) * mu{i};
    covs_dif{i} = Sigma{i} + noise_cov;
end

end


% Multivariate Normal (Gaussian) Distribution
function [y, logSqrtDetSigma, quadform] = elli_gaus(X, Mu, Sigma)

if nargin<1
    error(message('stats:mvnpdf:TooFewInputs'));
elseif ndims(X)~=2
    error(message('stats:mvnpdf:InvalidData'));
end

% Get size of data.  Column vectors provisionally interpreted as multiple scalar data.
[n,d] = size(X);
if d<1
    error(message('stats:mvnpdf:TooFewDimensions'));
end

% Assume zero mean, data are already centered
if nargin < 2 || isempty(Mu)
    X0 = X;

% Get scalar mean, and use it to center data
elseif numel(Mu) == 1
    X0 = X - Mu;

% Get vector mean, and use it to center data
elseif ndims(Mu) == 2
    [n2,d2] = size(Mu);
    if d2 ~= d % has to have same number of coords as X
        error(message('stats:mvnpdf:ColSizeMismatch'));
    elseif (n2 == 1) || (n2 == n) % mean is a single row or a full vector.
        X0 = X - Mu;
    elseif n == 1 % data is a single row, rep it out to match mean
        n = n2;
        X0 = X - Mu;  
    else % sizes don't match
        error(message('stats:mvnpdf:RowSizeMismatch'));
    end
    
else
    error(message('stats:mvnpdf:BadMu'));
end

% Assume identity covariance, data are already standardized
if nargin < 3 || isempty(Sigma)
    % Special case: if Sigma isn't supplied, then interpret X
    % and Mu as row vectors if they were both column vectors
    if (d == 1) && (numel(X) > 1)
        X0 = X0';
        d = size(X0,2);
    end
    xRinv = X0;
    logSqrtDetSigma = 0;
    
% Single covariance matrix
elseif ndims(Sigma) == 2
    sz = size(Sigma);
    if sz(1)==1 && sz(2)>1
        % Just the diagonal of Sigma has been passed in.
        sz(1) = sz(2);
        sigmaIsDiag = true;
    else
        sigmaIsDiag = false;
    end
    
    % Special case: if Sigma is supplied, then use it to try to interpret
    % X and Mu as row vectors if they were both column vectors.
    if (d == 1) && (numel(X) > 1) && (sz(1) == n)
        X0 = X0';
        d = size(X0,2);
    end
    
    %Check that sigma is the right size
    if sz(1) ~= sz(2)
        error(message('stats:mvnpdf:BadCovariance'));
    elseif ~isequal(sz, [d d])
        error(message('stats:mvnpdf:CovSizeMismatch'));
    else
        if sigmaIsDiag
            if any(Sigma<=0)
                error(message('stats:mvnpdf:BadDiagSigma'));
            end
            R = sqrt(Sigma);
            xRinv = X0./R;
            logSqrtDetSigma = sum(log(R));
        else
            % Make sure Sigma is a valid covariance matrix
            [R,err] = cholcov(Sigma,0);
            if err ~= 0
                error(message('stats:mvnpdf:BadMatrixSigma'));
            end
            % Create array of standardized data, and compute log(sqrt(det(Sigma)))
            xRinv = X0 / R;
            logSqrtDetSigma = sum(log(diag(R)));
        end
    end
    
% Multiple covariance matrices
elseif ndims(Sigma) == 3
    
    sz = size(Sigma);
    if sz(1)==1 && sz(2)>1
        % Just the diagonal of Sigma has been passed in.
        sz(1) = sz(2);
        Sigma = reshape(Sigma,sz(2),sz(3))';
        sigmaIsDiag = true;
    else
        sigmaIsDiag = false;
    end

    % Special case: if Sigma is supplied, then use it to try to interpret
    % X and Mu as row vectors if they were both column vectors.
    if (d == 1) && (numel(X) > 1) && (sz(1) == n)
        X0 = X0';
        [n,d] = size(X0);
    end
    
    % Data and mean are a single row, rep them out to match covariance
    if n == 1 % already know size(Sigma,3) > 1
        n = sz(3);
        X0 = repmat(X0,n,1); % rep centered data out to match cov
    end

    % Make sure Sigma is the right size
    if sz(1) ~= sz(2)
        error(message('stats:mvnpdf:BadCovarianceMultiple'));
    elseif (sz(1) ~= d) || (sz(2) ~= d) % Sigma is a stack of dxd matrices
        error(message('stats:mvnpdf:CovSizeMismatchMultiple'));
    elseif sz(3) ~= n
        error(message('stats:mvnpdf:CovSizeMismatchPages'));
    else
        if sigmaIsDiag
            if any(any(Sigma<=0))
                error(message('stats:mvnpdf:BadDiagSigma'));
            end
            R = sqrt(Sigma);
            xRinv = X0./R;
            logSqrtDetSigma = sum(log(R),2);
        else
            % Create array of standardized data, and vector of log(sqrt(det(Sigma)))
            xRinv = zeros(n,d,'like',internal.stats.dominantType(X0,Sigma));
            logSqrtDetSigma = zeros(n,1,'like',Sigma);
            for i = 1:n
                % Make sure Sigma is a valid covariance matrix
                [R,err] = cholcov(Sigma(:,:,i),0);
                if err ~= 0
                    error(message('stats:mvnpdf:BadMatrixSigmaMultiple'));
                end
                xRinv(i,:) = X0(i,:) / R;
                logSqrtDetSigma(i) = sum(log(diag(R)));
            end
        end
    end
   
elseif ndims(Sigma) > 3
    error(message('stats:mvnpdf:BadCovariance'));
end

% The quadratic form is the inner products of the standardized data
quadform = sum(xRinv.^2, 2);

y = exp(-0.5*quadform - logSqrtDetSigma - d*log(2*pi)/2);

end
