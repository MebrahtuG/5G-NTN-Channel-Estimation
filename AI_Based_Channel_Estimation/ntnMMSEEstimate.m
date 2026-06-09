function H_mmse = ntnMMSEEstimate(rxGrid, dmrsIndices, dmrsSymbols, snrLin)

% ntnMMSEEstimate - Calculates the MMSE channel estimator.
% Syntax: H_mmse = ntnMMSEEstimate(rxGrid, dmrsIndices, dmrsSymbols, snrLin)

% Inputs:
%   rxGrid      - 2D Matrix of the received OFDM resource grid (Subcarriers x Symbols)
%   dmrsIndices - Linear indices or 1D vector pointing to DMRS locations in rxGrid
%   dmrsSymbols - Vector of the known transmitted pilot/DMRS symbols
%   snrLin      - Linear SNR value (assumed perfectly known)

% Output:
%   H_mmse      - Estimated channel matrix of the same size as rxGrid

    % 1. Initialization and dimensions
    [numSubcarriers, numSymbols] = size(rxGrid);
    numElements = numSubcarriers * numSymbols;
    
    % 2. Least Squares (LS) Estimation at DMRS locations
    % Extract received DMRS symbols
    rxDmrs = rxGrid(dmrsIndices);
    
    % LS Channel Estimate at pilot locations: H_ls = Y / X
    H_ls = rxDmrs ./ dmrsSymbols(:); 
    
    % 3. Define the Channel Autocorrelation Matrix (R_hh)
    % For a standard MMSE without explicit delay-profile knowledge, a 
    % robust choice is a uniform or exponential delay profile. Here we use 
    % a widely adopted sinc/exponential approximation for frequency-domain 
    % correlation, or construct a mock correlation based on pilot spacing.
    %
    % For a generalized solution, we can compute the sample covariance 
    % matrix or assume a normalized identity-based correlation if details 
    % about the power delay profile (PDP) are omitted.
    numPilots = length(dmrsIndices);
    
    % Assuming a normalized channel covariance matrix for the pilot positions
    % In a perfect setup, R_hh = E{H_pilot * H_pilot'}. 
    % We approximate it here using a standard exponential correlation model.
    [pIndX, pIndY] = ind2sub([numSubcarriers, numSymbols], dmrsIndices);
    R_hh_pilots = zeros(numPilots, numPilots);
    correlation_bandwidth = 0.1; % Normalized correlation parameter
    
    for i = 1:numPilots
        for j = 1:numPilots
            dist = abs(pIndX(i) - pIndX(j)) + abs(pIndY(i) - pIndY(j));
            R_hh_pilots(i,j) = exp(-correlation_bandwidth * dist);
        end
    end
    
    % Cross-correlation matrix between all grid elements and pilot elements (R_Hh)
    R_Hh = zeros(numElements, numPilots);
    [allX, allY] = ind2sub([numSubcarriers, numSymbols], (1:numElements)');
    
    for i = 1:numElements
        for j = 1:numPilots
            dist = abs(allX(i) - pIndX(j)) + abs(allY(i) - pIndY(j));
            R_Hh(i,j) = exp(-correlation_bandwidth * dist);
        end
    end

    % 4. Noise Variance
    % Since SNR = P_signal / P_noise, and assuming normalized signal power (1)
    sigma2 = 1 / snrLin;
    
    % 5. MMSE Estimation formula
    % H_mmse = R_Hh * inv(R_hh_pilots + sigma2 * I) * H_ls
    % Using matrix right-division (/) or left-division (\) for numerical stability
    W_mmse = R_Hh / (R_hh_pilots + sigma2 * eye(numPilots));
    H_mmse_vector = W_mmse * H_ls;
    
    % 6. Reshape the vector back into the OFDM grid dimensions
    H_mmse = reshape(H_mmse_vector, numSubcarriers, numSymbols);

end