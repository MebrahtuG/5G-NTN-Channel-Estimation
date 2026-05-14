function H_mmse = ntnMMSEEstimate(rxGrid, dmrsIndices, dmrsSymbols, ...
    carrier, snrLin, ntnParams)
% MMSE channel estimation using parametric LOS channel statistics.
%
% Model: h = h_bar + noise   (deterministic LOS → h_bar is the mean)
% MMSE estimator: H_mmse = R_hY * R_YY^{-1} * y_pilots
%
% For a pure LOS channel the channel covariance R_hh = 0 (no scatter),
% so the MMSE reduces to the Wiener filter based on the known steering
% vectors. Here we implement a practical approximation:
%   - Build the DM-RS observation matrix Y_p from received pilots
%   - Use the known LOS steering vectors as the MMSE prior
%   - Regularise with noise power: W = A * (A'*A + (1/snr)*I)^{-1} * A'
% Then interpolate to data positions.

K = carrier.NSizeGrid * 12;
L = carrier.SymbolsPerSlot;
Nfft = nrOFDMInfo(carrier).Nfft;
fs   = nrOFDMInfo(carrier).SampleRate;
cp   = nrOFDMInfo(carrier).CyclicPrefixLengths;

% Pilot positions
[rows, cols] = ind2sub([K, L], dmrsIndices);
Np = numel(dmrsIndices);

% Build steering matrix A: [Np x 1] (SISO, single path)
% Each entry = known H_perfect(k,l) at the pilot location
delaySamples = round(ntnParams.PropagationDelay * fs);
A = zeros(Np, 1);
for p = 1:Np
    k = rows(p) - 1;   % 0-based subcarrier index
    l = cols(p);
    t_sym = sum(Nfft + cp(1:l-1));
    A(p) = exp(1j*2*pi*ntnParams.DopplerShift*t_sym/fs) * ...
        exp(-1j*2*pi*k*delaySamples/Nfft);
end

% Received pilots
y_p = rxGrid(dmrsIndices);

% MMSE weight: scalar (single path)
% W = A' * (A*A' + (Np/snr)*I)^{-1}  →  scalar MMSE gain
R_yy = A * A' + (1/snrLin) * eye(Np);
alpha_mmse = A' * (R_yy \ y_p);  % complex scalar channel gain

% Reconstruct H_mmse over full grid using estimated gain
[lGrid, kGrid] = meshgrid(1:L, 1:K);
H_mmse = zeros(K, L);
t_all  = cumsum([0, Nfft + cp(1:end-1)]);
for l = 1:L
    for k = 1:K
        H_mmse(k, l) = alpha_mmse * ...
            exp(1j*2*pi*ntnParams.DopplerShift*t_all(l)/fs) * ...
            exp(-1j*2*pi*(k-1)*delaySamples/Nfft);
    end
end
% (lGrid, kGrid defined but used implicitly via loop — kept for clarity)
end