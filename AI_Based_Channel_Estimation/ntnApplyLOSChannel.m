function [rxWaveform, H_perfect, offset] = ntnApplyLOSChannel( ...
    txWaveform, carrier, ntnParams)
% Apply a deterministic LOS channel model:
%   1. Fractional Doppler frequency shift (fD = 5000 Hz)
%   2. Sample-domain integer propagation delay (tau = 5 us)
%
% The perfect channel response H_perfect is the known complex frequency
% response at every (subcarrier, OFDM-symbol) position of the resource grid.

waveInfo = nrOFDMInfo(carrier);
fs       = waveInfo.SampleRate;
Nfft     = waveInfo.Nfft;
nSamples = size(txWaveform, 1);

% --- Doppler phase rotation (per-sample) -------------------------
t = (0:nSamples-1).' / fs;
doppler_phase = exp(1j * 2*pi * ntnParams.DopplerShift * t);
rxWaveform_dop = txWaveform .* doppler_phase;

% --- Propagation delay (integer samples) -------------------------
delaySamples = round(ntnParams.PropagationDelay * fs);
rxWaveform   = [zeros(delaySamples, 1); rxWaveform_dop];
rxWaveform   = rxWaveform(1:nSamples + delaySamples);  % keep original length + guard

% Timing offset to remove (integer delay)
offset = delaySamples;

% --- Perfect channel response (frequency domain) -----------------
% For a LOS channel: H(k,l) = exp(j*2*pi*fD * t_l) * exp(-j*2*pi*k*tau/Nfft)
% where t_l = start time of OFDM symbol l, tau = delay in samples
K  = carrier.NSizeGrid * 12;   % subcarriers = 72
L  = carrier.SymbolsPerSlot;   % OFDM symbols = 14
cp_lengths = waveInfo.CyclicPrefixLengths;  % CP length per symbol

H_perfect = zeros(K, L);
t_sym = 0;
for l = 1:L
    % Phase due to delay (constant over symbol, linear in subcarrier)
    k_idx = (0:K-1).';
    phase_delay  = exp(-1j * 2*pi * k_idx * delaySamples / Nfft);

    % Phase due to Doppler (constant over subcarrier, varies per symbol)
    phase_doppler = exp(1j * 2*pi * ntnParams.DopplerShift * t_sym / fs);

    H_perfect(:, l) = phase_doppler * phase_delay;

    % Advance time to next symbol start
    t_sym = t_sym + Nfft + cp_lengths(l);
end
end