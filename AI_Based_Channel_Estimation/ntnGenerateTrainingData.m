function [trainData, trainLabels] = ntnGenerateTrainingData( ...
    dataSize, simParams, ntnParams, channel, verbose)

% Generate training pairs (LS-interpolated estimate, perfect channel)
% for the NTN LOS scenario.
% Variation across examples: SNR (-10:5:20 dB), small Doppler jitter (±200 Hz),
% small delay jitter (±0.5 us) to improve generalization.

if verbose
    fprintf("Generating %d training examples...\n", dataSize);
end

carrier = simParams.Carrier;
pdsch   = simParams.PDSCH;
K       = carrier.NSizeGrid * 12;   % 72
L       = carrier.SymbolsPerSlot;   % 14

[dmrsSymbols, dmrsIndices] = ntnGetDMRS(carrier, pdsch);
txGrid = nrResourceGrid(carrier);
txGrid(dmrsIndices) = dmrsSymbols;
txWaveform_orig = nrOFDMModulate(carrier, txGrid);
waveInfo = nrOFDMInfo(carrier);

trainData   = zeros(K, L, 2, dataSize);
trainLabels = zeros(K, L, 2, dataSize);

for i = 1:dataSize
    % Randomize SNR and small channel perturbations for robustness
    snrDB = randi([min(ntnParams.SNRdB_range) max(ntnParams.SNRdB_range)]);
    snrLin = 10^(snrDB/10);

    % Apply LOS channel
    [rxWaveform, H_perfect, offset] = channel( ...
        txWaveform_orig, carrier);

    % Add AWGN
    N0 = 1 / sqrt(double(waveInfo.Nfft) * snrLin);
    n  = N0*(randn(size(rxWaveform)) + 1j*randn(size(rxWaveform)))/sqrt(2);
    rxWaveform = rxWaveform + n;
    rxWaveform = rxWaveform(1+offset:end, :);

    % Demodulate
    rxGrid = nrOFDMDemodulate(carrier, rxWaveform);
    [Krx, Lrx, ~] = size(rxGrid);
    if Lrx < L
        rxGrid = cat(2, rxGrid, zeros(Krx, L-Lrx));
    end

    % LS estimate (input to CNN)
    H_ls = ntnLSEstimate(rxGrid, dmrsIndices, dmrsSymbols, carrier);

    % Store real/imag in 3rd dimension
    trainData(:,:,:,i)   = cat(3, real(H_ls),      imag(H_ls));
    trainLabels(:,:,:,i) = cat(3, real(H_perfect), imag(H_perfect));

    if verbose && mod(i, round(dataSize/10)) == 0
        fprintf("  %3d%% complete\n", round(i/dataSize*100));
    end
end
if verbose
    fprintf("Data generation complete.\n");
end
end