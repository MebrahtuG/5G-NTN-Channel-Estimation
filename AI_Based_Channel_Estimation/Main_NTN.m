%% 5G NTN Deep Learning Channel Estimation
%  Scenario: Satellite-to-UE (NTN), PDSCH, SISO
%  Resource grid: 72 subcarriers (6 RBs x 12) x 14 OFDM symbols
%  Channel: Deterministic LOS (AWGN + Doppler + delay)
%  Estimators: LS (interpolated), MMSE (analytical), CNN
%  Output: MSE vs SNR comparison
 
%% =====================================================================
%  CONFIGURATION FLAGS
% ======================================================================
trainModel = true;   % Set to true to retrain the CNN from scratch
rng(42, "twister");
 
%% =====================================================================
%  SECTION 1: SIMULATION PARAMETERS
% ======================================================================
simParams = ntnSimParameters();
carrier   = simParams.Carrier;
pdsch     = simParams.PDSCH;
 
% NTN channel parameters
ntnParams.DopplerShift    = 5000;   % Hz  — LEO satellite Doppler
ntnParams.PropagationDelay = 2e-6;  % sec — 2 us time offset
ntnParams.SNRdB_range     = -5:5:25; % SNR sweep (dB)
 
%% =====================================================================
%  SECTION 2: TRAIN OR LOAD THE CNN
% ======================================================================
if trainModel
    fprintf("=== Training Phase ===\n");
    [trainData, trainLabels] = ntnGenerateTrainingData(5000, simParams, ntnParams, true);
 
    batchSize    = 16;
    valSplit     = batchSize;
 
    % Stack real/imag as separate samples along the batch dim
    trainData   = cat(4, trainData(:,:,1,:), trainData(:,:,2,:));
    trainLabels = cat(4, trainLabels(:,:,1,:), trainLabels(:,:,2,:));
 
    valData     = trainData(:,:,:,1:valSplit);
    valLabels   = trainLabels(:,:,:,1:valSplit);
    trainData   = trainData(:,:,:,valSplit+1:end);
    trainLabels = trainLabels(:,:,:,valSplit+1:end);
 
    valFreq = round(size(trainData,4)/batchSize/5);
 
    % CNN architecture — input: [72 x 14 x 1]
    layers = [
        imageInputLayer([72 14 1], Normalization="none")
        convolution2dLayer([9 9], 16, Padding="same")
        batchNormalizationLayer
        reluLayer
        convolution2dLayer([7 7], 16, Padding="same")
        batchNormalizationLayer
        reluLayer
        convolution2dLayer([5 5], 8,  Padding="same")
        reluLayer
        convolution2dLayer([3 3], 4,  Padding="same")
        reluLayer
        convolution2dLayer([3 3], 1,  Padding="same")
    ];
 
    options = trainingOptions("adam", ...
        "InitialLearnRate",    1e-3, ...
        "LearnRateSchedule",   "piecewise", ...
        "LearnRateDropFactor", 0.5, ...
        "LearnRateDropPeriod", 5, ...
        "MaxEpochs",           20, ...
        "Shuffle",             "every-epoch", ...
        "MiniBatchSize",       batchSize, ...
        "ValidationData",      {valData, valLabels}, ...
        "ValidationFrequency", valFreq, ...
        "ValidationPatience",  8, ...
        "Verbose",             false, ...
        "Plots",               "training-progress");
 
    [ntnCNN, trainingInfo] = trainnet(trainData, trainLabels, layers, ...
                                      "mean-squared-error", options);
    save("trainedNTNCNN.mat", "ntnCNN");
    fprintf("Model saved to trainedNTNCNN.mat\n");
else
    % Load pretrained network
    if isfile("trainedNTNCNN.mat")
        load("trainedNTNCNN.mat", "ntnCNN");
        fprintf("Pretrained NTN CNN loaded.\n");
    else
        error("No pretrained model found. Set trainModel=true to train.");
    end
end
 
%% =====================================================================
%  SECTION 3: MSE vs SNR EVALUATION
% ======================================================================
fprintf("\n=== MSE vs SNR Evaluation ===\n");
nSNR    = numel(ntnParams.SNRdB_range);
nTrials = 50;   % Monte Carlo trials per SNR point
 
mse_ls   = zeros(1, nSNR);
mse_mmse = zeros(1, nSNR);
mse_cnn  = zeros(1, nSNR);
 
% Pre-compute fixed DM-RS info (same every slot)
[dmrsSymbols, dmrsIndices] = ntnGetDMRS(carrier, pdsch);
 
% Build the fixed transmitted resource grid
txGrid = nrResourceGrid(carrier);
txGrid(dmrsIndices) = dmrsSymbols;
 
for iSNR = 1:nSNR
    snrDB  = ntnParams.SNRdB_range(iSNR);
    snrLin = 10^(snrDB/10);
 
    mseAccum_ls   = 0;
    mseAccum_mmse = 0;
    mseAccum_cnn  = 0;
 
    for iTrial = 1:nTrials
        % ---- Transmit -----------------------------------------------
        txWaveform = nrOFDMModulate(carrier, txGrid);
 
        % ---- Apply NTN LOS channel ----------------------------------
        [rxWaveform, H_perfect, offset] = ntnApplyLOSChannel( ...
            txWaveform, carrier, ntnParams);
 
        % ---- Add AWGN -----------------------------------------------
        waveInfo = nrOFDMInfo(carrier);
        N0 = 1 / sqrt(double(waveInfo.Nfft) * snrLin);
        noise = N0 * (randn(size(rxWaveform)) + 1j*randn(size(rxWaveform))) / sqrt(2);
        rxWaveform = rxWaveform + noise;
 
        % ---- Timing synchronisation & demodulation ------------------
        rxWaveform = rxWaveform(1+offset:end, :);
        rxGrid = nrOFDMDemodulate(carrier, rxWaveform);
        [K, L, ~] = size(rxGrid);
        if L < carrier.SymbolsPerSlot
            rxGrid = cat(2, rxGrid, zeros(K, carrier.SymbolsPerSlot-L));
        end
 
        % ---- LS estimate (pilot extraction + linear interpolation) --
        H_ls = ntnLSEstimate(rxGrid, dmrsIndices, dmrsSymbols, carrier);
 
        % ---- MMSE estimate ------------------------------------------
        H_mmse = ntnMMSEEstimate(rxGrid, dmrsIndices, dmrsSymbols, ...
                                  carrier, snrLin, ntnParams);
 
        % ---- CNN estimate -------------------------------------------
        nnIn = cat(4, real(H_ls), imag(H_ls));  % [72 14 1 2]
        % Predict real and imaginary parts separately
        nnOut_re = predict(ntnCNN, nnIn(:,:,:,1));
        nnOut_im = predict(ntnCNN, nnIn(:,:,:,2));
        H_cnn = complex(nnOut_re(:,:,1,1), nnOut_im(:,:,1,1));
 
        % ---- Accumulate MSE -----------------------------------------
        err_ls   = H_perfect(:) - H_ls(:);
        err_mmse = H_perfect(:) - H_mmse(:);
        err_cnn  = H_perfect(:) - H_cnn(:);
 
        mseAccum_ls   = mseAccum_ls   + mean(abs(err_ls).^2);
        mseAccum_mmse = mseAccum_mmse + mean(abs(err_mmse).^2);
        mseAccum_cnn  = mseAccum_cnn  + mean(abs(err_cnn).^2);
    end
 
    mse_ls(iSNR)   = mseAccum_ls   / nTrials;
    mse_mmse(iSNR) = mseAccum_mmse / nTrials;
    mse_cnn(iSNR)  = mseAccum_cnn  / nTrials;
 
    fprintf("SNR = %4.1f dB | LS = %.4e | MMSE = %.4e | CNN = %.4e\n", ...
        snrDB, mse_ls(iSNR), mse_mmse(iSNR), mse_cnn(iSNR));
end
 
%% =====================================================================
%  SECTION 4: PLOT RESULTS
% ======================================================================
ntnPlotMSEvsSNR(ntnParams.SNRdB_range, mse_ls, mse_mmse, mse_cnn);
ntnPlotChannelEstimates(carrier, pdsch, ntnParams, ntnCNN, 15);