%% 5G NTN Channel Estimation using Deep Learning – NTN-TDL-C (LOS, Satellite-to-UE)
%
% This script adapts the terrestrial TDL deep learning channel estimation
% example to a Non-Terrestrial Network (NTN) scenario using:
%   - NTN-TDL-C delay profile  (LOS, elevation = 30 deg)
%   - LEO satellite at 600 km altitude (S-band, 2.1 GHz)
%   - Static UE  (MaximumDopplerShift = 0 Hz for UE movement)
%   - Satellite Doppler shift applied via SatelliteDopplerShift property
%   - Monte Carlo MSE evaluation over SNR = -10:1:20 dB


clear; clc; close all;
rng(42,"twister")



trainModel = true; % true  → train CNN from scratch
                      % false → load pretrained network
N_sub = 72;
N_sym = 14;

%  1.  NTN COMMON PARAMETERS  (3GPP TR 38.821 table 6.1.2-4)

ntnParams.CarrierFrequency  = 2.1e9;          % S-band [Hz]
ntnParams.ElevationAngle    = 30;           % elevation angle [deg]
ntnParams.SatelliteAltitude = 600000;       % LEO altitude [m]
ntnParams.MobileAltitude    = 0;            % UE on ground [m]
ntnParams.MobileSpeed       = 0;            % STATIC UE [m/s]
ntnParams.DelayProfile      = "NTN-TDL-C"; % LOS NTN profile
ntnParams.DelaySpread       = 30e-9;        % 30 ns [s]
% ntnParams.MIMOCorrelation   = "Low";
% ntnParams.Polarization      = "Co-Polar";
ntnParams.NumTxAntennas     = 1;            % SISO downlink
ntnParams.NumRxAntennas     = 1;


% Satellite Doppler shift
satelliteDopplerShift = dopplerShiftCircularOrbit( ...
    ntnParams.ElevationAngle, ...
    ntnParams.SatelliteAltitude, ...
    ntnParams.MobileAltitude, ...
    ntnParams.CarrierFrequency);

fprintf("Satellite Doppler Shift : %.2f Hz\n", satelliteDopplerShift);

mobileMaxDoppler=0;

% SIMULATION PARAMETERS  (carrier / PDSCH / DM-RS)

simParameters = hDeepLearningChanEstSimParameters();
carrier       = simParameters.Carrier;
pdsch         = simParameters.PDSCH;
waveformInfo  = nrOFDMInfo(carrier);


% CNN TRAINING (or load pretrained)

if trainModel
    fprintf("\n=== Generating NTN Training Data ===\n");
    [trainData, trainLabels] = hGenerateNTNTrainingData( ...
        500, ntnParams, simParameters, satelliteDopplerShift, ...
        mobileMaxDoppler, true);

    batchSize = 8;

    % Stack real & imaginary grids along batch dimension
    trainData   = cat(4, trainData(:,:,1,:),   trainData(:,:,2,:));
    trainLabels = cat(4, trainLabels(:,:,1,:), trainLabels(:,:,2,:));

    % Validation set = first mini-batch
    valData   = trainData(:,:,:,1:batchSize);
    valLabels = trainLabels(:,:,:,1:batchSize);
    trainData   = trainData(:,:,:,batchSize+1:end);
    trainLabels = trainLabels(:,:,:,batchSize+1:end);

    valFrequency = round(size(trainData,4)/batchSize/5);

    % CNN architecture (unchanged from original example)
    layers = [
        imageInputLayer([N_sub N_sym 1], Normalization="none")
        convolution2dLayer([9 9], 2, Padding="same"); reluLayer
        convolution2dLayer([9 9], 2, Padding="same"); reluLayer
        convolution2dLayer([5 5], 2, Padding="same"); reluLayer
        convolution2dLayer([5 5], 2, Padding="same"); reluLayer
        convolution2dLayer([5 5], 1, Padding="same")
        ];

    options = trainingOptions("adam", ...
        InitialLearnRate   = 3e-4, ...
        MaxEpochs          = 10, ...
        Shuffle            = "every-epoch", ...
        Verbose            = false, ...
        Plots              = "training-progress", ...
        MiniBatchSize      = batchSize, ...
        ValidationData     = {valData, valLabels}, ...
        ValidationFrequency = valFrequency, ...
        ValidationPatience  = 5);

    [channelEstimationCNN, trainingInfo] = trainnet( ...
        trainData, trainLabels, layers, "mean-squared-error", options);

    save("trainedNTN_TDL_C_Network.mat","channelEstimationCNN");
    fprintf("Network trained and saved.\n");
else
    load("trainedChannelEstimationNetwork.mat");
    fprintf("Pre-trained network loaded.\n");
end

disp(channelEstimationCNN.Layers);

%  4.  MONTE CARLO  MSE vs SNR  (-10 : 1 : 20 dB)

fprintf("\n=== Monte Carlo Simulation  (SNR = -10 to 20 dB) ===\n");

SNRdB_vec  = -10:1:20;
numMC      = 100;          % trials per SNR point
numSNR     = numel(SNRdB_vec);

mse_nn       = zeros(1, numSNR);
mse_interp   = zeros(1, numSNR);
mse_practical = zeros(1, numSNR);

% DM-RS symbols and indices 
dmrsSymbols = nrPDSCHDMRS(carrier, pdsch);
dmrsIndices = nrPDSCHDMRSIndices(carrier, pdsch);

% Fixed transmit waveform for all trials
pdschGrid_mc = nrResourceGrid(carrier);
pdschGrid_mc(dmrsIndices) = dmrsSymbols;
txWav_base   = nrOFDMModulate(carrier, pdschGrid_mc);

for iSNR = 1:numSNR
    SNRdB = SNRdB_vec(iSNR);
    SNR   = 10^(SNRdB/10);
    N0    = 1/sqrt(ntnParams.NumRxAntennas * double(waveformInfo.Nfft) * SNR);

    acc_nn   = 0;
    acc_int  = 0;
    acc_prac = 0;

    for iTrial = 1:numMC
        %  NTN-TDL-C channel realization 
        ch = nrTDLChannel;
        ch.DelayProfile          = ntnParams.DelayProfile;   % NTN-TDL-C (LOS)
        ch.DelaySpread           = ntnParams.DelaySpread;    % fixed 30 ns
        ch.TransmissionDirection = "Downlink";
        % ch.MIMOCorrelation       = ntnParams.MIMOCorrelation;
        % ch.Polarization          = ntnParams.Polarization;
        ch.NumTransmitAntennas   = ntnParams.NumTxAntennas;
        ch.NumReceiveAntennas    = ntnParams.NumRxAntennas;
        ch.SampleRate            = waveformInfo.SampleRate;
        ch.MaximumDopplerShift   = mobileMaxDoppler;         % 0 – static UE
        ch.SatelliteDopplerShift = satelliteDopplerShift;    % LEO offset
        ch.ChannelResponseOutput = "ofdm-response";
        % ch.RandomStream          = "mt19937ar with seed";
        ch.Seed                  = iTrial;                   % unique realization

        chInfoMC   = info(ch);
        maxDelayMC = chInfoMC.MaximumChannelDelay;

        txWav = [txWav_base; zeros(maxDelayMC, 1)];
        [rxWav, ofdmResp, off] = ch(txWav, carrier);

        % Add AWGN
        noise = N0 * randn(size(rxWav), "like", rxWav);
        rxWav = rxWav + noise;
        rxWav = rxWav(1+off:end, :);

        % OFDM demodulation
        rxG = nrOFDMDemodulate(carrier, rxWav);
        [Kg,Lg,~] = size(rxG);
        if Lg < carrier.SymbolsPerSlot
            rxG = cat(2, rxG, zeros(Kg, carrier.SymbolsPerSlot-Lg, 1));
        end

        H_perf = ofdmResp;

        % Linear interpolation
        H_int = hPreprocessInput(rxG, dmrsIndices, dmrsSymbols);

        % Practical estimator (nrChannelEstimate)
        [H_prac, ~] = nrChannelEstimate(carrier, rxG, dmrsIndices, ...
            dmrsSymbols, "CDMLengths", pdsch.DMRS.CDMLengths);

        % Neural network estimator
        nnIn = cat(4, real(H_int), imag(H_int));
        H_nn = predict(channelEstimationCNN, nnIn);
        H_nn = complex(H_nn(:,:,:,1), H_nn(:,:,:,2));

        % Accumulate MSE
        acc_nn   = acc_nn   + mean(abs(H_perf(:) - H_nn(:)).^2);
        acc_int  = acc_int  + mean(abs(H_perf(:) - H_int(:)).^2);
        acc_prac = acc_prac + mean(abs(H_perf(:) - H_prac(:)).^2);
    end

    mse_nn(iSNR)        = acc_nn   / numMC;
    mse_interp(iSNR)    = acc_int  / numMC;
    mse_practical(iSNR) = acc_prac / numMC;

    fprintf("SNR = %+4d dB | NN = %.4e | Practical = %.4e | Interp = %.4e\n", ...
        SNRdB, mse_nn(iSNR), mse_practical(iSNR), mse_interp(iSNR));
end


%  MONTE CARLO RESULT PLOT

figure("Name","Monte Carlo MSE vs SNR – NTN-TDL-C");
semilogy(SNRdB_vec, mse_interp,    'b--o', LineWidth=1.8, MarkerSize=6); hold on;
semilogy(SNRdB_vec, mse_practical, 'k-s',  LineWidth=1.8, MarkerSize=6);
semilogy(SNRdB_vec, mse_nn,        'r-^',  LineWidth=1.8, MarkerSize=6);
grid on; grid minor;
xlabel('SNR (dB)', FontSize=12);
ylabel('Mean Squared Error (MSE)', FontSize=12);
title({'Monte Carlo Channel Estimation – NTN-TDL-C (LOS)', ...
       'LEO 600 km | S-band 2.1 GHz | Static UE | Elevation 30 deg'}, FontSize=12);
legend('Linear Interpolation','Practical Estimator','Neural Network', ...
       Location='northeast', FontSize=11);
xlim([SNRdB_vec(1) SNRdB_vec(end)]);

%  LOCAL FUNCTIONS

function hest = hPreprocessInput(rxGrid, dmrsIndices, dmrsSymbols)
% hPreprocessInput  Linear interpolation from DM-RS pilots.
    dmrsRx   = rxGrid(dmrsIndices);
    dmrsEsts = dmrsRx .* conj(dmrsSymbols);

    [rxDMRSGrid, hest] = deal(zeros(size(rxGrid)));
    rxDMRSGrid(dmrsIndices) = dmrsSymbols;

    [rows, cols] = find(rxDMRSGrid ~= 0);
    dmrsSubs     = [rows, cols, ones(size(cols))];
    [l_hest, k_hest] = meshgrid(1:size(hest,2), 1:size(hest,1));

    f    = scatteredInterpolant(dmrsSubs(:,2), dmrsSubs(:,1), dmrsEsts);
    hest = f(l_hest, k_hest);
end


function [trainData, trainLabels] = hGenerateNTNTrainingData( ...
        dataSize, ntnParams, simParameters, satelliteDopplerShift, ...
       ~, printProgress)
%  hGenerateNTNTrainingData  Training data for NTN-TDL-C channel estimation CNN.
%  Each example randomizes: channel seed, delay spread (10–100 ns), SNR (−10 to 20 dB).


    if printProgress, fprintf("Starting NTN data generation...\n"); end

    carrier      = simParameters.Carrier;
    pdsch        = simParameters.PDSCH;
    waveformInfo = nrOFDMInfo(carrier);

    % DM-RS 
    dmrsSymbols = nrPDSCHDMRS(carrier, pdsch);
    dmrsIndices = nrPDSCHDMRSIndices(carrier, pdsch);

    grid = nrResourceGrid(carrier, ntnParams.NumTxAntennas);
    [~, dmrsAntIdx] = nrExtractResources(dmrsIndices, grid);
    grid(dmrsAntIdx) = dmrsSymbols;
    txWav0 = nrOFDMModulate(carrier, grid);

    % Interpolation mesh 
    [rows, cols] = find(grid ~= 0);
    dmrsSubs     = [rows, cols, ones(size(cols))];
    [l_h, k_h]   = meshgrid(1:size(grid,2), 1:size(grid,1));

    [trainData, trainLabels] = deal(zeros([72 14 2 dataSize]));

    for i = 1:dataSize
        ch = nrTDLChannel;
        ch.DelayProfile          = ntnParams.DelayProfile;    % NTN-TDL-C
        
        % ±5 ns jitter around 30 ns is added for training diversity.
        ch.DelaySpread           = (25 + randi([0 10])) * 1e-9;  % 25–35 ns
        ch.TransmissionDirection = "Downlink";
        % ch.MIMOCorrelation       = ntnParams.MIMOCorrelation;
        % ch.Polarization          = ntnParams.Polarization;
        ch.NumTransmitAntennas   = ntnParams.NumTxAntennas;
        ch.NumReceiveAntennas    = ntnParams.NumRxAntennas;
        ch.SampleRate            = waveformInfo.SampleRate;
        % ch.MaximumDopplerShift   = mobileMaxDoppler;          % 0 Hz – static UE
        ch.SatelliteDopplerShift = satelliteDopplerShift;     % LEO Doppler offset
        ch.ChannelResponseOutput = "ofdm-response";
        % ch.RandomStream          = "mt19937ar with seed";
        ch.Seed                  = randi([1001 2000]);

        chInfo = info(ch);
        txWav  = [txWav0; zeros(chInfo.MaximumChannelDelay, size(txWav0,2))];

        [rxWav, ofdmChanResp, offset] = ch(txWav, carrier);

        % Random SNR in [-10, 20] dB for training diversity
        SNRdB = randi([-10 20]);
        SNR   = 10^(SNRdB/10);
        N0    = 1/sqrt(2 * ntnParams.NumRxAntennas * double(waveformInfo.Nfft) * SNR);
        noise = N0 * complex(randn(size(rxWav)), randn(size(rxWav)));
        rxWav = rxWav + noise;
        rxWav = rxWav(1+offset:end, :);

        rxGrid = nrOFDMDemodulate(carrier, rxWav);
        [K,L,~] = size(rxGrid);
        if L < carrier.SymbolsPerSlot
            rxGrid = cat(2, rxGrid, zeros(K, carrier.SymbolsPerSlot-L, 1));
        end

        % Linear interpolation from DM-RS
        dmrsRx   = rxGrid(dmrsIndices);
        dmrsEsts = dmrsRx .* conj(dmrsSymbols);
        f    = scatteredInterpolant(dmrsSubs(:,2), dmrsSubs(:,1), dmrsEsts);
        hest = f(k_h, l_h);

        trainData(:,:,:,i)   = cat(3, real(hest),        imag(hest));
        trainLabels(:,:,:,i) = cat(3, real(ofdmChanResp), imag(ofdmChanResp));

        if printProgress && mod(i, max(1,round(dataSize/25))) == 0
            fprintf("  %3.0f%% complete\n", i/dataSize*100);
        end
    end
    if printProgress, fprintf("Data generation complete!\n"); end
end


function simParameters = hDeepLearningChanEstSimParameters()
% hDeepLearningChanEstSimParameters  Carrier and PDSCH/DM-RS configuration.
    simParameters.Carrier = nrCarrierConfig;
    simParameters.Carrier.NSizeGrid         = 6;
    simParameters.Carrier.SubcarrierSpacing = 15;
    simParameters.Carrier.CyclicPrefix      = "Normal";
    simParameters.Carrier.NCellID           = 2;

    simParameters.NTxAnts = 1;
    simParameters.NRxAnts = 1;

    simParameters.PDSCH = nrPDSCHConfig;
    simParameters.PDSCH.PRBSet           = 0:simParameters.Carrier.NSizeGrid-1;
    simParameters.PDSCH.SymbolAllocation = [0, simParameters.Carrier.SymbolsPerSlot];
    simParameters.PDSCH.MappingType      = "A";
    simParameters.PDSCH.NID              = simParameters.Carrier.NCellID;
    simParameters.PDSCH.RNTI             = 1;
    simParameters.PDSCH.VRBToPRBInterleaving = 0;
    simParameters.PDSCH.NumLayers        = 1;
    simParameters.PDSCH.Modulation       = "QPSK";

    simParameters.PDSCH.DMRS.DMRSPortSet            = 0:simParameters.PDSCH.NumLayers-1;
    simParameters.PDSCH.DMRS.DMRSTypeAPosition       = 2;
    simParameters.PDSCH.DMRS.DMRSLength              = 1;
    simParameters.PDSCH.DMRS.DMRSAdditionalPosition  = 1;
    simParameters.PDSCH.DMRS.DMRSConfigurationType   = 2;
    simParameters.PDSCH.DMRS.NumCDMGroupsWithoutData = 1;
    simParameters.PDSCH.DMRS.NIDNSCID               = 1;
    simParameters.PDSCH.DMRS.NSCID                  = 0;
end