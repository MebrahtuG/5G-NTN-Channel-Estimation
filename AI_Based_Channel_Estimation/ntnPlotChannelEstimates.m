function ntnPlotChannelEstimates(carrier, pdsch, ntnParams, ntnCNN, snrDB)
% Plot a single-slot comparison of LS, MMSE, CNN, and perfect channel

[dmrsSymbols, dmrsIndices] = ntnGetDMRS(carrier, pdsch);
txGrid = nrResourceGrid(carrier);
txGrid(dmrsIndices) = dmrsSymbols;
txWaveform = nrOFDMModulate(carrier, txGrid);
waveInfo   = nrOFDMInfo(carrier);
snrLin     = 10^(snrDB/10);

[rxWaveform, H_perfect, offset] = ntnApplyLOSChannel(txWaveform, carrier, ntnParams);
N0 = 1/sqrt(double(waveInfo.Nfft)*snrLin);
rxWaveform = rxWaveform + N0*(randn(size(rxWaveform))+1j*randn(size(rxWaveform)))/sqrt(2);
rxWaveform = rxWaveform(1+offset:end);

rxGrid = nrOFDMDemodulate(carrier, rxWaveform);
K = carrier.NSizeGrid*12; L = carrier.SymbolsPerSlot;
if size(rxGrid,2) < L
    rxGrid = cat(2, rxGrid, zeros(K, L-size(rxGrid,2)));
end

H_ls   = ntnLSEstimate(rxGrid, dmrsIndices, dmrsSymbols, carrier);
H_mmse = ntnMMSEEstimate(rxGrid, dmrsIndices, dmrsSymbols, carrier, snrLin, ntnParams);

nnIn_re = cat(4, real(H_ls), imag(H_ls));
H_cnn_re = predict(ntnCNN, nnIn_re(:,:,:,1));
H_cnn_im = predict(ntnCNN, nnIn_re(:,:,:,2));
H_cnn = complex(H_cnn_re(:,:,1,1), H_cnn_im(:,:,1,1));

mse_ls   = mean(abs(H_perfect(:) - H_ls(:)).^2);
mse_mmse = mean(abs(H_perfect(:) - H_mmse(:)).^2);
mse_cnn  = mean(abs(H_perfect(:) - H_cnn(:)).^2);

cmax = max(abs([H_ls(:); H_mmse(:); H_cnn(:); H_perfect(:)]));

figure("Name", sprintf("Channel Estimates @ SNR=%d dB", snrDB), "NumberTitle","off");
subplot(1,4,1); imagesc(abs(H_ls));
title(["LS (interpolated)", sprintf("MSE=%.3e", mse_ls)]);
xlabel("OFDM Symbol"); ylabel("Subcarrier"); clim([0 cmax]);

subplot(1,4,2); imagesc(abs(H_mmse));
title(["MMSE (LOS prior)", sprintf("MSE=%.3e", mse_mmse)]);
xlabel("OFDM Symbol"); ylabel("Subcarrier"); clim([0 cmax]);

subplot(1,4,3); imagesc(abs(H_cnn));
title(["CNN (deep learning)", sprintf("MSE=%.3e", mse_cnn)]);
xlabel("OFDM Symbol"); ylabel("Subcarrier"); clim([0 cmax]);

subplot(1,4,4); imagesc(abs(H_perfect));
title("Perfect channel"); xlabel("OFDM Symbol"); ylabel("Subcarrier");
clim([0 cmax]);
colormap turbo;
sgtitle(sprintf("5G NTN LOS Channel Estimation — SNR = %d dB", snrDB), "FontSize",13);
end