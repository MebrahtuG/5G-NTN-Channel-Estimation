function ntnPlotMSEvsSNR(snr_dB, mse_ls, mse_mmse, mse_cnn)
% Plot MSE vs SNR comparison for all three estimators

figure("Name", "MSE vs SNR — 5G NTN Channel Estimation", "NumberTitle", "off");
semilogy(snr_dB, mse_ls,   "b-o",  "LineWidth", 1.8, "MarkerSize", 7, ...
    "DisplayName", "LS (interpolated)");
hold on;
semilogy(snr_dB, mse_mmse, "r-s",  "LineWidth", 1.8, "MarkerSize", 7, ...
    "DisplayName", "MMSE (parametric LOS)");
semilogy(snr_dB, mse_cnn,  "g-^",  "LineWidth", 1.8, "MarkerSize", 7, ...
    "DisplayName", "CNN (DL-based)");
grid on;
xlabel("SNR (dB)", "FontSize", 12);
ylabel("MSE", "FontSize", 12);
title("Channel Estimation MSE vs SNR — 5G NTN LOS", "FontSize", 13);
legend("Location", "southwest", "FontSize", 11);
set(gca, "FontSize", 11);
end