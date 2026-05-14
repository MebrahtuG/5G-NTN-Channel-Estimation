function H_ls = ntnLSEstimate(rxGrid, dmrsIndices, dmrsSymbols, carrier)
% Least-Squares channel estimation:
%   1. Extract received pilots and divide by known pilots → LS pilot estimates
%   2. Scattered interpolation over the full resource grid

K = carrier.NSizeGrid * 12;
L = carrier.SymbolsPerSlot;

% LS pilot estimates
dmrsRx  = rxGrid(dmrsIndices);
H_pilot = dmrsRx ./ dmrsSymbols;   % element-wise division

% Coordinates of pilot positions
[rows, cols] = ind2sub([K, L], dmrsIndices);

% Interpolate over the full (K x L) grid
[lGrid, kGrid] = meshgrid(1:L, 1:K);
F = scatteredInterpolant(double(cols), double(rows), H_pilot, ...
    "linear", "nearest");
H_ls = F(lGrid, kGrid);
end