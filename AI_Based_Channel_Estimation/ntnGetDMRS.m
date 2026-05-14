function [dmrsSymbols, dmrsIndices] = ntnGetDMRS(carrier, pdsch)
% Return DM-RS symbols and indices for the given carrier/PDSCH config
dmrsSymbols = nrPDSCHDMRS(carrier, pdsch);
dmrsIndices = nrPDSCHDMRSIndices(carrier, pdsch);
end