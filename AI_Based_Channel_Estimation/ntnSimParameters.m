function simParams = ntnSimParameters()
% Simulation parameters for 5G NTN PDSCH channel estimation
% Grid: 72 subcarriers (6 RBs) x 14 OFDM symbols

% Carrier: 6 RBs at 15 kHz SCS  → 72 subcarriers, fits NTN numerology
simParams.Carrier = nrCarrierConfig;
simParams.Carrier.NSizeGrid          = 6;       % 6 RBs = 72 subcarriers
simParams.Carrier.SubcarrierSpacing  = 15;      % kHz (NTN: 15 or 30 kHz)
simParams.Carrier.CyclicPrefix       = "Normal";
simParams.Carrier.NCellID            = 1;

% Antennas (SISO)
simParams.NTxAnts = 1;
simParams.NRxAnts = 1;

% PDSCH configuration
simParams.PDSCH = nrPDSCHConfig;
simParams.PDSCH.PRBSet           = 0:simParams.Carrier.NSizeGrid-1;
simParams.PDSCH.SymbolAllocation = [0, simParams.Carrier.SymbolsPerSlot];
simParams.PDSCH.MappingType      = "A";
simParams.PDSCH.NID              = simParams.Carrier.NCellID;
simParams.PDSCH.RNTI             = 1;
simParams.PDSCH.NumLayers        = 1;
simParams.PDSCH.Modulation       = "QPSK";

% DM-RS configuration (type 1, single symbol, positions at l=2 and l=11)
simParams.PDSCH.DMRS.DMRSPortSet             = 0;
simParams.PDSCH.DMRS.DMRSTypeAPosition       = 2;
simParams.PDSCH.DMRS.DMRSLength              = 1;
simParams.PDSCH.DMRS.DMRSAdditionalPosition  = 1;
simParams.PDSCH.DMRS.DMRSConfigurationType   = 1;
simParams.PDSCH.DMRS.NumCDMGroupsWithoutData = 2;
simParams.PDSCH.DMRS.NIDNSCID               = 1;
simParams.PDSCH.DMRS.NSCID                  = 0;
end