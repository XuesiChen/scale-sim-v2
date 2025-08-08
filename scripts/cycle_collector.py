import os
import pandas as pd

absolute_dir_path = os.path.dirname(os.path.realpath(__file__))
results_folder = absolute_dir_path + "/../test_runs/"

config_list = ['TPU256x256', 'TPU128x128', 'TPU64x64', 'TPU32x32']
total_cycles = {}
total_compute_utils = {}
total_sram_access = {}
total_dram_access = {}

for config in config_list:
    result_csv = results_folder + config + "/COMPUTE_REPORT.csv"
    df = pd.read_csv(result_csv)
    # sum total cycles for all layers
    total_cycle = df[' Total Cycles'].sum()
    total_cycles[config] = total_cycle
    total_compute_utils[config] = (df[' Compute Util %'] / 100 * df[' Total Cycles']).sum() / total_cycle

for config in config_list:
    mem_access_csv = results_folder + config + "/DETAILED_ACCESS_REPORT.csv"
    df = pd.read_csv(mem_access_csv)

    SRAM_IFMAP_access = df[' SRAM IFMAP Reads'].sum()
    SRAM_FILTER_access = df[' SRAM Filter Reads'].sum()
    SRAM_OFMAP_access = df[' SRAM OFMAP Writes'].sum()
    SRAM_access = SRAM_IFMAP_access + SRAM_FILTER_access + SRAM_OFMAP_access # words
    SRAM_access_bits = SRAM_access * 16
    total_sram_access[config] = SRAM_access_bits

    DRAM_IFMAP_access = df[' DRAM IFMAP Reads'].sum()
    DRAM_FILTER_access = df[' DRAM Filter Reads'].sum()
    DRAM_OFMAP_access = df[' DRAM OFMAP Writes'].sum()
    DRAM_access = DRAM_IFMAP_access + DRAM_FILTER_access + DRAM_OFMAP_access # words
    DRAM_access_bits = DRAM_access * 16
    total_dram_access[config] = DRAM_access_bits

for config in config_list:
    print(config)
    print("Total Cycles: ", total_cycles[config])
    print("Total Compute Utilization: ", total_compute_utils[config])
    print("Total SRAM Access: ", total_sram_access[config])
    print("Total DRAM Access: ", total_dram_access[config])


# plot the total cycles for each configuration normalized to the TPU256x256 configuration

