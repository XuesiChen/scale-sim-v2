from scalesim.scale_sim import scalesim
import os

absolute_dir_path = os.path.dirname(os.path.realpath(__file__))

config_list = ['TPU256x256', 'TPU128x128', 'TPU64x64', 'TPU32x32']

topology = absolute_dir_path + "/../topologies/CSV/DeepBench.csv"
path_to_config = absolute_dir_path + "/../configs/"
logpath = absolute_dir_path + "/../test_runs"
inp_type = absolute_dir_path + "/conv"

gemm_input = False
if inp_type == 'gemm':
    gemm_input = True

for conf in config_list:
    config = absolute_dir_path + "/../configs/" + conf + ".cfg"
    s = scalesim(save_disk_space=True, verbose=True,
                    config=config,
                    topology=topology,
                    input_type_gemm=gemm_input
                    )
    s.run_scale(top_path=logpath)