from scalesim.scale_sim import scalesim
import os

absolute_dir_path = os.path.dirname(os.path.realpath(__file__))

# config_list = ['google_VSM']
config_list = ['google']
# config_list = ['google_2xBW']
# config_list = ['google','google_VLM']

# topology = absolute_dir_path + "/../topologies/translation/GPT2.csv"
# topology = absolute_dir_path + "/../topologies/conv_nets/Resnet50_copy.csv"
topology = absolute_dir_path + "/../topologies/CSV/DeepBench.csv"
path_to_config = absolute_dir_path + "/../configs/"
# logpath = absolute_dir_path + "/../ECE6950_runs"
# logpath = absolute_dir_path + "/../ECE6950_Resnet50_BWlimited"
# logpath = absolute_dir_path + "/../ECE6950_GPT_BWlimited_improved2"
logpath = absolute_dir_path + "/../ECE6950_deepbench_BWlimited_improved2"
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