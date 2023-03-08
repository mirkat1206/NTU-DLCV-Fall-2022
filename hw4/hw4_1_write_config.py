import os
import sys

if len(sys.argv) != 2:
    print('Error: wrong format')
    print('\tpython3 hw4_1_write_config.py <path to transform_test.json>')

json_path = sys.argv[1]

# at DVGO directory
json_dirpath = "../" + os.path.dirname(json_path)

with open('./DirectVoxGO/configs/nerf/hw4.py', 'w') as f:
    s = "_base_ = '../default.py'\n\n" + \
        "expname = 'dvgo_final'\n" + \
        "basedir = './logs/nerf_synthetic'\n\n" + \
        "data = dict(\n" + \
        "\tdatadir='" + json_dirpath + "',\n" + \
        "\tdataset_type='blender',\n" + \
        "\twhite_bkgd=True,\n" + \
        ")\n"
    f.write(s)

print('Writing /DirectVoxGO/configs/nerf/hw4.py finished...')
