# iter_func.py
# created by Yichao

# region
import os
import glob
import json
import dpdata
import logging
import numpy as np
import random
import sys
import itertools

from packaging.version import Version 
from dpgen.generator.lib.run_calypso import run_calypso_model_devi

from packaging.version import Version


from dpgen.generator.lib.utils import (
    create_path,
    make_iter_name,
    symlink_user_forward_files,
    log_task
)

from workflow.lammps_run import run_lammps

from fp_interface.fp_vasp import make_fp_vasp, post_fp_vasp, _vasp_check_fin
from fp_interface.fp_pwscf import make_fp_pwscf, post_fp_pwscf  
from fp_interface.fp_siesta import make_fp_siesta, post_fp_siesta
from fp_interface.fp_gaussian import make_fp_gaussian, post_fp_gaussian
from fp_interface.fp_cp2k import make_fp_cp2k, post_fp_cp2k


from dpgen.dispatcher.Dispatcher import make_submission


from model_devi import (
    _make_model_devi_native, _make_model_devi_native_gromacs,
    _make_model_devi_amber, _make_model_devi_revmat
)

from config.config import *


from utils.utils import (
    expand_idx,
    make_model_devi_conf_name
)

from dpgen.util import convert_training_data_to_hdf5, expand_sys_str
# endregion





def detect_batch_size(batch_size, system=None):
    if type(batch_size) == int:
        return batch_size
    elif batch_size == "auto":
        # automaticcaly set batch size, batch_size = 32 // atom_numb (>=1, <=fram_numb)
        # check if h5 file
        format = "deepmd/npy" if "#" not in system else "deepmd/hdf5"
        s = dpdata.LabeledSystem(system, fmt=format)
        return int(
            min(np.ceil(32.0 / float(s["coords"].shape[1])), s["coords"].shape[0])
        )
    else:
        raise RuntimeError("Unsupported batch size")




def _check_empty_iter(iter_index, max_v=0):
    fp_path = os.path.join(make_iter_name(iter_index), fp_name)
    # check the number of collected data
    sys_data = glob.glob(os.path.join(fp_path, "data.*"))
    empty_sys = []
    for ii in sys_data:
        nframe = 0
        sys_paths = expand_sys_str(ii)
        for single_sys in sys_paths:
            sys = dpdata.LabeledSystem(os.path.join(single_sys), fmt="deepmd/npy")
            nframe += len(sys)
        empty_sys.append(nframe < max_v)
    return all(empty_sys)





