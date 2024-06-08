# workflow/steps/step_8_post_fp.py
# Created by Yichao


# region
from fp_interface.fp_vasp import  post_fp_vasp
from fp_interface.fp_pwscf import  post_fp_pwscf  
from fp_interface.fp_siesta import  post_fp_siesta
from fp_interface.fp_gaussian import  post_fp_gaussian
from fp_interface.fp_cp2k import post_fp_cp2k


from config.config import *

# endregion


def post_fp(iter_index, jdata, base_dir):
    """Post-process first-principles calculations.

    Args:
        iter_index (int): Current iteration index.
        jdata (dict): Job configuration dictionary.
    """
    fp_style = jdata['fp_style']
    if fp_style == 'vasp':
        post_fp_vasp(iter_index, jdata, base_dir)
    elif fp_style == 'pwscf':
        post_fp_pwscf(iter_index, jdata)
    elif fp_style == 'siesta':
        post_fp_siesta(iter_index, jdata)
    elif fp_style == 'gaussian':
        post_fp_gaussian(iter_index, jdata)
    elif fp_style == 'cp2k':
        post_fp_cp2k(iter_index, jdata)
    else:
        raise RuntimeError('unsupported fp style')
