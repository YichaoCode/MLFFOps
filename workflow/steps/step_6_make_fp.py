# workflow/steps/step_6_make_fp.py
# Created by Yichao


from config.config import *

from fp_interface.fp_vasp import make_fp_vasp
from fp_interface.fp_pwscf import make_fp_pwscf  
from fp_interface.fp_siesta import make_fp_siesta
from fp_interface.fp_gaussian import make_fp_gaussian
from fp_interface.fp_cp2k import make_fp_cp2k

def make_fp(iter_index, jdata, mdata, base_dir):
    """Generate first-principles data.

    Args:
        iter_index (int): Current iteration index.
        jdata (dict): Job configuration dictionary.
        mdata (dict): Machine configuration dictionary.
    """
    fp_style = jdata['fp_style']
    if fp_style == 'vasp':
        make_fp_vasp(iter_index, jdata, base_dir)



    elif fp_style == 'pwscf':
        make_fp_pwscf(iter_index, jdata)
    elif fp_style == 'siesta':
        make_fp_siesta(iter_index, jdata)
    elif fp_style == 'gaussian':
        make_fp_gaussian(iter_index, jdata)
    elif fp_style == 'cp2k':
        make_fp_cp2k(iter_index, jdata)
    else:
        raise RuntimeError('unsupported fp style')
