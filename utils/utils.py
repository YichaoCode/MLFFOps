# utils/utils.py
# created by Yichao

import os
import glob
import shutil
import numpy as np
import dpdata
import logging
from config import *

from utils.path_utils import find_only_one_key
from packaging.version import Version


from dpgen.generator.lib.utils import (
    copy_file_list,
    create_path,
    log_iter,
    log_task,
    make_iter_name,
    record_iter,
    replace,
    symlink_user_forward_files,
)
# 定义 MODEL_DEVI_TASK_FMT 格式字符串

# MODEL_DEVI_TASK_FMT = "model_devi_task_%03d_%04d"

# 定义 MODEL_DEVI_CONF_FMT 格式字符串

# MODEL_DEVI_CONF_FMT = "model_devi_conf_%02d_%03d"

# 定义 fp_task_fmt 格式字符串
fp_task_fmt = "fp_task_%d_%04d"
def get_job_names(jdata):
    jobkeys = []
    for ii in jdata.keys():
        if ii.split("_")[0] == "job":
            jobkeys.append(ii)
    jobkeys.sort()
    return jobkeys


def make_model_devi_task_name(sys_idx, task_idx):
    logging.debug("Generating model deviation task name...")
    logging.debug("Format string: %s", "task." + MODEL_DEVI_TASK_FMT)
    logging.debug("System index: %d", sys_idx)
    logging.debug("Task index: %d", task_idx)

    task_name = "task." + MODEL_DEVI_TASK_FMT % (sys_idx, task_idx)

    logging.debug("Generated task name: %s", task_name)
    return task_name


def make_model_devi_conf_name(sys_idx, task_idx):
    logging.debug("Generating model deviation task name...")
    logging.debug("Format string: %s", "task." + MODEL_DEVI_TASK_FMT)
    logging.debug("System index: %d", sys_idx)
    logging.debug("Task index: %d", task_idx)

    task_name = "task." + MODEL_DEVI_TASK_FMT % (sys_idx, task_idx)

    logging.debug("Generated task name: %s", task_name)
    return task_name


def make_model_devi_conf_name_save(sys_idx, conf_idx):
    logging.debug("Generating model deviation conf name...")
    logging.debug("Format string: %s", MODEL_DEVI_CONF_FMT)
    logging.debug("System index: %d", sys_idx)
    logging.debug("Configuration index: %d", conf_idx)

    conf_name = MODEL_DEVI_CONF_FMT % (sys_idx, conf_idx)

    logging.debug("Generated conf name: %s", conf_name)
    return conf_name


def make_fp_task_name(sys_idx, co_check_empty_iterunter):
    return "task." + fp_task_fmt % (sys_idx, counter)


def get_sys_index(task):
    task.sort()
    system_index = []
    for ii in task:
        system_index.append(os.path.basename(ii).split(".")[1])
    set_tmp = set(system_index)
    system_index = list(set_tmp)
    system_index.sort()
    return system_index


def check_empty_iter(iter_index, max_v=0):
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


def copy_model(numb_model, prv_iter_index, cur_iter_index):
    cwd = os.getcwd()
    prv_train_path = os.path.join(make_iter_name(prv_iter_index), train_name)
    cur_train_path = os.path.join(make_iter_name(cur_iter_index), train_name)
    prv_train_path = os.path.abspath(prv_train_path)
    cur_train_path = os.path.abspath(cur_train_path)
    create_path(cur_train_path)
    for ii in range(numb_model):
        prv_train_task = os.path.join(prv_train_path, train_task_fmt % ii)
        os.chdir(cur_train_path)
        os.symlink(os.path.relpath(prv_train_task), train_task_fmt % ii)
        os.symlink(
            os.path.join(train_task_fmt % ii, "frozen_model.pb"), "graph.%03d.pb" % ii
        )
        os.chdir(cwd)
    with open(os.path.join(cur_train_path, "copied"), "w") as fp:
        None


def poscar_natoms(lines):
    numb_atoms = 0
    for ii in lines[6].split():
        numb_atoms += int(ii)
    return numb_atoms


def poscar_shuffle(poscar_in, poscar_out):
    with open(poscar_in, "r") as fin:
        lines = list(fin)
    numb_atoms = poscar_natoms(lines)
    idx = np.arange(8, 8 + numb_atoms)
    np.random.shuffle(idx)
    out_lines = lines[0:8]
    for ii in range(numb_atoms):
        out_lines.append(lines[idx[ii]])
    with open(poscar_out, "w") as fout:
        fout.write("".join(out_lines))


def expand_idx(in_list):
    ret = []
    for ii in in_list:
        if type(ii) == int:
            ret.append(ii)
        elif type(ii) == str:
            step_str = ii.split(":")
            if len(step_str) > 1:
                step = int(step_str[1])
            else:
                step = 1
            range_str = step_str[0].split("-")
            assert (len(range_str)) == 2
            ret += range(int(range_str[0]), int(range_str[1]), step)
    return ret


def _check_skip_train(job):
    try:
        skip = _get_param_alias(job, ["s_t", "sk_tr", "skip_train", "skip_training"])
    except ValueError:
        skip = False
    return skip


def poscar_to_conf(poscar, conf):
    sys = dpdata.System(poscar, fmt="vasp/poscar")
    sys.to_lammps_lmp(conf)


# def dump_to_poscar(dump, poscar, type_map, fmt = "lammps/dump") :
#    sys = dpdata.System(dump, fmt = fmt, type_map = type_map)
#    sys.to_vasp_poscar(poscar)


def dump_to_deepmd_raw(dump, deepmd_raw, type_map, fmt="gromacs/gro", charge=None):
    system = dpdata.System(dump, fmt=fmt, type_map=type_map)
    system.to_deepmd_raw(deepmd_raw)
    if charge is not None:
        with open(os.path.join(deepmd_raw, "charge"), "w") as f:
            f.write(str(charge))


def revise_lmp_input_model(lmp_lines, task_model_list, trj_freq, deepmd_version="1"):
    idx = find_only_one_key(lmp_lines, ["pair_style", "deepmd"])
    graph_list = " ".join(task_model_list)
    if Version(deepmd_version) < Version("1"):
        lmp_lines[idx] = "pair_style      deepmd %s %d model_devi.out\n" % (
            graph_list,
            trj_freq,
        )
    else:
        lmp_lines[
            idx
        ] = "pair_style      deepmd %s out_freq %d out_file model_devi.out\n" % (
            graph_list,
            trj_freq,
        )
    return lmp_lines


def revise_lmp_input_dump(lmp_lines, trj_freq, model_devi_merge_traj=False):
    idx = find_only_one_key(lmp_lines, ["dump", "dpgen_dump"])
    if model_devi_merge_traj:
        lmp_lines[idx] = (
            "dump            dpgen_dump all custom %d    all.lammpstrj id type x y z\n"
            % trj_freq
        )
    else:
        lmp_lines[idx] = (
            "dump            dpgen_dump all custom %d traj/*.lammpstrj id type x y z\n"
            % trj_freq
        )

    return lmp_lines


def revise_lmp_input_plm(lmp_lines, in_plm, out_plm="output.plumed"):
    idx = find_only_one_key(lmp_lines, ["fix", "dpgen_plm"])
    lmp_lines[
        idx
    ] = "fix            dpgen_plm all plumed plumedfile %s outfile %s\n" % (
        in_plm,
        out_plm,
    )
    return lmp_lines


def revise_by_keys(lmp_lines, keys, values):
    for kk, vv in zip(keys, values):
        for ii in range(len(lmp_lines)):
            lmp_lines[ii] = lmp_lines[ii].replace(kk, str(vv))
    return lmp_lines


def _get_param_alias(jdata, names):
    for ii in names:
        if ii in jdata:
            return jdata[ii]
    raise ValueError(
        "one of the keys %s should be in jdata %s"
        % (str(names), (json.dumps(jdata, indent=4)))
    )


def update_mass_map(jdata):
    if jdata["mass_map"] == "MLFFOps":
        jdata["mass_map"] = [get_atomic_masses(i) for i in jdata["type_map"]]


def get_atomic_masses(atom):
    element_names = [
        "Hydrogen",
        "Helium",
        "Lithium",
        "Beryllium",
        "Boron",
        "Carbon",
        "Nitrogen",
        "Oxygen",
        "Fluorine",
        "Neon",
        "Sodium",
        "Magnesium",
        "Aluminium",
        "Silicon",
        "Phosphorus",
        "Sulfur",
        "Chlorine",
        "Argon",
        "Potassium",
        "Calcium",
        "Scandium",
        "Titanium",
        "Vanadium",
        "Chromium",
        "Manganese",
        "Iron",
        "Cobalt",
        "Nickel",
        "Copper",
        "Zinc",
        "Gallium",
        "Germanium",
        "Arsenic",
        "Selenium",
        "Bromine",
        "Krypton",
        "Rubidium",
        "Strontium",
        "Yttrium",
        "Zirconium",
        "Niobium",
        "Molybdenum",
        "Technetium",
        "Ruthenium",
        "Rhodium",
        "Palladium",
        "Silver",
        "Cadmium",
        "Indium",
        "Tin",
        "Antimony",
        "Tellurium",
        "Iodine",
        "Xenon",
        "Caesium",
        "Barium",
        "Lanthanum",
        "Cerium",
        "Praseodymium",
        "Neodymium",
        "Promethium",
        "Samarium",
        "Europium",
        "Gadolinium",
        "Terbium",
        "Dysprosium",
        "Holmium",
        "Erbium",
        "Thulium",
        "Ytterbium",
        "Lutetium",
        "Hafnium",
        "Tantalum",
        "Tungsten",
        "Rhenium",
        "Osmium",
        "Iridium",
        "Platinum",
        "Gold",
        "Mercury",
        "Thallium",
        "Lead",
        "Bismuth",
        "Polonium",
        "Astatine",
        "Radon",
        "Francium",
        "Radium",
        "Actinium",
        "Thorium",
        "Protactinium",
        "Uranium",
        "Neptunium",
        "Plutonium",
        "Americium",
        "Curium",
        "Berkelium",
        "Californium",
        "Einsteinium",
        "Fermium",
        "Mendelevium",
        "Nobelium",
        "Lawrencium",
        "Rutherfordium",
        "Dubnium",
        "Seaborgium",
        "Bohrium",
        "Hassium",
        "Meitnerium",
        "Darmastadtium",
        "Roentgenium",
        "Copernicium",
        "Nihonium",
        "Flerovium",
        "Moscovium",
        "Livermorium",
        "Tennessine",
        "Oganesson",
    ]
    chemical_symbols = [
        "H",
        "He",
        "Li",
        "Be",
        "B",
        "C",
        "N",
        "O",
        "F",
        "Ne",
        "Na",
        "Mg",
        "Al",
        "Si",
        "P",
        "S",
        "Cl",
        "Ar",
        "K",
        "Ca",
        "Sc",
        "Ti",
        "V",
        "Cr",
        "Mn",
        "Fe",
        "Co",
        "Ni",
        "Cu",
        "Zn",
        "Ga",
        "Ge",
        "As",
        "Se",
        "Br",
        "Kr",
        "Rb",
        "Sr",
        "Y",
        "Zr",
        "Nb",
        "Mo",
        "Tc",
        "Ru",
        "Rh",
        "Pd",
        "Ag",
        "Cd",
        "In",
        "Sn",
        "Sb",
        "Te",
        "I",
        "Xe",
        "Cs",
        "Ba",
        "La",
        "Ce",
        "Pr",
        "Nd",
        "Pm",
        "Sm",
        "Eu",
        "Gd",
        "Tb",
        "Dy",
        "Ho",
        "Er",
        "Tm",
        "Yb",
        "Lu",
        "Hf",
        "Ta",
        "W",
        "Re",
        "Os",
        "Ir",
        "Pt",
        "Au",
        "Hg",
        "Tl",
        "Pb",
        "Bi",
        "Po",
        "At",
        "Rn",
        "Fr",
        "Ra",
        "Ac",
        "Th",
        "Pa",
        "U",
        "Np",
        "Pu",
        "Am",
        "Cm",
        "Bk",
        "Cf",
        "Es",
        "Fm",
        "Md",
        "No",
        "Lr",
        "Rf",
        "Db",
        "Sg",
        "Bh",
        "Hs",
        "Mt",
        "Ds",
        "Rg",
        "Cn",
        "Nh",
        "Fl",
        "Mc",
        "Lv",
        "Ts",
        "Og",
    ]
    atomic_number = [i + 1 for i in range(len(chemical_symbols))]

    # NIST Standard Reference Database 144
    # URL: https://physics.nist.gov/cgi-bin/Compositions/stand_alone.pl?ele=&ascii=ascii&isotype=all
    atomic_masses_common = [
        1.00782503223,
        4.00260325413,
        7.0160034366,
        9.012183065,
        11.00930536,
        12.0,
        14.00307400443,
        15.99491461957,
        18.99840316273,
        19.9924401762,
        22.989769282,
        23.985041697,
        26.98153853,
        27.97692653465,
        30.97376199842,
        31.9720711744,
        34.968852682,
        39.9623831237,
        38.9637064864,
        39.962590863,
        44.95590828,
        47.94794198,
        50.94395704,
        51.94050623,
        54.93804391,
        55.93493633,
        58.93319429,
        57.93534241,
        62.92959772,
        63.92914201,
        68.9255735,
        73.921177761,
        74.92159457,
        79.9165218,
        78.9183376,
        83.9114977282,
        84.9117897379,
        87.9056125,
        88.9058403,
        89.9046977,
        92.906373,
        97.90540482,
        96.9063667,
        101.9043441,
        102.905498,
        105.9034804,
        106.9050916,
        113.90336509,
        114.903878776,
        119.90220163,
        120.903812,
        129.906222748,
        126.9044719,
        131.9041550856,
        132.905451961,
        137.905247,
        138.9063563,
        139.9054431,
        140.9076576,
        141.907729,
        144.9127559,
        151.9197397,
        152.921238,
        157.9241123,
        158.9253547,
        163.9291819,
        164.9303288,
        165.9302995,
        168.9342179,
        173.9388664,
        174.9407752,
        179.946557,
        180.9479958,
        183.95093092,
        186.9557501,
        191.961477,
        192.9629216,
        194.9647917,
        196.96656879,
        201.9706434,
        204.9744278,
        207.9766525,
        208.9803991,
        208.9824308,
        209.9871479,
        222.0175782,
        223.019736,
        226.0254103,
        227.0277523,
        232.0380558,
        231.0358842,
        238.0507884,
        237.0481736,
        244.0642053,
        243.0613813,
        247.0703541,
        247.0703073,
        251.0795886,
        252.08298,
        257.0951061,
        258.0984315,
        259.10103,
        262.10961,
        267.12179,
        268.12567,
        271.13393,
        272.13826,
        270.13429,
        276.15159,
        281.16451,
        280.16514,
        285.17712,
        284.17873,
        289.19042,
        288.19274,
        293.20449,
        292.20746,
        294.21392,
    ]
    # IUPAC Technical Report
    # doi:10.1515/pac-2015-0305
    atomic_masses_2013 = [
        1.00784,
        4.002602,
        6.938,
        9.0121831,
        10.806,
        12.0096,
        14.00643,
        15.99903,
        18.99840316,
        20.1797,
        22.98976928,
        24.304,
        26.9815385,
        28.084,
        30.973762,
        32.059,
        35.446,
        39.948,
        39.0983,
        40.078,
        44.955908,
        47.867,
        50.9415,
        51.9961,
        54.938044,
        55.845,
        58.933194,
        58.6934,
        63.546,
        65.38,
        69.723,
        72.63,
        74.921595,
        78.971,
        79.901,
        83.798,
        85.4678,
        87.62,
        88.90584,
        91.224,
        92.90637,
        95.95,
        None,
        101.07,
        102.9055,
        106.42,
        107.8682,
        112.414,
        114.818,
        118.71,
        121.76,
        127.6,
        126.90447,
        131.293,
        132.905452,
        137.327,
        138.90547,
        140.116,
        140.90766,
        144.242,
        None,
        150.36,
        151.964,
        157.25,
        158.92535,
        162.5,
        164.93033,
        167.259,
        168.93422,
        173.054,
        174.9668,
        178.49,
        180.94788,
        183.84,
        186.207,
        190.23,
        192.217,
        195.084,
        196.966569,
        200.592,
        204.382,
        207.2,
        208.9804,
        None,
        None,
        None,
        None,
        None,
        None,
        232.0377,
        231.03588,
        238.02891,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
    ]
    # IUPAC Technical Report
    # doi:10.1515/pac-2019-0603
    atomic_masses_2021 = [
        1.00784,
        4.002602,
        6.938,
        9.0121831,
        10.806,
        12.0096,
        14.00643,
        15.99903,
        18.99840316,
        20.1797,
        22.98976928,
        24.304,
        26.9815384,
        28.084,
        30.973762,
        32.059,
        35.446,
        39.792,
        39.0983,
        40.078,
        44.955907,
        47.867,
        50.9415,
        51.9961,
        54.938043,
        55.845,
        58.933194,
        58.6934,
        63.546,
        65.38,
        69.723,
        72.63,
        74.921595,
        78.971,
        79.901,
        83.798,
        85.4678,
        87.62,
        88.905838,
        91.224,
        92.90637,
        95.95,
        None,
        101.07,
        102.90549,
        106.42,
        107.8682,
        112.414,
        114.818,
        118.71,
        121.76,
        127.6,
        126.90447,
        131.293,
        132.905452,
        137.327,
        138.90547,
        140.116,
        140.90766,
        144.242,
        None,
        150.36,
        151.964,
        157.25,
        158.925354,
        162.5,
        164.930329,
        167.259,
        168.934219,
        173.045,
        174.9668,
        178.486,
        180.94788,
        183.84,
        186.207,
        190.23,
        192.217,
        195.084,
        196.96657,
        200.592,
        204.382,
        206.14,
        208.9804,
        None,
        None,
        None,
        None,
        None,
        None,
        232.0377,
        231.03588,
        238.02891,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
    ]

    atomic_masses = [
        atomic_masses_common[n] if i is None else i
        for n, i in enumerate(atomic_masses_2021)
    ]

    if atom in element_names:
        return atomic_masses[element_names.index(atom)]
    elif atom in chemical_symbols:
        return atomic_masses[chemical_symbols.index(atom)]
    elif atom in atomic_number:
        return atomic_masses[atomic_number.index(atom)]
    else:
        raise RuntimeError(
            "unknown atomic identifier",
            atom,
            'if one want to use isotopes, or non-standard element names, chemical symbols, or atomic number in the type_map list, please customize the mass_map list instead of using "MLFFOps".',
        )
