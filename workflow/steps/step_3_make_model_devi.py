# step_3_make_model_devi.py
# Created by Yichao


# region
import os
import glob
import json
import dpdata

from dpgen.generator.lib.utils import (
    create_path,
    make_iter_name,
    symlink_user_forward_files
)

from workflow.lammps_run import run_lammps


from model_devi import (
    _make_model_devi_native, _make_model_devi_native_gromacs,
    _make_model_devi_amber, _make_model_devi_revmat
)

from config.config import *


from utils.utils import (
    expand_idx,
    make_model_devi_conf_name
)

# endregion


def make_model_devi(iter_index, jdata, mdata):
    # The MD engine to perform model deviation
    # Default is lammps
    model_devi_engine = jdata.get("model_devi_engine", "lammps")

    model_devi_jobs = jdata["model_devi_jobs"]
    if model_devi_engine != "calypso":
        if iter_index >= len(model_devi_jobs):
            return False
    else:
        # mode 1: generate structures according to the user-provided input.dat file, so calypso_input_path and model_devi_max_iter are needed
        run_mode = 1
        if "calypso_input_path" in jdata:
            try:
                maxiter = jdata.get("model_devi_max_iter")
            except KeyError:
                raise KeyError(
                    "calypso_input_path key exists so you should provide model_devi_max_iter key to control the max iter number"
                )
        # mode 2: control each iteration to generate structures in specific way by providing model_devi_jobs key
        else:
            try:
                maxiter = max(model_devi_jobs[-1].get("times"))
                run_mode = 2
            except KeyError:
                raise KeyError('did not find model_devi_jobs["times"] key')
        if iter_index > maxiter:
            dlog.info(f"iter_index is {iter_index} and maxiter is {maxiter}")
            return False

    if "sys_configs_prefix" in jdata:
        sys_configs = []
        for sys_list in jdata["sys_configs"]:
            # assert (isinstance(sys_list, list) ), "Currently only support type list for sys in 'sys_conifgs' "
            temp_sys_list = [
                os.path.join(jdata["sys_configs_prefix"], sys) for sys in sys_list
            ]
            sys_configs.append(temp_sys_list)
    else:
        sys_configs = jdata["sys_configs"]
    shuffle_poscar = jdata.get("shuffle_poscar", False)

    if model_devi_engine != "calypso":
        cur_job = model_devi_jobs[iter_index]
        sys_idx = expand_idx(cur_job["sys_idx"])
    else:
        cur_job = {"model_devi_engine": "calypso", "input.dat": "user_provided"}
        sys_idx = []

    if len(sys_idx) != len(list(set(sys_idx))):
        raise RuntimeError("system index should be uniq")
    conf_systems = []
    for idx in sys_idx:
        cur_systems = []
        ss = sys_configs[idx]
        for ii in ss:
            ii_systems = sorted(glob.glob(ii))
            if ii_systems == []:
                warnings.warn(
                    "There is no system in the path %s. Please check if the path is correct."
                    % ii
                )
            cur_systems += ii_systems
        # cur_systems should not be sorted, as we may add specific constrict to the similutions
        # cur_systems.sort()
        cur_systems = [os.path.abspath(ii) for ii in cur_systems]
        conf_systems.append(cur_systems)

    iter_name = make_iter_name(iter_index)
    train_path = os.path.join(iter_name, TRAIN_NAME)
    train_path = os.path.abspath(train_path)
    models = sorted(glob.glob(os.path.join(train_path, "graph*pb")))
    work_path = os.path.join(iter_name, model_devi_name)
    create_path(work_path)
    if model_devi_engine == "calypso":
        _calypso_run_opt_path = os.path.join(work_path, calypso_run_opt_name)
        calypso_model_devi_path = os.path.join(work_path, calypso_model_devi_name)
        create_path(calypso_model_devi_path)
        # run model devi script
        calypso_run_model_devi_script = os.path.join(
            calypso_model_devi_path, "calypso_run_model_devi.py"
        )
        shutil.copyfile(calypso_run_model_devi_file, calypso_run_model_devi_script)
        # Create work path list
        calypso_run_opt_path = []

        # mode 1: generate structures according to the user-provided input.dat file,
        # so calypso_input_path and model_devi_max_iter are needed
        if run_mode == 1:
            if jdata.get("vsc", False) and len(jdata.get("type_map")) > 1:
                # [input.dat.Li.250, input.dat.Li.300]
                one_ele_inputdat_list = glob.glob(
                    f"{jdata.get('calypso_input_path')}/input.dat.{jdata.get('type_map')[0]}.*"
                )
                if len(one_ele_inputdat_list) == 0:
                    number_of_pressure = 1
                else:
                    number_of_pressure = len(list(set(one_ele_inputdat_list)))

                # calypso_run_opt_path = ['gen_struc_analy.000','gen_struc_analy.001']
                for temp_idx in range(number_of_pressure):
                    calypso_run_opt_path.append(
                        "%s.%03d" % (_calypso_run_opt_path, temp_idx)
                    )
            elif not jdata.get("vsc", False):
                calypso_run_opt_path.append("%s.%03d" % (_calypso_run_opt_path, 0))

        # mode 2: control each iteration to generate structures in specific way
        # by providing model_devi_jobs key
        elif run_mode == 2:
            for iiidx, jobbs in enumerate(model_devi_jobs):
                if iter_index in jobbs.get("times"):
                    cur_job = model_devi_jobs[iiidx]

            pressures_list = cur_job.get("PSTRESS", [0.0001])
            for temp_idx in range(len(pressures_list)):
                calypso_run_opt_path.append(
                    "%s.%03d" % (_calypso_run_opt_path, temp_idx)
                )
        # to different directory
        # calypso_run_opt_path = ['gen_struc_analy.000','gen_struc_analy.001','gen_struc_analy.002',]
        for temp_calypso_run_opt_path in calypso_run_opt_path:
            create_path(temp_calypso_run_opt_path)
            # run confs opt script
            run_opt_script = os.path.join(
                temp_calypso_run_opt_path, "calypso_run_opt.py"
            )
            shutil.copyfile(run_opt_file, run_opt_script)
            # check outcar script
            check_outcar_script = os.path.join(
                temp_calypso_run_opt_path, "check_outcar.py"
            )
            shutil.copyfile(check_outcar_file, check_outcar_script)

    for mm in models:
        model_name = os.path.basename(mm)
        if model_devi_engine != "calypso":
            os.symlink(mm, os.path.join(work_path, model_name))
        else:
            for temp_calypso_run_opt_path in calypso_run_opt_path:
                models_path = os.path.join(temp_calypso_run_opt_path, model_name)
                if not os.path.exists(models_path):
                    os.symlink(mm, models_path)

    with open(os.path.join(work_path, "cur_job.json"), "w") as outfile:
        json.dump(cur_job, outfile, indent=4)

    conf_path = os.path.join(work_path, "confs")
    create_path(conf_path)
    sys_counter = 0
    for ss in conf_systems:
        conf_counter = 0
        for cc in ss:
            if (model_devi_engine == "lammps") or (model_devi_engine == "dimer"):
                conf_name = make_model_devi_conf_name(
                    sys_idx[sys_counter], conf_counter
                )
                orig_poscar_name = conf_name + ".orig.poscar"
                poscar_name = conf_name + ".poscar"
                lmp_name = conf_name + ".lmp"
                if shuffle_poscar:
                    os.symlink(cc, os.path.join(conf_path, orig_poscar_name))
                    poscar_shuffle(
                        os.path.join(conf_path, orig_poscar_name),
                        os.path.join(conf_path, poscar_name),
                    )
                else:
                    os.symlink(cc, os.path.join(conf_path, poscar_name))
                if "sys_format" in jdata:
                    fmt = jdata["sys_format"]
                else:
                    fmt = "vasp/poscar"
                system = dpdata.System(
                    os.path.join(conf_path, poscar_name),
                    fmt=fmt,
                    type_map=jdata["type_map"],
                )
                if jdata.get("model_devi_nopbc", False):
                    system.remove_pbc()
                system.to_lammps_lmp(os.path.join(conf_path, lmp_name))
            elif model_devi_engine == "gromacs":
                pass
            elif model_devi_engine == "amber":
                # Jinzhe's specific Amber version
                conf_name = make_model_devi_conf_name(
                    sys_idx[sys_counter], conf_counter
                )
                rst7_name = conf_name + ".rst7"
                # link restart file
                os.symlink(cc, os.path.join(conf_path, rst7_name))
            conf_counter += 1
        sys_counter += 1

    input_mode = "native"
    if "calypso_input_path" in jdata:
        input_mode = "buffet"
    if "template" in cur_job:
        input_mode = "revise_template"
    use_plm = jdata.get("model_devi_plumed", False)
    use_plm_path = jdata.get("model_devi_plumed_path", False)
    if input_mode == "native":
        if (model_devi_engine == "lammps") or (model_devi_engine == "dimer"):
            _make_model_devi_native(iter_index, jdata, mdata, conf_systems)
        elif model_devi_engine == "gromacs":
            _make_model_devi_native_gromacs(iter_index, jdata, mdata, conf_systems)
        elif model_devi_engine == "amber":
            _make_model_devi_amber(iter_index, jdata, mdata, conf_systems)
        elif model_devi_engine == "calypso":
            _make_model_devi_native_calypso(
                iter_index, model_devi_jobs, calypso_run_opt_path
            )  # generate input.dat automatic in each iter
        else:
            raise RuntimeError("unknown model_devi engine", model_devi_engine)
    elif input_mode == "revise_template":
        _make_model_devi_revmat(iter_index, jdata, mdata, conf_systems)
    elif input_mode == "buffet":
        _make_model_devi_buffet(
            jdata, calypso_run_opt_path
        )  # generate confs according to the input.dat provided
    else:
        raise RuntimeError("unknown model_devi input mode", input_mode)
    # Copy user defined forward_files
    symlink_user_forward_files(mdata=mdata, task_type="model_devi", work_path=work_path)
    return True
