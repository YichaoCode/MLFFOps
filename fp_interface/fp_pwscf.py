# fp_pwscf.py


def make_fp_pwscf(iter_index, jdata):
    # make config
    fp_tasks = _make_fp_vasp_configs(iter_index, jdata)
    if len(fp_tasks) == 0:
        return
    # make pwscf input
    iter_name = make_iter_name(iter_index)
    work_path = os.path.join(iter_name, fp_name)
    fp_pp_files = jdata["fp_pp_files"]
    if "user_fp_params" in jdata.keys():
        fp_params = jdata["user_fp_params"]
        user_input = True
    else:
        fp_params = jdata["fp_params"]
        user_input = False
    cwd = os.getcwd()
    for ii in fp_tasks:
        os.chdir(ii)
        sys_data = dpdata.System("POSCAR").data
        sys_data["atom_masses"] = []
        pps = []
        for iii in sys_data["atom_names"]:
            sys_data["atom_masses"].append(
                jdata["mass_map"][jdata["type_map"].index(iii)]
            )
            pps.append(fp_pp_files[jdata["type_map"].index(iii)])
        ret = make_pwscf_input(sys_data, pps, fp_params, user_input=user_input)
        with open("input", "w") as fp:
            fp.write(ret)
        os.chdir(cwd)
    # link pp files
    _link_fp_vasp_pp(iter_index, jdata)



def post_fp_pwscf(iter_index, jdata):
    model_devi_jobs = jdata["model_devi_jobs"]
    assert iter_index < len(model_devi_jobs)

    iter_name = make_iter_name(iter_index)
    work_path = os.path.join(iter_name, fp_name)
    fp_tasks = glob.glob(os.path.join(work_path, "task.*"))
    fp_tasks.sort()
    if len(fp_tasks) == 0:
        return

    system_index = []
    for ii in fp_tasks:
        system_index.append(os.path.basename(ii).split(".")[1])
    system_index.sort()
    set_tmp = set(system_index)
    system_index = list(set_tmp)
    system_index.sort()

    cwd = os.getcwd()
    for ss in system_index:
        sys_output = glob.glob(os.path.join(work_path, "task.%s.*/output" % ss))
        sys_input = glob.glob(os.path.join(work_path, "task.%s.*/input" % ss))
        sys_output.sort()
        sys_input.sort()

        flag = True
        for ii, oo in zip(sys_input, sys_output):
            if flag:
                _sys = dpdata.LabeledSystem(
                    oo, fmt="qe/pw/scf", type_map=jdata["type_map"]
                )
                if len(_sys) > 0:
                    all_sys = _sys
                    flag = False
                else:
                    pass
            else:
                _sys = dpdata.LabeledSystem(
                    oo, fmt="qe/pw/scf", type_map=jdata["type_map"]
                )
                if len(_sys) > 0:
                    all_sys.append(_sys)

        sys_data_path = os.path.join(work_path, "data.%s" % ss)
        all_sys.to_deepmd_raw(sys_data_path)
        all_sys.to_deepmd_npy(sys_data_path, set_size=len(sys_output))
