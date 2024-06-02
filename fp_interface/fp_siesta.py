# fp_siesta.py

def make_fp_siesta(iter_index, jdata):
    # make config
    fp_tasks = _make_fp_vasp_configs(iter_index, jdata)
    if len(fp_tasks) == 0:
        return
    # make siesta input
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
        ret = make_siesta_input(sys_data, fp_pp_files, fp_params)
        with open("input", "w") as fp:
            fp.write(ret)
        os.chdir(cwd)
    # link pp files
    _link_fp_vasp_pp(iter_index, jdata)



def _siesta_check_fin(ii):
    if os.path.isfile(os.path.join(ii, "output")):
        with open(os.path.join(ii, "output"), "r") as fp:
            content = fp.read()
            count = content.count("End of run")
            if count != 1:
                return False
    else:
        return False
    return True



def post_fp_siesta(iter_index, jdata):
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
        for idx, oo in enumerate(sys_output):
            _sys = dpdata.LabeledSystem()
            (
                _sys.data["atom_names"],
                _sys.data["atom_numbs"],
                _sys.data["atom_types"],
                _sys.data["cells"],
                _sys.data["coords"],
                _sys.data["energies"],
                _sys.data["forces"],
                _sys.data["virials"],
            ) = dpdata.siesta.output.obtain_frame(oo)
            if idx == 0:
                all_sys = _sys
            else:
                all_sys.append(_sys)

        sys_data_path = os.path.join(work_path, "data.%s" % ss)
        all_sys.to_deepmd_raw(sys_data_path)
        all_sys.to_deepmd_npy(sys_data_path, set_size=len(sys_output))
