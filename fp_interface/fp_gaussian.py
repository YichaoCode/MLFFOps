# fp_gaussian.py


def make_fp_gaussian(iter_index, jdata):
    # make config
    fp_tasks = _make_fp_vasp_configs(iter_index, jdata)
    if len(fp_tasks) == 0:
        return
    # make gaussian gjf file
    iter_name = make_iter_name(iter_index)
    work_path = os.path.join(iter_name, fp_name)
    if "user_fp_params" in jdata.keys():
        fp_params = jdata["user_fp_params"]
    else:
        fp_params = jdata["fp_params"]
    cwd = os.getcwd()

    model_devi_engine = jdata.get("model_devi_engine", "lammps")
    for ii in fp_tasks:
        os.chdir(ii)
        if (model_devi_engine == "lammps") or (model_devi_engine == "dimer"):
            sys_data = dpdata.System("POSCAR").data
        elif model_devi_engine == "gromacs":
            sys_data = dpdata.System("deepmd.raw", fmt="deepmd/raw").data
            if os.path.isfile("deepmd.raw/charge"):
                sys_data["charge"] = int(np.loadtxt("deepmd.raw/charge", dtype=int))
        ret = make_gaussian_input(sys_data, fp_params)
        with open("input", "w") as fp:
            fp.write(ret)
        os.chdir(cwd)



def _gaussian_check_fin(ii):
    if os.path.isfile(os.path.join(ii, "output")):
        with open(os.path.join(ii, "output"), "r") as fp:
            content = fp.read()
            count = content.count("termination")
            if count == 0:
                return False
    else:
        return False
    return True



def post_fp_gaussian(iter_index, jdata):
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
        sys_output.sort()
        for idx, oo in enumerate(sys_output):
            sys = dpdata.LabeledSystem(oo, fmt="gaussian/log")
            if len(sys) > 0:
                sys.check_type_map(type_map=jdata["type_map"])
            if jdata.get("use_atom_pref", False):
                sys.data["atom_pref"] = np.load(
                    os.path.join(os.path.dirname(oo), "atom_pref.npy")
                )
            if idx == 0:
                if jdata.get("use_clusters", False):
                    all_sys = dpdata.MultiSystems(sys, type_map=jdata["type_map"])
                else:
                    all_sys = sys
            else:
                all_sys.append(sys)
        sys_data_path = os.path.join(work_path, "data.%s" % ss)
        all_sys.to_deepmd_raw(sys_data_path)
        all_sys.to_deepmd_npy(sys_data_path, set_size=len(sys_output))
