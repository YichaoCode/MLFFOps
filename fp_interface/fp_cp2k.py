# fp_cp2k.py


def make_fp_cp2k(iter_index, jdata):
    # make config
    fp_tasks = _make_fp_vasp_configs(iter_index, jdata)
    if len(fp_tasks) == 0:
        return
    # make cp2k input
    iter_name = make_iter_name(iter_index)
    work_path = os.path.join(iter_name, fp_name)
    if "user_fp_params" in jdata.keys():
        fp_params = jdata["user_fp_params"]
    # some users might use own inputs
    # specify the input path string
    elif "external_input_path" in jdata.keys():
        fp_params = None
        exinput_path = os.path.abspath(jdata["external_input_path"])
    else:
        fp_params = jdata["fp_params"]
    cwd = os.getcwd()
    for ii in fp_tasks:
        os.chdir(ii)
        sys_data = dpdata.System("POSCAR").data
        # make input for every task
        # if fp_params exits, make keys
        if fp_params:
            cp2k_input = make_cp2k_input(sys_data, fp_params)
        else:
            # else read from user input
            cp2k_input = make_cp2k_input_from_external(sys_data, exinput_path)
        with open("input.inp", "w") as fp:
            fp.write(cp2k_input)
            fp.close()
        # make coord.xyz used by cp2k for every task
        cp2k_coord = make_cp2k_xyz(sys_data)
        with open("coord.xyz", "w") as fp:
            fp.write(cp2k_coord)
            fp.close()
        os.chdir(cwd)

    # link pp files
    _link_fp_vasp_pp(iter_index, jdata)



def _cp2k_check_fin(ii):
    if os.path.isfile(os.path.join(ii, "output")):
        with open(os.path.join(ii, "output"), "r") as fp:
            content = fp.read()
            count = content.count("SCF run converged")
            if count == 0:
                return False
    else:
        return False
    return True



def post_fp_cp2k(iter_index, jdata, rfailed=None):

    ratio_failed = rfailed if rfailed else jdata.get("ratio_failed", 0.10)
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
    # tcount: num of all fp tasks
    tcount = 0
    # icount: num of converged fp tasks
    icount = 0
    for ss in system_index:
        sys_output = glob.glob(os.path.join(work_path, "task.%s.*/output" % ss))
        sys_output.sort()
        tcount += len(sys_output)
        all_sys = None
        for oo in sys_output:
            _sys = dpdata.LabeledSystem(oo, fmt="cp2k/output")
            # _sys.check_type_map(type_map = jdata['type_map'])
            if all_sys is None:
                all_sys = _sys
            else:
                all_sys.append(_sys)

        icount += len(all_sys)
        if all_sys is not None:
            sys_data_path = os.path.join(work_path, "data.%s" % ss)
            all_sys.to_deepmd_raw(sys_data_path)
            all_sys.to_deepmd_npy(sys_data_path, set_size=len(sys_output))

    if tcount == 0:
        rfail = 0.0
        dlog.info("failed frame: %6d in %6d " % (tcount - icount, tcount))
    else:
        rfail = float(tcount - icount) / float(tcount)
        dlog.info(
            "failed frame: %6d in %6d  %6.2f %% "
            % (tcount - icount, tcount, rfail * 100.0)
        )

    if rfail > ratio_failed:
        raise RuntimeError(
            "find too many unsuccessfully terminated jobs. Too many FP tasks are not converged. Please check your files in directories 'iter.*.*/02.fp/task.*.*/.'"
        )
