# fp_pwmat.py



def make_pwmat_input(jdata, filename):
    if "fp_incar" in jdata.keys():
        fp_incar_path = jdata["fp_incar"]
        assert os.path.exists(fp_incar_path)
        fp_incar_path = os.path.abspath(fp_incar_path)
        fr = open(fp_incar_path)
        input = fr.read()
        fr.close()
    elif "user_fp_params" in jdata.keys():
        fp_params = jdata["user_fp_params"]
        node1 = fp_params["node1"]
        node2 = fp_params["node2"]
        atom_config = fp_params["in.atom"]
        ecut = fp_params["ecut"]
        e_error = fp_params["e_error"]
        rho_error = fp_params["rho_error"]
        kspacing = fp_params["kspacing"]
        flag_symm = fp_params["flag_symm"]
        os.system("command -v poscar2config.x | wc -l > 1.txt")
        fc = open("1.txt")
        flag_command = fc.read()
        fc.close()
        if int(flag_command) == 1:
            os.system("poscar2config.x < POSCAR > tmp.config")
        else:
            os.system(
                "cp ../../../out_data_post_fp_pwmat/02.fp/task.000.000000/poscar2config.x ./"
            )
            os.system("./poscar2config.x < POSCAR > tmp.config")
        os.system("rm -rf tmp.config")
        input_dict = make_pwmat_input_dict(
            node1,
            node2,
            atom_config,
            ecut,
            e_error,
            rho_error,
            icmix=None,
            smearing=None,
            sigma=None,
            kspacing=kspacing,
            flag_symm=flag_symm,
        )

        input = write_input_dict(input_dict)
    else:
        input = make_pwmat_input_user_dict(jdata["fp_params"])
    if "IN.PSP" in input or "in.psp" in input:
        with open(filename, "w") as fp:
            fp.write(input)
            fp.write("job=scf\n")
            if "OUT.MLMD" in input or "out.mlmd" in input:
                return input
            else:
                fp.write("OUT.MLMD = T")
                return input
    else:
        with open(filename, "w") as fp:
            fp.write(input)
            fp.write("job=scf\n")
            fp_pp_files = jdata["fp_pp_files"]
            for idx, ii in enumerate(fp_pp_files):
                fp.write("IN.PSP%d = %s\n" % (idx + 1, ii))
            if "OUT.MLMD" in input or "out.mlmd" in input:
                return input
            else:
                fp.write("OUT.MLMD = T")
                return input



def _make_fp_pwmat_input(iter_index, jdata):
    iter_name = make_iter_name(iter_index)
    work_path = os.path.join(iter_name, fp_name)
    fp_tasks = glob.glob(os.path.join(work_path, "task.*"))
    fp_tasks.sort()
    if len(fp_tasks) == 0:
        return
    cwd = os.getcwd()
    for ii in fp_tasks:
        os.chdir(ii)
        make_pwmat_input(jdata, "etot.input")
        os.system("sed -i '1,2c 4 1' etot.input")
        os.chdir(cwd)



def make_fp_pwmat(iter_index, jdata):
    # make config
    fp_tasks = _make_fp_vasp_configs(iter_index, jdata)
    if len(fp_tasks) == 0:
        return
    # abs path for fp_incar if it exists
    if "fp_incar" in jdata:
        jdata["fp_incar"] = os.path.abspath(jdata["fp_incar"])
    # order is critical!
    # 1, link pp files
    _link_fp_vasp_pp(iter_index, jdata)
    # 2, create pwmat input
    _make_fp_pwmat_input(iter_index, jdata)



def _pwmat_check_fin(ii):
    if os.path.isfile(os.path.join(ii, "REPORT")):
        with open(os.path.join(ii, "REPORT"), "r") as fp:
            content = fp.read()
            count = content.count("time")
            if count != 1:
                return False
    else:
        return False
    return True



def post_fp_pwmat(iter_index, jdata, rfailed=None):

    ratio_failed = rfailed if rfailed else jdata.get("ratio_failed", 0.05)
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

    tcount = 0
    icount = 0
    for ss in system_index:
        sys_output = glob.glob(os.path.join(work_path, "task.%s.*/OUT.MLMD" % ss))
        sys_output.sort()
        tcount += len(sys_output)
        all_sys = None
        for oo in sys_output:
            _sys = dpdata.LabeledSystem(oo, type_map=jdata["type_map"])
            if len(_sys) == 1:
                if all_sys is None:
                    all_sys = _sys
                else:
                    all_sys.append(_sys)
            else:
                icount += 1
        if all_sys is not None:
            sys_data_path = os.path.join(work_path, "data.%s" % ss)
            all_sys.to_deepmd_raw(sys_data_path)
            all_sys.to_deepmd_npy(sys_data_path, set_size=len(sys_output))
    dlog.info("failed frame number: %s " % icount)
    dlog.info("total frame number: %s " % tcount)
    reff = icount / tcount
    dlog.info("ratio of failed frame:  {:.2%}".format(reff))

    if reff > ratio_failed:
        raise RuntimeError("find too many unsuccessfully terminated jobs")
