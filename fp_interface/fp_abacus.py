# fp_abacus.py


def _link_fp_abacus_pporb_descript(iter_index, jdata):
    # assume pp orbital files, numerical descrptors and model for dpks are all in fp_pp_path.
    fp_pp_path = os.path.abspath(jdata["fp_pp_path"])

    iter_name = make_iter_name(iter_index)
    work_path = os.path.join(iter_name, fp_name)
    fp_tasks = glob.glob(os.path.join(work_path, "task.*"))
    fp_tasks.sort()
    if len(fp_tasks) == 0:
        return

    cwd = os.getcwd()
    for ii in fp_tasks:
        os.chdir(ii)

        # get value of 'deepks_model' from INPUT
        input_param = get_abacus_input_parameters("INPUT")
        fp_dpks_model = input_param.get("deepks_model", None)
        if fp_dpks_model != None:
            model_file = os.path.join(fp_pp_path, fp_dpks_model)
            assert os.path.isfile(model_file), (
                "Can not find the deepks model file %s, which is defined in %s/INPUT"
                % (model_file, ii)
            )
            os.symlink(model_file, fp_dpks_model)

        # get pp, orb, descriptor filenames from STRU
        stru_param = get_abacus_STRU("STRU")
        pp_files = stru_param.get("pp_files", [])
        orb_files = stru_param.get("orb_files", [])
        descriptor_file = stru_param.get("dpks_descriptor", None)
        pp_files = [] if pp_files == None else pp_files
        orb_files = [] if orb_files == None else orb_files

        for jj in pp_files:
            ifile = os.path.join(fp_pp_path, jj)
            assert os.path.isfile(ifile), (
                "Can not find the pseudopotential file %s, which is defined in %s/STRU"
                % (ifile, ii)
            )
            os.symlink(ifile, jj)

        for jj in orb_files:
            ifile = os.path.join(fp_pp_path, jj)
            assert os.path.isfile(
                ifile
            ), "Can not find the orbital file %s, which is defined in %s/STRU" % (
                ifile,
                ii,
            )
            os.symlink(ifile, jj)

        if descriptor_file != None:
            ifile = os.path.join(fp_pp_path, descriptor_file)
            assert os.path.isfile(ifile), (
                "Can not find the deepks descriptor file %s, which is defined in %s/STRU"
                % (ifile, ii)
            )
            os.symlink(ifile, descriptor_file)
        os.chdir(cwd)



def make_fp_abacus_scf(iter_index, jdata):
    # make config
    fp_tasks = _make_fp_vasp_configs(iter_index, jdata)
    if len(fp_tasks) == 0:
        return
    # make abacus/pw/scf input
    iter_name = make_iter_name(iter_index)
    work_path = os.path.join(iter_name, fp_name)
    fp_pp_files = jdata["fp_pp_files"]
    fp_orb_files = None
    fp_dpks_descriptor = None
    # get paramters for writting INPUT file
    fp_params = {}
    if "user_fp_params" in jdata.keys():
        fp_params = jdata["user_fp_params"]
    elif "fp_incar" in jdata.keys():
        fp_input_path = jdata["fp_incar"]
        assert os.path.exists(fp_input_path)
        fp_input_path = os.path.abspath(fp_input_path)
        fp_params = get_abacus_input_parameters(fp_input_path)
    else:
        raise RuntimeError(
            "Set 'user_fp_params' or 'fp_incar' in json file to make INPUT of ABACUS"
        )
    ret_input = make_abacus_scf_input(fp_params)

    # Get orbital and deepks setting
    if "basis_type" in fp_params:
        if fp_params["basis_type"] == "lcao":
            assert (
                "fp_orb_files" in jdata
                and type(jdata["fp_orb_files"]) == list
                and len(jdata["fp_orb_files"]) == len(fp_pp_files)
            )
            fp_orb_files = jdata["fp_orb_files"]
    dpks_out_labels = fp_params.get("deepks_out_labels", 0)
    dpks_scf = fp_params.get("deepks_scf", 0)
    if dpks_out_labels or dpks_scf:
        assert (
            "fp_dpks_descriptor" in jdata and type(jdata["fp_dpks_descriptor"]) == str
        )
        fp_dpks_descriptor = jdata["fp_dpks_descriptor"]

    # get paramters for writting KPT file
    if "kspacing" not in fp_params.keys():
        if "gamma_only" in fp_params.keys():
            if fp_params["gamma_only"] == 1:
                gamma_param = {"k_points": [1, 1, 1, 0, 0, 0]}
                ret_kpt = make_abacus_scf_kpt(gamma_param)
            else:
                if "k_points" in jdata.keys():
                    ret_kpt = make_abacus_scf_kpt(jdata)
                elif "fp_kpt_file" in jdata.keys():
                    fp_kpt_path = jdata["fp_kpt_file"]
                    assert os.path.exists(fp_kpt_path)
                    fp_kpt_path = os.path.abspath(fp_kpt_path)
                    fk = open(fp_kpt_path)
                    ret_kpt = fk.read()
                    fk.close()
                else:
                    raise RuntimeError("Cannot find any k-points information")
        else:
            if "k_points" in jdata.keys():
                ret_kpt = make_abacus_scf_kpt(jdata)
            elif "fp_kpt_file" in jdata.keys():
                fp_kpt_path = jdata["fp_kpt_file"]
                assert os.path.exists(fp_kpt_path)
                fp_kpt_path = os.path.abspath(fp_kpt_path)
                fk = open(fp_kpt_path)
                ret_kpt = fk.read()
                fk.close()
            else:
                gamma_param = {"k_points": [1, 1, 1, 0, 0, 0]}
                ret_kpt = make_abacus_scf_kpt(gamma_param)
                warnings.warn(
                    "Cannot find k-points information, gamma_only will be generated."
                )

    cwd = os.getcwd()
    for ii in fp_tasks:
        os.chdir(ii)
        sys_data = dpdata.System("POSCAR").data
        if "mass_map" in jdata:
            sys_data["atom_masses"] = jdata["mass_map"]
        with open("INPUT", "w") as fp:
            fp.write(ret_input)
        if "kspacing" not in fp_params.keys():
            with open("KPT", "w") as fp:
                fp.write(ret_kpt)
        ret_stru = make_abacus_scf_stru(
            sys_data,
            fp_pp_files,
            fp_orb_files,
            fp_dpks_descriptor,
            fp_params,
            type_map=jdata["type_map"],
        )
        with open("STRU", "w") as fp:
            fp.write(ret_stru)

        os.chdir(cwd)
    # link pp and orbital files
    _link_fp_abacus_pporb_descript(iter_index, jdata)



def _abacus_scf_check_fin(ii):
    if os.path.isfile(os.path.join(ii, "OUT.ABACUS/running_scf.log")):
        with open(os.path.join(ii, "OUT.ABACUS/running_scf.log"), "r") as fp:
            content = fp.read()
            count = content.count("!FINAL_ETOT_IS")
            if count != 1:
                return False
    else:
        return False
    return True



def post_fp_abacus_scf(iter_index, jdata):
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
        sys_output = glob.glob(os.path.join(work_path, "task.%s.*" % ss))
        sys_input = glob.glob(os.path.join(work_path, "task.%s.*/INPUT" % ss))
        sys_output.sort()
        sys_input.sort()

        all_sys = None
        for ii, oo in zip(sys_input, sys_output):
            _sys = dpdata.LabeledSystem(
                oo, fmt="abacus/scf", type_map=jdata["type_map"]
            )
            if len(_sys) > 0:
                if all_sys == None:
                    all_sys = _sys
                else:
                    all_sys.append(_sys)

        if all_sys != None:
            sys_data_path = os.path.join(work_path, "data.%s" % ss)
            all_sys.to_deepmd_raw(sys_data_path)
            all_sys.to_deepmd_npy(sys_data_path, set_size=len(sys_output))
