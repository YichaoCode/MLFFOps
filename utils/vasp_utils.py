#vasp_utils.py

def make_vasp_incar(jdata, filename):
    if "fp_incar" in jdata.keys():
        fp_incar_path = jdata["fp_incar"]
        assert os.path.exists(fp_incar_path)
        fp_incar_path = os.path.abspath(fp_incar_path)
        fr = open(fp_incar_path)
        incar = fr.read()
        fr.close()
    elif "user_fp_params" in jdata.keys():
        incar = write_incar_dict(jdata["user_fp_params"])
    else:
        incar = make_vasp_incar_user_dict(jdata["fp_params"])
    with open(filename, "w") as fp:
        fp.write(incar)
    return incar


def make_vasp_incar_ele_temp(jdata, filename, ele_temp, nbands_esti=None):
    with open(filename) as fp:
        incar = fp.read()
    incar = incar_upper(Incar.from_string(incar))
    incar["ISMEAR"] = -1
    incar["SIGMA"] = ele_temp * pc.Boltzmann / pc.electron_volt
    incar.write_file("INCAR")
    if nbands_esti is not None:
        nbands = nbands_esti.predict(".")
        with open(filename) as fp:
            incar = Incar.from_string(fp.read())
        incar["NBANDS"] = nbands
        incar.write_file("INCAR")


def make_fp_vasp_incar(iter_index, jdata, nbands_esti=None):
    iter_name = make_iter_name(iter_index)
    work_path = os.path.join(iter_name, fp_name)
    fp_tasks = glob.glob(os.path.join(work_path, "task.*"))
    fp_tasks.sort()
    if len(fp_tasks) == 0:
        return
    cwd = os.getcwd()
    for ii in fp_tasks:
        os.chdir(ii)
        make_vasp_incar(jdata, "INCAR")
        if os.path.exists("job.json"):
            with open("job.json") as fp:
                job_data = json.load(fp)
            if "ele_temp" in job_data:
                make_vasp_incar_ele_temp(
                    jdata, "INCAR", job_data["ele_temp"], nbands_esti=nbands_esti
                )
        os.chdir(cwd)


def make_fp_vasp_cp_cvasp(iter_index, jdata):
    # Move cvasp interface to jdata
    if ("cvasp" in jdata) and (jdata["cvasp"] == True):
        pass
    else:
        return
    iter_name = make_iter_name(iter_index)
    work_path = os.path.join(iter_name, fp_name)
    fp_tasks = glob.glob(os.path.join(work_path, "task.*"))
    fp_tasks.sort()
    if len(fp_tasks) == 0:
        return
    cwd = os.getcwd()
    for ii in fp_tasks:
        os.chdir(ii)
        # copy cvasp.py
        shutil.copyfile(cvasp_file, "cvasp.py")
        os.chdir(cwd)


def make_fp_vasp_kp(iter_index, jdata):
    iter_name = make_iter_name(iter_index)
    work_path = os.path.join(iter_name, fp_name)
    fp_aniso_kspacing = jdata.get("fp_aniso_kspacing")

    fp_tasks = glob.glob(os.path.join(work_path, "task.*"))
    fp_tasks.sort()
    if len(fp_tasks) == 0:
        return
    cwd = os.getcwd()
    for ii in fp_tasks:
        os.chdir(ii)
        # get kspacing and kgamma from incar
        assert os.path.exists("INCAR")
        with open("INCAR") as fp:
            incar = fp.read()
        standard_incar = incar_upper(Incar.from_string(incar))
        if fp_aniso_kspacing is None:
            try:
                kspacing = standard_incar["KSPACING"]
            except KeyError:
                raise RuntimeError("KSPACING must be given in INCAR")
        else:
            kspacing = fp_aniso_kspacing
        try:
            gamma = standard_incar["KGAMMA"]
            if isinstance(gamma, bool):
                pass
            else:
                if gamma[0].upper() == "T":
                    gamma = True
                else:
                    gamma = False
        except KeyError:
            raise RuntimeError("KGAMMA must be given in INCAR")
        # check poscar
        assert os.path.exists("POSCAR")
        # make kpoints
        ret = make_kspacing_kpoints("POSCAR", kspacing, gamma)
        kp = Kpoints.from_string(ret)
        kp.write_file("KPOINTS")
        os.chdir(cwd)


def _vasp_check_fin(ii):
    if os.path.isfile(os.path.join(ii, "OUTCAR")):
        with open(os.path.join(ii, "OUTCAR"), "r") as fp:
            content = fp.read()
            count = content.count("Elapse")
            if count != 1:
                return False
    else:
        return False
    return True



def post_fp_vasp(iter_index, jdata, rfailed=None):

    ratio_failed = rfailed if rfailed else jdata.get("ratio_failed", 0.05)
    model_devi_engine = jdata.get("model_devi_engine", "lammps")
    if model_devi_engine != "calypso":
        model_devi_jobs = jdata["model_devi_jobs"]
        assert iter_index < len(model_devi_jobs)
    use_ele_temp = jdata.get("use_ele_temp", 0)

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
        sys_outcars = glob.glob(os.path.join(work_path, "task.%s.*/OUTCAR" % ss))
        sys_outcars.sort()
        tcount += len(sys_outcars)
        all_sys = None
        all_te = []
        for oo in sys_outcars:
            try:
                _sys = dpdata.LabeledSystem(oo, type_map=jdata["type_map"])
            except Exception:
                dlog.info("Try to parse from vasprun.xml")
                try:
                    _sys = dpdata.LabeledSystem(
                        oo.replace("OUTCAR", "vasprun.xml"), type_map=jdata["type_map"]
                    )
                except Exception:
                    _sys = dpdata.LabeledSystem()
                    dlog.info("Failed fp path: %s" % oo.replace("OUTCAR", ""))
            print(type(_sys))
            if len(_sys) == 1:
                if all_sys is None:
                    all_sys = _sys
                else:
                    energy=_sys['energies']
                    print(energy)
                    if len(energy)==0:
                      continue
                    if abs(energy[0] + 188)>10:
                       continue
                    print(_sys)
                    if len(_sys[0])!=1:
                       continue
                    all_sys.append(_sys)
                # save ele_temp, if any
                if os.path.exists(oo.replace("OUTCAR", "job.json")):
                    with open(oo.replace("OUTCAR", "job.json")) as fp:
                        job_data = json.load(fp)
                    if "ele_temp" in job_data:
                        assert use_ele_temp
                        ele_temp = job_data["ele_temp"]
                        all_te.append(ele_temp)
            elif len(_sys) >= 2:
           #     print(1)
                raise RuntimeError("The vasp parameter NSW should be set as 1")
            else:
                icount += 1
        all_te = np.array(all_te)
        if all_sys is not None:
            sys_data_path = os.path.join(work_path, "data.%s" % ss)
            all_sys.to_deepmd_raw(sys_data_path)
            all_sys.to_deepmd_npy(sys_data_path, set_size=len(sys_outcars))
            if all_te.size > 0:
                assert len(all_sys) == all_sys.get_nframes()
                assert len(all_sys) == all_te.size
                all_te = np.reshape(all_te, [-1, 1])
                if use_ele_temp == 0:
                    raise RuntimeError(
                        "should not get ele temp at setting: use_ele_temp == 0"
                    )
                elif use_ele_temp == 1:
                    np.savetxt(os.path.join(sys_data_path, "fparam.raw"), all_te)
                    np.save(
                        os.path.join(sys_data_path, "set.000", "fparam.npy"), all_te
                    )
                elif use_ele_temp == 2:
                    tile_te = np.tile(all_te, [1, all_sys.get_natoms()])
                    np.savetxt(os.path.join(sys_data_path, "aparam.raw"), tile_te)
                    np.save(
                        os.path.join(sys_data_path, "set.000", "aparam.npy"), tile_te
                    )
                else:
                    raise RuntimeError(
                        "invalid setting of use_ele_temp " + str(use_ele_temp)
                    )

    if tcount == 0:
        rfail = 0.0
        dlog.info("failed frame: %6d in %6d " % (icount, tcount))
    else:
        rfail = float(icount) / float(tcount)
        dlog.info(
            "failed frame: %6d in %6d  %6.2f %% " % (icount, tcount, rfail * 100.0)
        )

    if rfail > ratio_failed:
        raise RuntimeError(
            "find too many unsuccessfully terminated jobs. Too many FP tasks are not converged. Please check your input parameters (e.g. INCAR) or configuration (e.g. POSCAR) in directories 'iter.*.*/02.fp/task.*.*/.'"
        )
