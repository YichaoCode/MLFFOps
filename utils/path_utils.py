# path_utils.py

import os
import glob
import shutil

def find_only_one_key(lmp_lines, key):
    found = []
    for idx in range(len(lmp_lines)):
        words = lmp_lines[idx].split()
        nkey = len(key)
        if len(words) >= nkey and words[:nkey] == key:
            found.append(idx)
    if len(found) > 1:
        raise RuntimeError("found %d keywords %s" % (len(found), key))
    if len(found) == 0:
        raise RuntimeError("failed to find keyword %s" % (key))
    return found[0]


def _link_fp_vasp_pp(iter_index, jdata):
    fp_pp_path = jdata["fp_pp_path"]
    fp_pp_files = jdata["fp_pp_files"]
    assert os.path.exists(fp_pp_path)
    fp_pp_path = os.path.abspath(fp_pp_path)

    iter_name = make_iter_name(iter_index)
    work_path = os.path.join(iter_name, fp_name)

    fp_tasks = glob.glob(os.path.join(work_path, "task.*"))
    fp_tasks.sort()
    if len(fp_tasks) == 0:
        return
    cwd = os.getcwd()
    for ii in fp_tasks:
        os.chdir(ii)
        for jj in fp_pp_files:
            pp_file = os.path.join(fp_pp_path, jj)
            os.symlink(pp_file, jj)
        os.chdir(cwd)


def sys_link_fp_vasp_pp(iter_index, jdata):
    fp_pp_path = jdata["fp_pp_path"]
    fp_pp_files = jdata["fp_pp_files"]
    fp_pp_path = os.path.abspath(fp_pp_path)
    type_map = jdata["type_map"]
    assert os.path.exists(fp_pp_path)
   # assert len(fp_pp_files) == len(
    #    type_map
    #), "size of fp_pp_files should be the same as the size of type_map"

    iter_name = make_iter_name(iter_index)
    work_path = os.path.join(iter_name, fp_name)

    fp_tasks = glob.glob(os.path.join(work_path, "task.*"))
    fp_tasks.sort()
    if len(fp_tasks) == 0:
        return

    system_idx_str = [os.path.basename(ii).split(".")[1] for ii in fp_tasks]
    system_idx_str = list(set(system_idx_str))
    system_idx_str.sort()
    for ii in system_idx_str:
        potcars = []
        sys_tasks = glob.glob(os.path.join(work_path, "task.%s.*" % ii))
        assert len(sys_tasks) != 0
        sys_poscar = os.path.join(sys_tasks[0], "POSCAR")
        sys = dpdata.System(sys_poscar, fmt="vasp/poscar")
        
        #for ele_name in sys["atom_names"]:
         #   ele_idx = jdata["type_map"].index(ele_name)
        for  fi in fp_pp_files:
          potcars.append(fi)#fp_pp_files[ele_idx])
        with open(os.path.join(work_path, "POTCAR.%s" % ii), "w") as fp_pot:
            for jj in potcars:
                with open(os.path.join(fp_pp_path, jj)) as fp:
                    fp_pot.write(fp.read())
        sys_tasks = glob.glob(os.path.join(work_path, "task.%s.*" % ii))
        cwd = os.getcwd()
        for jj in sys_tasks:
            os.chdir(jj)
            os.symlink(os.path.join("..", "POTCAR.%s" % ii), "POTCAR")
            os.chdir(cwd)


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


