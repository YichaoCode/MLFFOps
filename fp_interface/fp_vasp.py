# fp_vasp.py
# created by Yichao


import os
import glob
import json
import shutil
import random
import warnings
from collections import Counter
from packaging.version import Version
from typing import List


import numpy as np
import scipy.constants as pc
import dpdata

from pymatgen.io.vasp import Incar, Kpoints

from dpgen.dispatcher.Dispatcher import make_submission
from dpgen.generator.lib.utils import create_path, make_iter_name
from dpgen.generator.lib.ele_temp import NBandsEsti
from dpgen.generator.lib.vasp import make_vasp_incar_user_dict, write_incar_dict
from dpgen.auto_test.lib.vasp import make_kspacing_kpoints, incar_upper
from dpgen.generator.lib.gaussian import take_cluster
from dpgen.generator.lib.parse_calypso import _parse_calypso_input, _parse_calypso_dis_mtx
from dpgen import dlog


fp_name = "02.fp"
model_devi_name = "01.model_devi"
data_system_fmt = "%03d"
fp_task_fmt = data_system_fmt + ".%06d"


def _make_fp_vasp_inner(
    iter_index,
    modd_path,
    work_path,
    model_devi_skip,
    v_trust_lo,
    v_trust_hi,
    f_trust_lo,
    f_trust_hi,
    fp_task_min,
    fp_task_max,
    fp_link_files,
    type_map,
    jdata,
):
    """
    iter_index          int             iter index
    modd_path           string          path of model devi
    work_path           string          path of fp
    fp_task_max         int             max number of tasks
    fp_link_files       [string]        linked files for fp, POTCAR for example
    fp_params           map             parameters for fp
    """

    # --------------------------------------------------------------------------------------------------------------------------------------
    model_devi_engine = jdata.get("model_devi_engine", "lammps")
    if model_devi_engine == "calypso":
        iter_name = work_path.split("/")[0]
        _work_path = os.path.join(iter_name, model_devi_name)
        # calypso_run_opt_path = os.path.join(_work_path,calypso_run_opt_name)
        calypso_run_opt_path = glob.glob(
            "%s/%s.*" % (_work_path, calypso_run_opt_name)
        )[0]
        numofspecies = _parse_calypso_input("NumberOfSpecies", calypso_run_opt_path)
        min_dis = _parse_calypso_dis_mtx(numofspecies, calypso_run_opt_path)

        calypso_total_fp_num = 300
        modd_path = os.path.join(modd_path, calypso_model_devi_name)
        model_devi_skip = -1
        with open(os.path.join(modd_path, "Model_Devi.out"), "r") as summfile:
            summary = np.loadtxt(summfile)
        summaryfmax = summary[:, -4]
        dis = summary[:, -1]
        acc = np.where((summaryfmax <= f_trust_lo) & (dis > float(min_dis)))
        fail = np.where((summaryfmax > f_trust_hi) | (dis <= float(min_dis)))
        nnan = np.where(np.isnan(summaryfmax))

        acc_num = len(acc[0])
        fail_num = len(fail[0])
        nan_num = len(nnan[0])
        tot = len(summaryfmax) - nan_num
        candi_num = tot - acc_num - fail_num
        dlog.info(
            "summary  accurate_ratio: {0:8.4f}%  candidata_ratio: {1:8.4f}%  failed_ratio: {2:8.4f}%  in {3:d} structures".format(
                acc_num * 100 / tot, candi_num * 100 / tot, fail_num * 100 / tot, tot
            )
        )
    # --------------------------------------------------------------------------------------------------------------------------------------

    modd_task = glob.glob(os.path.join(modd_path, "task.*"))
    modd_task.sort()
    system_index = []
    for ii in modd_task:
        system_index.append(os.path.basename(ii).split(".")[1])

    set_tmp = set(system_index)
    system_index = list(set_tmp)
    system_index.sort()

    fp_tasks = []

    charges_recorder = []  # record charges for each fp_task
    charges_map = jdata.get("sys_charges", [])

    cluster_cutoff = jdata.get("cluster_cutoff", None)
    model_devi_adapt_trust_lo = jdata.get("model_devi_adapt_trust_lo", False)
    model_devi_f_avg_relative = jdata.get("model_devi_f_avg_relative", False)
    model_devi_merge_traj = jdata.get("model_devi_merge_traj", False)
    # skip save *.out if detailed_report_make_fp is False, default is True
    detailed_report_make_fp = jdata.get("detailed_report_make_fp", True)
    # skip bad box criteria
    skip_bad_box = jdata.get("fp_skip_bad_box")
    # skip discrete structure in cluster
    fp_cluster_vacuum = jdata.get("fp_cluster_vacuum", None)

    def _trust_limitation_check(sys_idx, lim):
        if isinstance(lim, list):
            sys_lim = lim[sys_idx]
        elif isinstance(lim, dict):
            sys_lim = lim[str(sys_idx)]
        else:
            sys_lim = lim
        return sys_lim

    for ss in system_index:
        modd_system_glob = os.path.join(modd_path, "task." + ss + ".*")
        modd_system_task = glob.glob(modd_system_glob)
        modd_system_task.sort()
        if model_devi_engine in ("lammps", "gromacs", "calypso","dimer"):
            # convert global trust limitations to local ones
            f_trust_lo_sys = _trust_limitation_check(int(ss), f_trust_lo)
            f_trust_hi_sys = _trust_limitation_check(int(ss), f_trust_hi)
            v_trust_lo_sys = _trust_limitation_check(int(ss), v_trust_lo)
            v_trust_hi_sys = _trust_limitation_check(int(ss), v_trust_hi)

            # assumed e -> v
            if not model_devi_adapt_trust_lo:
                (
                    fp_rest_accurate,
                    fp_candidate,
                    fp_rest_failed,
                    counter,
                ) = _select_by_model_devi_standard(
                    modd_system_task,
                    f_trust_lo_sys,
                    f_trust_hi_sys,
                    v_trust_lo_sys,
                    v_trust_hi_sys,
                    cluster_cutoff,
                    model_devi_engine,
                    model_devi_skip,
                    model_devi_f_avg_relative=model_devi_f_avg_relative,
                    model_devi_merge_traj=model_devi_merge_traj,
                    detailed_report_make_fp=detailed_report_make_fp,
                )
            else:
                numb_candi_f = jdata.get("model_devi_numb_candi_f", 10)
                numb_candi_v = jdata.get("model_devi_numb_candi_v", 0)
                perc_candi_f = jdata.get("model_devi_perc_candi_f", 0.0)
                perc_candi_v = jdata.get("model_devi_perc_candi_v", 0.0)
                (
                    fp_rest_accurate,
                    fp_candidate,
                    fp_rest_failed,
                    counter,
                    f_trust_lo_ad,
                    v_trust_lo_ad,
                ) = _select_by_model_devi_adaptive_trust_low(
                    modd_system_task,
                    f_trust_hi_sys,
                    numb_candi_f,
                    perc_candi_f,
                    v_trust_hi_sys,
                    numb_candi_v,
                    perc_candi_v,
                    model_devi_skip=model_devi_skip,
                    model_devi_f_avg_relative=model_devi_f_avg_relative,
                    model_devi_merge_traj=model_devi_merge_traj,
                )
                dlog.info(
                    "system {0:s} {1:9s} : f_trust_lo {2:6.3f}   v_trust_lo {3:6.3f}".format(
                        ss, "adapted", f_trust_lo_ad, v_trust_lo_ad
                    )
                )
        elif model_devi_engine == "amber":
            counter = Counter()
            counter["candidate"] = 0
            counter["failed"] = 0
            counter["accurate"] = 0
            fp_rest_accurate = []
            fp_candidate = []
            fp_rest_failed = []
            for tt in modd_system_task:
                cc = 0
                with open(os.path.join(tt, "rc.mdout")) as f:
                    skip_first = False
                    first_active = True
                    for line in f:
                        if line.startswith("     ntx     =       1"):
                            skip_first = True
                        if line.startswith(
                            "Active learning frame written with max. frc. std.:"
                        ):
                            if skip_first and first_active:
                                first_active = False
                                continue
                            model_devi = (
                                float(line.split()[-2])
                                * dpdata.unit.EnergyConversion("kcal_mol", "eV").value()
                            )
                            if model_devi < f_trust_lo:
                                # accurate
                                if detailed_report_make_fp:
                                    fp_rest_accurate.append([tt, cc])
                                counter["accurate"] += 1
                            elif model_devi > f_trust_hi:
                                # failed
                                if detailed_report_make_fp:
                                    fp_rest_failed.append([tt, cc])
                                counter["failed"] += 1
                            else:
                                # candidate
                                fp_candidate.append([tt, cc])
                                counter["candidate"] += 1
                            cc += 1

        else:
            raise RuntimeError("unknown model_devi_engine", model_devi_engine)

        # print a report
        fp_sum = sum(counter.values())

        if fp_sum == 0:
            dlog.info(
                "system {0:s} has no fp task, maybe the model devi is nan %".format(ss)
            )
            continue
        for cc_key, cc_value in counter.items():
            dlog.info(
                "system {0:s} {1:9s} : {2:6d} in {3:6d} {4:6.2f} %".format(
                    ss, cc_key, cc_value, fp_sum, cc_value / fp_sum * 100
                )
            )
        random.shuffle(fp_candidate)
        if detailed_report_make_fp:
            random.shuffle(fp_rest_failed)
            random.shuffle(fp_rest_accurate)
            with open(
                os.path.join(work_path, "candidate.shuffled.%s.out" % ss), "w"
            ) as fp:
                for ii in fp_candidate:
                    fp.write(" ".join([str(nn) for nn in ii]) + "\n")
            with open(
                os.path.join(work_path, "rest_accurate.shuffled.%s.out" % ss), "w"
            ) as fp:
                for ii in fp_rest_accurate:
                    fp.write(" ".join([str(nn) for nn in ii]) + "\n")
            with open(
                os.path.join(work_path, "rest_failed.shuffled.%s.out" % ss), "w"
            ) as fp:
                for ii in fp_rest_failed:
                    fp.write(" ".join([str(nn) for nn in ii]) + "\n")

        # set number of tasks
        accurate_ratio = float(counter["accurate"]) / float(fp_sum)
        fp_accurate_threshold = jdata.get("fp_accurate_threshold", 1)
        fp_accurate_soft_threshold = jdata.get(
            "fp_accurate_soft_threshold", fp_accurate_threshold
        )
        if accurate_ratio < fp_accurate_soft_threshold:
            this_fp_task_max = fp_task_max
        elif (
            accurate_ratio >= fp_accurate_soft_threshold
            and accurate_ratio < fp_accurate_threshold
        ):
            this_fp_task_max = int(
                fp_task_max
                * (accurate_ratio - fp_accurate_threshold)
                / (fp_accurate_soft_threshold - fp_accurate_threshold)
            )
        else:
            this_fp_task_max = 0
        # ----------------------------------------------------------------------------
        if model_devi_engine == "calypso":
            calypso_intend_fp_num_temp = (
                len(fp_candidate) / candi_num
            ) * calypso_total_fp_num
            if calypso_intend_fp_num_temp < 1:
                calypso_intend_fp_num = 1
            else:
                calypso_intend_fp_num = int(calypso_intend_fp_num_temp)
        # ----------------------------------------------------------------------------
        numb_task = min(this_fp_task_max, len(fp_candidate))
        if numb_task < fp_task_min:
            numb_task = 0

        # ----------------------------------------------------------------------------
        if (model_devi_engine == "calypso" and len(jdata.get("type_map")) == 1) or (
            model_devi_engine == "calypso"
            and len(jdata.get("type_map")) > 1
            and candi_num <= calypso_total_fp_num
        ):
            numb_task = min(this_fp_task_max, len(fp_candidate))
            if numb_task < fp_task_min:
                numb_task = 0
        elif (
            model_devi_engine == "calypso"
            and len(jdata.get("type_map")) > 1
            and candi_num > calypso_total_fp_num
        ):
            numb_task = calypso_intend_fp_num
            if len(fp_candidate) < numb_task:
                numb_task = 0
        # ----------------------------------------------------------------------------
        dlog.info(
            "system {0:s} accurate_ratio: {1:8.4f}    thresholds: {2:6.4f} and {3:6.4f}   eff. task min and max {4:4d} {5:4d}   number of fp tasks: {6:6d}".format(
                ss,
                accurate_ratio,
                fp_accurate_soft_threshold,
                fp_accurate_threshold,
                fp_task_min,
                this_fp_task_max,
                numb_task,
            )
        )
        # make fp tasks

        # read all.lammpstrj, save in all_sys for each system_index
        all_sys = []
        trj_freq = None
        if model_devi_merge_traj:
            for ii in modd_system_task:
                all_traj = os.path.join(ii, "all.lammpstrj")
                all_sys_per_task = dpdata.System(
                    all_traj, fmt="lammps/dump", type_map=type_map
                )
                all_sys.append(all_sys_per_task)
            model_devi_jobs = jdata["model_devi_jobs"]
            cur_job = model_devi_jobs[iter_index]
            trj_freq = int(
                _get_param_alias(cur_job, ["t_freq", "trj_freq", "traj_freq"])
            )

        count_bad_box = 0
        count_bad_cluster = 0
        fp_candidate = sorted(fp_candidate[:numb_task])

        for cc in range(numb_task):
            tt = fp_candidate[cc][0]
            ii = fp_candidate[cc][1]
            ss = os.path.basename(tt).split(".")[1]
            conf_name = os.path.join(tt, "traj")
            conf_sys = None
            if (model_devi_engine == "lammps") or (model_devi_engine == "dimer") :
                if model_devi_merge_traj:
                    conf_sys = all_sys[int(os.path.basename(tt).split(".")[-1])][
                        int(int(ii) / trj_freq)
                    ]
                else:
                    conf_name = os.path.join(conf_name, str(ii) + ".lammpstrj")
                ffmt = "lammps/dump"
            elif model_devi_engine == "gromacs":
                conf_name = os.path.join(conf_name, str(ii) + ".gromacstrj")
                ffmt = "lammps/dump"
            elif model_devi_engine == "amber":
                conf_name = os.path.join(tt, "rc.nc")
                rst_name = os.path.abspath(os.path.join(tt, "init.rst7"))
            elif model_devi_engine == "calypso":
                conf_name = os.path.join(conf_name, str(ii) + ".poscar")
                ffmt = "vasp/poscar"
            else:
                raise RuntimeError("unknown model_devi engine", model_devi_engine)
            conf_name = os.path.abspath(conf_name)
            if skip_bad_box is not None:
                skip = check_bad_box(conf_name, skip_bad_box, fmt=ffmt)
                if skip:
                    count_bad_box += 1
                    continue

            if fp_cluster_vacuum is not None:
                assert fp_cluster_vacuum > 0
                skip_cluster = check_cluster(conf_name, fp_cluster_vacuum)
                if skip_cluster:
                    count_bad_cluster += 1
                    continue

            if model_devi_engine != "calypso":
                # link job.json
                job_name = os.path.join(tt, "job.json")
                job_name = os.path.abspath(job_name)

            if cluster_cutoff is not None:
                # take clusters
                jj = fp_candidate[cc][2]
                poscar_name = "{}.cluster.{}.POSCAR".format(conf_name, jj)
                new_system = take_cluster(conf_name, type_map, jj, jdata)
                new_system.to_vasp_poscar(poscar_name)
            fp_task_name = make_fp_task_name(int(ss), cc)
            fp_task_path = os.path.join(work_path, fp_task_name)
            create_path(fp_task_path)
            fp_tasks.append(fp_task_path)
            if charges_map:
                charges_recorder.append(charges_map[int(ss)])
            cwd = os.getcwd()
            os.chdir(fp_task_path)
            if cluster_cutoff is None:
                if (model_devi_engine == "lammps") or (model_devi_engine == "dimer"):
                    if model_devi_merge_traj:
                        conf_sys.to("lammps/lmp", "conf.dump")
                    else:
                        os.symlink(os.path.relpath(conf_name), "conf.dump")
                    os.symlink(os.path.relpath(job_name), "job.json")
                elif model_devi_engine == "gromacs":
                    os.symlink(os.path.relpath(conf_name), "conf.dump")
                    os.symlink(os.path.relpath(job_name), "job.json")
                elif model_devi_engine == "amber":
                    # read and write with ase
                    from ase.io.netcdftrajectory import (
                        NetCDFTrajectory,
                        write_netcdftrajectory,
                    )

                    if cc > 0 and tt == fp_candidate[cc - 1][0]:
                        # same MD task, use the same file
                        pass
                    else:
                        # not the same file
                        if cc > 0:
                            # close the old file
                            netcdftraj.close()
                        netcdftraj = NetCDFTrajectory(conf_name)
                    # write nc file
                    write_netcdftrajectory("rc.nc", netcdftraj[ii])
                    if cc >= numb_task - 1:
                        netcdftraj.close()
                    # link restart since it's necessary to start Amber
                    os.symlink(os.path.relpath(rst_name), "init.rst7")
                    os.symlink(os.path.relpath(job_name), "job.json")
                elif model_devi_engine == "calypso":
                    os.symlink(os.path.relpath(conf_name), "POSCAR")
                    fjob = open("job.json", "w+")
                    fjob.write('{"model_devi_engine":"calypso"}')
                    fjob.close()
                    # os.system('touch job.json')
                else:
                    raise RuntimeError("unknown model_devi_engine", model_devi_engine)
            else:
                os.symlink(os.path.relpath(poscar_name), "POSCAR")
                np.save("atom_pref", new_system.data["atom_pref"])
            for pair in fp_link_files:
                os.symlink(pair[0], pair[1])
            os.chdir(cwd)
        if count_bad_box > 0:
            dlog.info(
                "system {0:s} skipped {1:6d} confs with bad box, {2:6d} remains".format(
                    ss, count_bad_box, numb_task - count_bad_box
                )
            )
        if count_bad_cluster > 0:
            dlog.info(
                "system {0:s} skipped {1:6d} confs with bad cluster, {2:6d} remains".format(
                    ss, count_bad_cluster, numb_task - count_bad_cluster
                )
            )
    if model_devi_engine == "calypso":
        dlog.info(
            "summary  accurate_ratio: {0:8.4f}%  candidata_ratio: {1:8.4f}%  failed_ratio: {2:8.4f}%  in {3:d} structures".format(
                acc_num * 100 / tot, candi_num * 100 / tot, fail_num * 100 / tot, tot
            )
        )
    if cluster_cutoff is None:
        cwd = os.getcwd()
        for idx, task in enumerate(fp_tasks):
            os.chdir(task)
            if  (model_devi_engine == "lammps") or (model_devi_engine == "dimer"):
                sys = None
                if model_devi_merge_traj:
                    sys = dpdata.System(
                        "conf.dump", fmt="lammps/lmp", type_map=type_map
                    )
                else:
                    sys = dpdata.System(
                        "conf.dump", fmt="lammps/dump", type_map=type_map
                    )
                sys.to_vasp_poscar("POSCAR")
                # dump to poscar

                if charges_map:
                    warnings.warn(
                        '"sys_charges" keyword only support for gromacs engine now.'
                    )
            elif model_devi_engine == "gromacs":
                # dump_to_poscar('conf.dump', 'POSCAR', type_map, fmt = "gromacs/gro")
                if charges_map:
                    dump_to_deepmd_raw(
                        "conf.dump",
                        "deepmd.raw",
                        type_map,
                        fmt="gromacs/gro",
                        charge=charges_recorder[idx],
                    )
                else:
                    dump_to_deepmd_raw(
                        "conf.dump",
                        "deepmd.raw",
                        type_map,
                        fmt="gromacs/gro",
                        charge=None,
                    )
            elif model_devi_engine in ("amber", "calypso"):
                pass
            else:
                raise RuntimeError("unknown model_devi engine", model_devi_engine)
            os.chdir(cwd)
    return fp_tasks


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



def _make_fp_vasp_configs(iter_index, jdata):
    fp_task_max = jdata["fp_task_max"]
    model_devi_skip = jdata["model_devi_skip"]
    type_map = jdata["type_map"]
    iter_name = make_iter_name(iter_index)
    work_path = os.path.join(iter_name, fp_name)
    create_path(work_path)

    modd_path = os.path.join(iter_name, model_devi_name)
    task_min = -1
    if os.path.isfile(os.path.join(modd_path, "cur_job.json")):
        cur_job = json.load(open(os.path.join(modd_path, "cur_job.json"), "r"))
        if "task_min" in cur_job:
            task_min = cur_job["task_min"]
    else:
        cur_job = {}
    # support iteration dependent trust levels
    v_trust_lo = cur_job.get(
        "model_devi_v_trust_lo", jdata.get("model_devi_v_trust_lo", 1e10)
    )
    v_trust_hi = cur_job.get(
        "model_devi_v_trust_hi", jdata.get("model_devi_v_trust_hi", 1e10)
    )
    if cur_job.get("model_devi_f_trust_lo") is not None:
        f_trust_lo = cur_job.get("model_devi_f_trust_lo")
    else:
        f_trust_lo = jdata["model_devi_f_trust_lo"]
    if cur_job.get("model_devi_f_trust_hi") is not None:
        f_trust_hi = cur_job.get("model_devi_f_trust_hi")
    else:
        f_trust_hi = jdata["model_devi_f_trust_hi"]

    # make configs
    fp_tasks = _make_fp_vasp_inner(
        iter_index,
        modd_path,
        work_path,
        model_devi_skip,
        v_trust_lo,
        v_trust_hi,
        f_trust_lo,
        f_trust_hi,
        task_min,
        fp_task_max,
        [],
        type_map,
        jdata,
    )
    return fp_tasks


def make_fp_vasp(iter_index, jdata):
    # make config
    fp_tasks = _make_fp_vasp_configs(iter_index, jdata)
    if len(fp_tasks) == 0:
        return
    # abs path for fp_incar if it exists
    if "fp_incar" in jdata:
        jdata["fp_incar"] = os.path.abspath(jdata["fp_incar"])
    # get nbands esti if it exists
    if "fp_nbands_esti_data" in jdata:
        nbe = NBandsEsti(jdata["fp_nbands_esti_data"])
    else:
        nbe = None
    # order is critical!
    # 1, create potcar
    sys_link_fp_vasp_pp(iter_index, jdata)
    # 2, create incar
    make_fp_vasp_incar(iter_index, jdata, nbands_esti=nbe)
    # 3, create kpoints
    make_fp_vasp_kp(iter_index, jdata)
    # 4, copy cvasp
    make_fp_vasp_cp_cvasp(iter_index, jdata)



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














def _select_by_model_devi_standard(
    modd_system_task: List[str],
    f_trust_lo: float,
    f_trust_hi: float,
    v_trust_lo: float,
    v_trust_hi: float,
    cluster_cutoff: float,
    model_devi_engine: str,
    model_devi_skip: int = 0,
    model_devi_f_avg_relative: bool = False,
    model_devi_merge_traj: bool = False,
    detailed_report_make_fp: bool = True,
):
    if model_devi_engine == "calypso":
        iter_name = modd_system_task[0].split("/")[0]
        _work_path = os.path.join(iter_name, model_devi_name)
        # calypso_run_opt_path = os.path.join(_work_path,calypso_run_opt_name)
        calypso_run_opt_path = glob.glob(
            "%s/%s.*" % (_work_path, calypso_run_opt_name)
        )[0]
        numofspecies = _parse_calypso_input("NumberOfSpecies", calypso_run_opt_path)
        min_dis = _parse_calypso_dis_mtx(numofspecies, calypso_run_opt_path)
    fp_candidate = []
    if detailed_report_make_fp:
        fp_rest_accurate = []
        fp_rest_failed = []
    cc = 0
    counter = Counter()
    counter["candidate"] = 0
    counter["failed"] = 0
    counter["accurate"] = 0
    for tt in modd_system_task:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            all_conf = _read_model_devi_file(
                tt, model_devi_f_avg_relative, model_devi_merge_traj
            )

            if all_conf.shape == (7,):
                all_conf = all_conf.reshape(1, all_conf.shape[0])
            elif model_devi_engine == "calypso" and all_conf.shape == (8,):
                all_conf = all_conf.reshape(1, all_conf.shape[0])
            for ii in range(all_conf.shape[0]):
                if all_conf[ii][0] < model_devi_skip:
                    continue
                cc = int(all_conf[ii][0])
                if cluster_cutoff is None:
                    if model_devi_engine == "calypso":
                        if float(all_conf[ii][-1]) <= float(min_dis):
                            if detailed_report_make_fp:
                                fp_rest_failed.append([tt, cc])
                            counter["failed"] += 1
                            continue
                    if (
                        all_conf[ii][1] < v_trust_hi and all_conf[ii][1] >= v_trust_lo
                    ) or (
                        all_conf[ii][4] < f_trust_hi and all_conf[ii][4] >= f_trust_lo
                    ):
                        fp_candidate.append([tt, cc])
                        counter["candidate"] += 1
                    elif (all_conf[ii][1] >= v_trust_hi) or (
                        all_conf[ii][4] >= f_trust_hi
                    ):
                        if detailed_report_make_fp:
                            fp_rest_failed.append([tt, cc])
                        counter["failed"] += 1
                    elif all_conf[ii][1] < v_trust_lo and all_conf[ii][4] < f_trust_lo:
                        if detailed_report_make_fp:
                            fp_rest_accurate.append([tt, cc])
                        counter["accurate"] += 1
                    else:
                        if model_devi_engine == "calypso":
                            dlog.info(
                                "ase opt traj %s frame %d with f devi %f does not belong to either accurate, candidiate and failed "
                                % (tt, ii, all_conf[ii][4])
                            )
                        else:
                            raise RuntimeError(
                                "md traj %s frame %d with f devi %f does not belong to either accurate, candidiate and failed, it should not happen"
                                % (tt, ii, all_conf[ii][4])
                            )
                else:
                    idx_candidate = np.where(
                        np.logical_and(
                            all_conf[ii][7:] < f_trust_hi,
                            all_conf[ii][7:] >= f_trust_lo,
                        )
                    )[0]
                    for jj in idx_candidate:
                        fp_candidate.append([tt, cc, jj])
                    counter["candidate"] += len(idx_candidate)
                    idx_rest_accurate = np.where(all_conf[ii][7:] < f_trust_lo)[0]
                    if detailed_report_make_fp:
                        for jj in idx_rest_accurate:
                            fp_rest_accurate.append([tt, cc, jj])
                    counter["accurate"] += len(idx_rest_accurate)
                    idx_rest_failed = np.where(all_conf[ii][7:] >= f_trust_hi)[0]
                    if detailed_report_make_fp:
                        for jj in idx_rest_failed:
                            fp_rest_failed.append([tt, cc, jj])
                    counter["failed"] += len(idx_rest_failed)

    return fp_rest_accurate, fp_candidate, fp_rest_failed, counter





def _read_model_devi_file(
    task_path: str,
    model_devi_f_avg_relative: bool = False,
    model_devi_merge_traj: bool = False,
):
    model_devi = np.loadtxt(os.path.join(task_path, "model_devi.out"))
    if model_devi_f_avg_relative:
        if model_devi_merge_traj is True:
            all_traj = os.path.join(task_path, "all.lammpstrj")
            all_f = get_all_dumped_forces(all_traj)
        else:
            trajs = glob.glob(os.path.join(task_path, "traj", "*.lammpstrj"))
            all_f = []
            for ii in trajs:
                all_f.append(get_dumped_forces(ii))

        all_f = np.array(all_f)
        all_f = all_f.reshape([-1, 3])
        avg_f = np.sqrt(np.average(np.sum(np.square(all_f), axis=1)))
        model_devi[:, 4:7] = model_devi[:, 4:7] / avg_f
        np.savetxt(
            os.path.join(task_path, "model_devi_avgf.out"), model_devi, fmt="%16.6e"
        )
    return model_devi






def make_fp_task_name(sys_idx, counter):
    return "task." + fp_task_fmt % (sys_idx, counter)
