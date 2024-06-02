# model_devi.py
# created by Yichao

import os
import glob
import json
import shutil
from packaging.version import Version


from dpgen.generator.lib.make_calypso import (
    _make_model_devi_buffet,
    _make_model_devi_native_calypso,
)

from dpgen.generator.lib.utils import (
    make_iter_name
)

from utils.utils import (
    expand_idx,
    create_path,
    _get_param_alias,
    make_model_devi_task_name,
    make_model_devi_conf_name,
    revise_lmp_input_model,
    revise_by_keys
)
from config import *



def _make_model_devi_native(iter_index, jdata, mdata, conf_systems):
    model_devi_jobs = jdata["model_devi_jobs"]
    if iter_index >= len(model_devi_jobs):
        return False
    cur_job = model_devi_jobs[iter_index]
    ensemble, nsteps, trj_freq, temps, press, pka_e, dt = parse_cur_job(cur_job)
    if dt is not None:
        model_devi_dt = dt
    sys_idx = expand_idx(cur_job["sys_idx"])
    if len(sys_idx) != len(list(set(sys_idx))):
        raise RuntimeError("system index should be uniq")

    use_ele_temp = jdata.get("use_ele_temp", 0)
    model_devi_dt = jdata["model_devi_dt"]
    model_devi_neidelay = None
    if "model_devi_neidelay" in jdata:
        model_devi_neidelay = jdata["model_devi_neidelay"]
    model_devi_taut = 0.1
    if "model_devi_taut" in jdata:
        model_devi_taut = jdata["model_devi_taut"]
    model_devi_taup = 0.5
    if "model_devi_taup" in jdata:
        model_devi_taup = jdata["model_devi_taup"]
    mass_map = jdata["mass_map"]
    nopbc = jdata.get("model_devi_nopbc", False)

    iter_name = make_iter_name(iter_index)
    train_path = os.path.join(iter_name, train_name)
    train_path = os.path.abspath(train_path)
    models = glob.glob(os.path.join(train_path, "graph*pb"))
    task_model_list = []
    for ii in models:
        task_model_list.append(os.path.join("..", os.path.basename(ii)))
    work_path = os.path.join(iter_name, model_devi_name)

    sys_counter = 0
    for ss in conf_systems:
        conf_counter = 0
        task_counter = 0
        for cc in ss:
            for tt_ in temps:
                if use_ele_temp:
                    if type(tt_) == list:
                        tt = tt_[0]
                        if use_ele_temp == 1:
                            te_f = tt_[1]
                            te_a = None
                        else:
                            te_f = None
                            te_a = tt_[1]
                    else:
                        assert type(tt_) == float or type(tt_) == int
                        tt = float(tt_)
                        if use_ele_temp == 1:
                            te_f = tt
                            te_a = None
                        else:
                            te_f = None
                            te_a = tt
                else:
                    tt = tt_
                    te_f = None
                    te_a = None
                for pp in press:
                    task_name = make_model_devi_task_name(
                        sys_idx[sys_counter], task_counter
                    )
                    conf_name = (
                        make_model_devi_conf_name(sys_idx[sys_counter], conf_counter)
                        + ".lmp"
                    )
                    task_path = os.path.join(work_path, task_name)
                    # dlog.info(task_path)
                    create_path(task_path)
                    model_devi_merge_traj = jdata.get("model_devi_merge_traj", False)
                    if not model_devi_merge_traj:
                        create_path(os.path.join(task_path, "traj"))
                    loc_conf_name = "conf.lmp"
                    os.symlink(
                        os.path.join(os.path.join("..", "confs"), conf_name),
                        os.path.join(task_path, loc_conf_name),
                    )
                    cwd_ = os.getcwd()
                    os.chdir(task_path)
                    try:
                        mdata["deepmd_version"]
                    except KeyError:
                        mdata = set_version(mdata)
                    deepmd_version = mdata["deepmd_version"]
                    file_c = make_lammps_input(
                        ensemble,
                        loc_conf_name,
                        task_model_list,
                        nsteps,
                        model_devi_dt,
                        model_devi_neidelay,
                        trj_freq,
                        mass_map,
                        tt,
                        jdata=jdata,
                        tau_t=model_devi_taut,
                        pres=pp,
                        tau_p=model_devi_taup,
                        pka_e=pka_e,
                        ele_temp_f=te_f,
                        ele_temp_a=te_a,
                        nopbc=nopbc,
                        deepmd_version=deepmd_version,
                    )
                    job = {}
                    job["ensemble"] = ensemble
                    job["press"] = pp
                    job["temps"] = tt
                    if te_f is not None:
                        job["ele_temp"] = te_f
                    if te_a is not None:
                        job["ele_temp"] = te_a
                    job["model_devi_dt"] = model_devi_dt
                    with open("job.json", "w") as _outfile:
                        json.dump(job, _outfile, indent=4)
                    os.chdir(cwd_)
                    with open(os.path.join(task_path, "input.lammps"), "w") as fp:
                        fp.write(file_c)
                    task_counter += 1
            conf_counter += 1
        sys_counter += 1


def _make_model_devi_native_gromacs(iter_index, jdata, mdata, conf_systems):
    try:
        from gromacs.fileformats.mdp import MDP
    except ImportError as e:
        raise RuntimeError(
            "GromacsWrapper>=0.8.0 is needed for DP-GEN + Gromacs."
        ) from e
    # only support for deepmd v2.0
    if Version(mdata["deepmd_version"]) < Version("2.0"):
        raise RuntimeError(
            "Only support deepmd-kit 2.x for model_devi_engine='gromacs'"
        )
    model_devi_jobs = jdata["model_devi_jobs"]
    if iter_index >= len(model_devi_jobs):
        return False
    cur_job = model_devi_jobs[iter_index]
    dt = cur_job.get("dt", None)
    if dt is not None:
        model_devi_dt = dt
    else:
        model_devi_dt = jdata["model_devi_dt"]
    nsteps = cur_job.get("nsteps", None)
    lambdas = cur_job.get("lambdas", [1.0])
    temps = cur_job.get("temps", [298.0])

    for ll in lambdas:
        assert ll >= 0.0 and ll <= 1.0, "Lambda should be in [0,1]"

    if nsteps is None:
        raise RuntimeError("nsteps is None, you should set nsteps in model_devi_jobs!")
    # Currently Gromacs engine is not supported for different temperatures!
    # If you want to change temperatures, you should change it in mdp files.

    sys_idx = expand_idx(cur_job["sys_idx"])
    if len(sys_idx) != len(list(set(sys_idx))):
        raise RuntimeError("system index should be uniq")

    mass_map = jdata["mass_map"]

    iter_name = make_iter_name(iter_index)
    train_path = os.path.join(iter_name, train_name)
    train_path = os.path.abspath(train_path)
    models = glob.glob(os.path.join(train_path, "graph*pb"))
    task_model_list = []
    for ii in models:
        task_model_list.append(os.path.join("..", os.path.basename(ii)))
    work_path = os.path.join(iter_name, model_devi_name)

    sys_counter = 0
    for ss in conf_systems:
        conf_counter = 0
        task_counter = 0
        for cc in ss:
            for ll in lambdas:
                for tt in temps:
                    task_name = make_model_devi_task_name(
                        sys_idx[sys_counter], task_counter
                    )
                    task_path = os.path.join(work_path, task_name)
                    create_path(task_path)
                    gromacs_settings = jdata.get("gromacs_settings", "")
                    for key, file in gromacs_settings.items():
                        if (
                            key != "traj_filename"
                            and key != "mdp_filename"
                            and key != "group_name"
                            and key != "maxwarn"
                        ):
                            os.symlink(
                                os.path.join(cc, file), os.path.join(task_path, file)
                            )
                    # input.json for DP-Gromacs
                    with open(os.path.join(cc, "input.json")) as f:
                        input_json = json.load(f)
                    input_json["graph_file"] = models[0]
                    input_json["lambda"] = ll
                    with open(os.path.join(task_path, "input.json"), "w") as _outfile:
                        json.dump(input_json, _outfile, indent=4)

                    # trj_freq
                    trj_freq = cur_job.get("trj_freq", 10)
                    mdp = MDP()
                    mdp.read(os.path.join(cc, gromacs_settings["mdp_filename"]))
                    mdp["nstcomm"] = trj_freq
                    mdp["nstxout"] = trj_freq
                    mdp["nstlog"] = trj_freq
                    mdp["nstenergy"] = trj_freq
                    # dt
                    mdp["dt"] = model_devi_dt
                    # nsteps
                    mdp["nsteps"] = nsteps
                    # temps
                    if "ref_t" in list(mdp.keys()):
                        mdp["ref_t"] = tt
                    else:
                        mdp["ref-t"] = tt
                    mdp.write(os.path.join(task_path, gromacs_settings["mdp_filename"]))

                    cwd_ = os.getcwd()
                    os.chdir(task_path)
                    job = {}
                    job["trj_freq"] = cur_job["trj_freq"]
                    job["model_devi_dt"] = model_devi_dt
                    job["nsteps"] = nsteps
                    with open("job.json", "w") as _outfile:
                        json.dump(job, _outfile, indent=4)
                    os.chdir(cwd_)
                    task_counter += 1
            conf_counter += 1
        sys_counter += 1


def _make_model_devi_amber(
    iter_index: int, jdata: dict, mdata: dict, conf_systems: list
):
    """Make amber's MD inputs.

    Parameters
    ----------
    iter_index : int
        iter index
    jdata : dict
        run parameters. The following parameters will be used in this method:
            model_devi_jobs : list[dict]
                The list including the dict for information of each cycle:
                    sys_idx : list[int]
                        list of systems to run
                    trj_freq : int
                        freq to dump trajectory
            low_level : str
                low level method
            cutoff : float
                cutoff radius of the DPRc model
            parm7_prefix : str
                The path prefix to AMBER PARM7 files
            parm7 : list[str]
                List of paths to AMBER PARM7 files. Each file maps to a system.
            mdin_prefix : str
                The path prefix to AMBER mdin files
            mdin : list[str]
                List of paths to AMBER mdin files. Each files maps to a system.
                The following keywords will be replaced by the actual value:
                    @freq@ : freq to dump trajectory
                    @nstlim@ : total time step to run
                    @qm_region@ : AMBER mask of the QM region
                    @qm_theory@ : The QM theory, such as DFTB2
                    @qm_charge@ : The total charge of the QM theory, such as -2
                    @rcut@ : cutoff radius of the DPRc model
                    @GRAPH_FILE0@, @GRAPH_FILE1@, ... : graph files
            qm_region : list[str]
                AMBER mask of the QM region. Each mask maps to a system.
            qm_charge : list[int]
                Charge of the QM region. Each charge maps to a system.
            nsteps : list[int]
                The number of steps to run. Each number maps to a system.
            r : list[list[float]] or list[list[list[float]]]
                Constrict values for the enhanced sampling. The first dimension maps to systems.
                The second dimension maps to confs in each system. The third dimension is the
                constrict value. It can be a single float for 1D or list of floats for nD.
            disang_prefix : str
                The path prefix to disang prefix.
            disang : list[str]
                List of paths to AMBER disang files. Each file maps to a sytem.
                The keyword RVAL will be replaced by the constrict values, or RVAL1, RVAL2, ...
                for an nD system.
    mdata : dict
        machine parameters. Nothing will be used in this method.
    conf_systems : list
        conf systems

    References
    ----------
    .. [1] Development of Range-Corrected Deep Learning Potentials for Fast, Accurate Quantum
       Mechanical/Molecular Mechanical Simulations of Chemical Reactions in Solution,
       Jinzhe Zeng, Timothy J. Giese, ??len Ekesan, and Darrin M. York, Journal of Chemical
       Theory and Computation 2021 17 (11), 6993-7009

    inputs: restart (coords), param, mdin, graph, disang (optional)

    """
    model_devi_jobs = jdata["model_devi_jobs"]
    if iter_index >= len(model_devi_jobs):
        return False
    cur_job = model_devi_jobs[iter_index]
    sys_idx = expand_idx(cur_job["sys_idx"])
    if len(sys_idx) != len(list(set(sys_idx))):
        raise RuntimeError("system index should be uniq")

    iter_name = make_iter_name(iter_index)
    train_path = os.path.join(iter_name, train_name)
    train_path = os.path.abspath(train_path)
    work_path = os.path.join(iter_name, model_devi_name)
    # parm7 - list
    parm7 = jdata["parm7"]
    parm7_prefix = jdata.get("parm7_prefix", "")
    parm7 = [os.path.join(parm7_prefix, pp) for pp in parm7]

    # link parm file
    for ii, pp in enumerate(parm7):
        os.symlink(pp, os.path.join(work_path, "qmmm%d.parm7" % ii))
    # TODO: consider writing input in json instead of a given file
    # mdin
    mdin = jdata["mdin"]
    mdin_prefix = jdata.get("mdin_prefix", "")
    mdin = [os.path.join(mdin_prefix, pp) for pp in mdin]

    qm_region = jdata["qm_region"]
    qm_charge = jdata["qm_charge"]
    nsteps = jdata["nsteps"]

    for ii, pp in enumerate(mdin):
        with open(pp) as f, open(
            os.path.join(work_path, "init%d.mdin" % ii), "w"
        ) as fw:
            mdin_str = f.read()
            # freq, nstlim, qm_region, qm_theory, qm_charge, rcut, graph
            mdin_str = (
                mdin_str.replace("@freq@", str(cur_job.get("trj_freq", 50)))
                .replace("@nstlim@", str(nsteps[ii]))
                .replace("@qm_region@", qm_region[ii])
                .replace("@qm_charge@", str(qm_charge[ii]))
                .replace("@qm_theory@", jdata["low_level"])
                .replace("@rcut@", str(jdata["cutoff"]))
            )
            models = sorted(glob.glob(os.path.join(train_path, "graph.*.pb")))
            task_model_list = []
            for ii in models:
                task_model_list.append(os.path.join("..", os.path.basename(ii)))
            # graph
            for jj, mm in enumerate(task_model_list):
                # replace graph
                mdin_str = mdin_str.replace("@GRAPH_FILE%d@" % jj, mm)
            fw.write(mdin_str)
    # disang - list
    disang = jdata["disang"]
    disang_prefix = jdata.get("disang_prefix", "")
    disang = [os.path.join(disang_prefix, pp) for pp in disang]

    for sys_counter, ss in enumerate(conf_systems):
        for idx_cc, cc in enumerate(ss):
            task_counter = idx_cc
            conf_counter = idx_cc

            task_name = make_model_devi_task_name(sys_idx[sys_counter], task_counter)
            conf_name = make_model_devi_conf_name(sys_idx[sys_counter], conf_counter)
            task_path = os.path.join(work_path, task_name)
            # create task path
            create_path(task_path)
            # link restart file
            loc_conf_name = "init.rst7"
            os.symlink(
                os.path.join(os.path.join("..", "confs"), conf_name + ".rst7"),
                os.path.join(task_path, loc_conf_name),
            )
            cwd_ = os.getcwd()
            # chdir to task path
            os.chdir(task_path)

            # reaction coordinates of umbrella sampling
            # TODO: maybe consider a better name instead of `r`?
            if "r" in jdata:
                r = jdata["r"][sys_idx[sys_counter]][conf_counter]
                # r can either be a float or a list of float (for 2D coordinates)
                if not isinstance(r, Iterable) or isinstance(r, str):
                    r = [r]
                # disang file should include RVAL, RVAL2, ...
                with open(disang[sys_idx[sys_counter]]) as f, open(
                    "TEMPLATE.disang", "w"
                ) as fw:
                    tl = f.read()
                    for ii, rr in enumerate(r):
                        if isinstance(rr, Iterable) and not isinstance(rr, str):
                            raise RuntimeError(
                                "rr should not be iterable! sys: %d rr: %s r: %s"
                                % (sys_idx[sys_counter], str(rr), str(r))
                            )
                        tl = tl.replace("RVAL" + str(ii + 1), str(rr))
                    if len(r) == 1:
                        tl = tl.replace("RVAL", str(r[0]))
                    fw.write(tl)

            with open("job.json", "w") as fp:
                json.dump(cur_job, fp, indent=4)
            os.chdir(cwd_)


def _make_model_devi_revmat(iter_index, jdata, mdata, conf_systems):
    model_devi_jobs = jdata["model_devi_jobs"]
    if iter_index >= len(model_devi_jobs):
        return False
    cur_job = model_devi_jobs[iter_index]
    sys_idx = expand_idx(cur_job["sys_idx"])
    if len(sys_idx) != len(list(set(sys_idx))):
        raise RuntimeError("system index should be uniq")
    mass_map = jdata["mass_map"]
    use_plm = jdata.get("model_devi_plumed", False)
    use_plm_path = jdata.get("model_devi_plumed_path", False)
    trj_freq = _get_param_alias(cur_job, ["t_freq", "trj_freq", "traj_freq"])

    rev_keys, rev_mat, num_lmp = parse_cur_job_revmat(cur_job, use_plm=use_plm)
    lmp_templ = cur_job["template"]["lmp"]
    lmp_templ = os.path.abspath(lmp_templ)
    if use_plm:
        plm_templ = cur_job["template"]["plm"]
        plm_templ = os.path.abspath(plm_templ)
        if use_plm_path:
            plm_path_templ = cur_job["template"]["plm_path"]
            plm_path_templ = os.path.abspath(plm_path_templ)

    iter_name = make_iter_name(iter_index)
    train_path = os.path.join(iter_name, TRAIN_NAME)
    train_path = os.path.abspath(train_path)
    models = sorted(glob.glob(os.path.join(train_path, "graph*pb")))
    task_model_list = []
    for ii in models:
        task_model_list.append(os.path.join("..", os.path.basename(ii)))
    work_path = os.path.join(iter_name, model_devi_name)
    try:
        mdata["deepmd_version"]
    except KeyError:
        mdata = set_version(mdata)
    deepmd_version = mdata["deepmd_version"]

    sys_counter = 0
    for ss in conf_systems:
        conf_counter = 0
        task_counter = 0
        for cc in ss:
            sys_rev = cur_job.get("sys_rev_mat", None)
            total_rev_keys = rev_keys
            total_rev_mat = rev_mat
            total_num_lmp = num_lmp
            if sys_rev is not None:
                total_rev_mat = []
                sys_rev_keys, sys_rev_mat, sys_num_lmp = parse_cur_job_sys_revmat(
                    cur_job, sys_idx=sys_idx[sys_counter], use_plm=use_plm
                )
                _lmp_keys = rev_keys[:num_lmp] + sys_rev_keys[:sys_num_lmp]
                if use_plm:
                    _plm_keys = rev_keys[num_lmp:] + sys_rev_keys[sys_num_lmp:]
                    _lmp_keys += _plm_keys
                total_rev_keys = _lmp_keys
                total_num_lmp = num_lmp + sys_num_lmp
                for pub in rev_mat:
                    for pri in sys_rev_mat:
                        _lmp_mat = pub[:num_lmp] + pri[:sys_num_lmp]
                        if use_plm:
                            _plm_mat = pub[num_lmp:] + pri[sys_num_lmp:]
                            _lmp_mat += _plm_mat
                        total_rev_mat.append(_lmp_mat)
            print(len(total_rev_mat))
            for ii in range(len(total_rev_mat)):
                total_rev_item = total_rev_mat[ii]
                task_name = make_model_devi_task_name(
                    sys_idx[sys_counter], task_counter
                )
                conf_name = (
                    make_model_devi_conf_name(sys_idx[sys_counter], conf_counter)
                    + ".lmp"
                )
                task_path = os.path.join(work_path, task_name)
                # create task path
                create_path(task_path)
                model_devi_merge_traj = jdata.get("model_devi_merge_traj", False)
                if not model_devi_merge_traj:
                    create_path(os.path.join(task_path, "traj"))
                # link conf
                loc_conf_name = "conf.lmp"
                os.symlink(
                    os.path.join(os.path.join("..", "confs"), conf_name),
                    os.path.join(task_path, loc_conf_name),
                )
                cwd_ = os.getcwd()
                # chdir to task path
                os.chdir(task_path)
                shutil.copyfile(lmp_templ, "input.lammps")
                # revise input of lammps
                with open("input.lammps") as fp:
                    lmp_lines = fp.readlines()
                # only revise the line "pair_style deepmd" if the user has not written the full line (checked by then length of the line)
                template_has_pair_deepmd = 1
                for line_idx, line_context in enumerate(lmp_lines):
                    if (
                        (line_context[0] != "#")
                        and ("pair_style" in line_context)
                        and ("deepmd" in line_context)
                    ):
                        template_has_pair_deepmd = 0
                        template_pair_deepmd_idx = line_idx
                if template_has_pair_deepmd == 0:
                    if Version(deepmd_version) < Version("1"):
                        if len(lmp_lines[template_pair_deepmd_idx].split()) != (
                            len(models)
                            + len(["pair_style", "deepmd", "10", "model_devi.out"])
                        ):
                            lmp_lines = revise_lmp_input_model(
                                lmp_lines,
                                task_model_list,
                                trj_freq,
                                deepmd_version=deepmd_version,
                            )
                    else:
                        if len(lmp_lines[template_pair_deepmd_idx].split()) != (
                            len(models)
                            + len(
                                [
                                    "pair_style",
                                    "deepmd",
                                    "out_freq",
                                    "10",
                                    "out_file",
                                    "model_devi.out",
                                ]
                            )
                        ):
                            lmp_lines = revise_lmp_input_model(
                                lmp_lines,
                                task_model_list,
                                trj_freq,
                                deepmd_version=deepmd_version,
                            )
                # use revise_lmp_input_model to raise error message if "part_style" or "deepmd" not found
                else:
                    lmp_lines = revise_lmp_input_model(
                        lmp_lines,
                        task_model_list,
                        trj_freq,
                        deepmd_version=deepmd_version,
                    )
                if not ("dimer" == jdata["model_devi_engine"]):
                  lmp_lines = revise_lmp_input_dump(
                    lmp_lines, trj_freq, model_devi_merge_traj
                  )
                #lmp_lines = revise_lmp_input_dump(
                 #   lmp_lines, trj_freq, model_devi_merge_traj
                #)
                lmp_lines = revise_by_keys(
                    lmp_lines,
                    total_rev_keys[:total_num_lmp],
                    total_rev_item[:total_num_lmp],
                )
                # revise input of plumed
                if use_plm:
                    lmp_lines = revise_lmp_input_plm(lmp_lines, "input.plumed")
                    shutil.copyfile(plm_templ, "input.plumed")
                    with open("input.plumed") as fp:
                        plm_lines = fp.readlines()
                    # allow using the same list as lmp
                    # user should not use the same key name for plm
                    plm_lines = revise_by_keys(
                        plm_lines, total_rev_keys, total_rev_item
                    )
                    with open("input.plumed", "w") as fp:
                        fp.write("".join(plm_lines))
                    if use_plm_path:
                        shutil.copyfile(plm_path_templ, "plmpath.pdb")
                # dump input of lammps
                with open("input.lammps", "w") as fp:
                    fp.write("".join(lmp_lines))
                with open("job.json", "w") as fp:
                    job = {}
                    for ii, jj in zip(total_rev_keys, total_rev_item):
                        job[ii] = jj
                    json.dump(job, fp, indent=4)
                os.chdir(cwd_)
                task_counter += 1
            conf_counter += 1
        sys_counter += 1


def parse_cur_job_revmat(cur_job, use_plm=False):
    templates = [cur_job["template"]["lmp"]]
    if use_plm:
        templates.append(cur_job["template"]["plm"])
    revise_keys = []
    revise_values = []
    if "rev_mat" not in cur_job.keys():
        cur_job["rev_mat"] = {}
    if "lmp" not in cur_job["rev_mat"].keys():
        cur_job["rev_mat"]["lmp"] = {}
    for ii in cur_job["rev_mat"]["lmp"].keys():
        revise_keys.append(ii)
        revise_values.append(cur_job["rev_mat"]["lmp"][ii])
    n_lmp_keys = len(revise_keys)
    if use_plm:
        if "plm" not in cur_job["rev_mat"].keys():
            cur_job["rev_mat"]["plm"] = {}
        for ii in cur_job["rev_mat"]["plm"].keys():
            revise_keys.append(ii)
            revise_values.append(cur_job["rev_mat"]["plm"][ii])
    revise_matrix = expand_matrix_values(revise_values)
    return revise_keys, revise_matrix, n_lmp_keys


def expand_matrix_values(target_list, cur_idx=0):
    nvar = len(target_list)
    if cur_idx == nvar:
        return [[]]
    else:
        res = []
        prev = expand_matrix_values(target_list, cur_idx + 1)
        for ii in target_list[cur_idx]:
            tmp = copy.deepcopy(prev)
            for jj in tmp:
                jj.insert(0, ii)
                res.append(jj)
        return res
