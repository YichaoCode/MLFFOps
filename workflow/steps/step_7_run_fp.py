# workflow/steps/step_7_run_fp.py
# Created by Yichao


# region
import os
import glob
from config.config import *

from packaging.version import Version 


from dpgen.generator.lib.utils import make_iter_name


from fp_interface.fp_vasp import _vasp_check_fin



from dpgen.dispatcher.Dispatcher import make_submission

# endregion




def run_fp(iter_index, jdata, mdata, base_dir):
    fp_style = jdata["fp_style"]
    fp_pp_files = jdata.get("fp_pp_files", [])

    if fp_style == "vasp":
        forward_files = ["POSCAR", "INCAR", "POTCAR", "KPOINTS"]
        # forward_files = ["input/POSCAR", "input/INCAR", "input/POTCAR", "input/KPOINTS"]

        backward_files = ["fp.log", "OUTCAR", "vasprun.xml"]
        # Move cvasp interface to jdata
        if ("cvasp" in jdata) and (jdata["cvasp"] == True):
            mdata["fp_resources"]["cvasp"] = True
        if ("cvasp" in mdata["fp_resources"]) and (
            mdata["fp_resources"]["cvasp"] == True
        ):
            dlog.info("cvasp is on !")
            forward_files.append("cvasp.py")
            forward_common_files = []
        else:
            forward_common_files = []
        run_fp_inner(
            iter_index,
            jdata,
            mdata,
            forward_files,
            backward_files,
            _vasp_check_fin,
            forward_common_files=forward_common_files,
            base_dir=base_dir
        )
    elif fp_style == "pwscf":
        forward_files = ["input"] + fp_pp_files
        backward_files = ["output"]
        run_fp_inner(
            iter_index,
            jdata,
            mdata,
            forward_files,
            backward_files,
            _qe_check_fin,
            log_file="output",
            base_dir=base_dir
        )
    elif fp_style == "abacus":
        fp_params = {}
        if "user_fp_params" in jdata.keys():
            fp_params = jdata["user_fp_params"]
        elif "fp_incar" in jdata.keys():
            fp_input_path = jdata["fp_incar"]
            assert os.path.exists(fp_input_path)
            fp_input_path = os.path.abspath(fp_input_path)
            fp_params = get_abacus_input_parameters(fp_input_path)
        forward_files = ["INPUT", "STRU"]
        if "kspacing" not in fp_params.keys():
            forward_files = ["INPUT", "STRU", "KPT"]
        forward_files += fp_pp_files
        if "fp_orb_files" in jdata:
            forward_files += jdata["fp_orb_files"]
        if "fp_dpks_descriptor" in jdata:
            forward_files.append(jdata["fp_dpks_descriptor"])
        if "user_fp_params" in jdata:
            if "deepks_model" in jdata["user_fp_params"]:
                forward_files.append(jdata["user_fp_params"]["deepks_model"])
        backward_files = ["output", "OUT.ABACUS"]
        run_fp_inner(
            iter_index,
            jdata,
            mdata,
            forward_files,
            backward_files,
            _abacus_scf_check_fin,
            log_file="output",
            base_dir=base_dir
        )
    elif fp_style == "siesta":
        forward_files = ["input"] + fp_pp_files
        backward_files = ["output"]
        run_fp_inner(
            iter_index,
            jdata,
            mdata,
            forward_files,
            backward_files,
            _siesta_check_fin,
            log_file="output",
            base_dir=base_dir
        )
    elif fp_style == "gaussian":
        forward_files = ["input"]
        backward_files = ["output"]
        run_fp_inner(
            iter_index,
            jdata,
            mdata,
            forward_files,
            backward_files,
            _gaussian_check_fin,
            log_file="output",
            base_dir=base_dir
        )
    elif fp_style == "cp2k":
        forward_files = ["input.inp", "coord.xyz"]
        backward_files = ["output"]
        run_fp_inner(
            iter_index,
            jdata,
            mdata,
            forward_files,
            backward_files,
            _cp2k_check_fin,
            log_file="output",
            base_dir=base_dir
        )
    elif fp_style == "pwmat":
        forward_files = ["atom.config", "etot.input"] + fp_pp_files
        backward_files = ["REPORT", "OUT.MLMD", "output"]
        run_fp_inner(
            iter_index,
            jdata,
            mdata,
            forward_files,
            backward_files,
            _pwmat_check_fin,
            log_file="output",
            base_dir=base_dir
        )
    elif fp_style == "amber/diff":
        forward_files = ["rc.nc"]
        backward_files = [
            "low_level.mdfrc",
            "low_level.mdout",
            "high_level.mdfrc",
            "high_level.mdout",
            "output",
            "dataset",
        ]
        forward_common_files = [
            "low_level*.mdin",
            "high_level*.mdin",
            "qmmm*.parm7",
            "qm_region",
            "init*.rst7",
        ]
        run_fp_inner(
            iter_index,
            jdata,
            mdata,
            forward_files,
            backward_files,
            None,
            log_file="output",
            forward_common_files=forward_common_files,
            base_dir=base_dir
        )
    else:
        raise RuntimeError("unsupported fp style")


def run_fp_inner(
    iter_index,
    jdata,
    mdata,
    forward_files,
    backward_files,
    check_fin,
    log_file="fp.log",
    forward_common_files=[],
    base_dir=None
):
    fp_command = mdata["fp_command"]
    fp_group_size = mdata["fp_group_size"]
    fp_resources = mdata["fp_resources"]
    mark_failure = fp_resources.get("mark_failure", False)

    iter_name = make_iter_name(iter_index)
    work_path = os.path.join(base_dir, iter_name, fp_name)

    fp_tasks = glob.glob(os.path.join(work_path, "task.*"))
    fp_tasks.sort()
    if len(fp_tasks) == 0:
        return

    fp_style = jdata["fp_style"]
    if fp_style == "amber/diff":
        # firstly get sys_idx
        fp_command = (
            (
                "TASK=$(basename $(pwd)) && "
                "SYS1=${TASK:5:3} && "
                "SYS=$((10#$SYS1)) && "
                'QM_REGION=$(awk "NR==$SYS+1" ../qm_region) &&'
            )
            + fp_command
            + (
                " -O -p ../qmmm$SYS.parm7 -c ../init$SYS.rst7 -i ../low_level$SYS.mdin -o low_level.mdout -r low_level.rst7 "
                "-x low_level.nc -y rc.nc -frc low_level.mdfrc -inf low_level.mdinfo && "
            )
            + fp_command
            + (
                " -O -p ../qmmm$SYS.parm7 -c ../init$SYS.rst7 -i ../high_level$SYS.mdin -o high_level.mdout -r high_level.rst7 "
                "-x high_level.nc -y rc.nc -frc high_level.mdfrc -inf high_level.mdinfo && "
            )
            + (
                'dpamber corr --cutoff %f --parm7_file ../qmmm$SYS.parm7 --nc rc.nc --hl high_level --ll low_level --qm_region "$QM_REGION"'
            )
            % (jdata["cutoff"],)
        )

    fp_run_tasks = fp_tasks
    # for ii in fp_tasks :
    #     if not check_fin(ii) :
    #         fp_run_tasks.append(ii)
    run_tasks = [os.path.basename(ii) for ii in fp_run_tasks]

    user_forward_files = mdata.get("fp" + "_user_forward_files", [])
    forward_files += [os.path.basename(file) for file in user_forward_files]
    backward_files += mdata.get("fp" + "_user_backward_files", [])

    api_version = mdata.get("api_version", "1.0")
    if Version(api_version) < Version("1.0"):
        raise RuntimeError(
            "API version %s has been removed. Please upgrade to 1.0." % api_version
        )

    elif Version(api_version) >= Version("1.0"):
        submission = make_submission(
            mdata["fp_machine"],
            mdata["fp_resources"],
            commands=[fp_command],
            work_path=work_path,
            run_tasks=run_tasks,
            group_size=fp_group_size,
            forward_common_files=forward_common_files,
            forward_files=forward_files,
            backward_files=backward_files,
            outlog=log_file,
            errlog=log_file,
        )
        submission.run_submission()
