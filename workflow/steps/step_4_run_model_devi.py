# step_4_run_model_devi.py
# Created by Yichao


# region
import os
import glob
import json
import logging

from packaging.version import Version 
from dpgen.generator.lib.run_calypso import run_calypso_model_devi

from packaging.version import Version


from dpgen.generator.lib.utils import make_iter_name
from dpgen.dispatcher.Dispatcher import make_submission


from config.config import *
# endregion


def run_model_devi(iter_index, jdata, mdata, base_dir):

    model_devi_engine = jdata.get("model_devi_engine", "lammps")
    if model_devi_engine != "calypso":
        run_md_model_devi(iter_index, jdata, mdata, base_dir)
    else:
        run_calypso_model_devi(iter_index, jdata, mdata)




def run_md_model_devi(iter_index, jdata, mdata, base_dir):
    logging.info("Running MD model deviation for iteration %d", iter_index)
    
    model_devi_exec = mdata["model_devi_command"]
    logging.info("Model deviation command: %s", model_devi_exec)

    model_devi_group_size = mdata["model_devi_group_size"]
    model_devi_resources = mdata["model_devi_resources"]
    logging.info("Model deviation group size: %d", model_devi_group_size)
    logging.info("Model deviation resources: %s", model_devi_resources)
    
    use_plm = jdata.get("model_devi_plumed", False)
    use_plm_path = jdata.get("model_devi_plumed_path", False)
    model_devi_merge_traj = jdata.get("model_devi_merge_traj", False)
    logging.info("Use PLUMED: %s", use_plm)
    logging.info("Use PLUMED path: %s", use_plm_path)
    logging.info("Merge trajectories: %s", model_devi_merge_traj)

    iter_name = make_iter_name(iter_index)
    logging.debug("Generated iteration name: %s", iter_name)









    work_path = os.path.join(base_dir, iter_name, model_devi_name,'confs')
    logging.debug("Joining paths to create work_path...")
    logging.debug("  base_dir:        %s", base_dir)
    logging.debug("  iter_name:       %s", iter_name)
    logging.debug("  model_devi_name: %s", model_devi_name)
    logging.debug("  Joined work_path: %s", work_path)

    assert os.path.isdir(work_path), f"Work path {work_path} does not exist"
    logging.info("Work path: %s", work_path)

    logging.debug("Searching for tasks in work_path...")
    logging.debug("  Work path: %s", work_path)
    logging.debug("  Pattern: task.*")


    all_task = glob.glob(os.path.join(work_path, "task.*"))
    all_task.sort()
    logging.debug("  Found %d tasks:", len(all_task))
    for task in all_task:
        logging.debug("    %s", task)
    logging.info("Found %d tasks in total", len(all_task))




    work_path = os.path.join(base_dir, iter_name, model_devi_name)




    logging.debug("Loading current job from cur_job.json...")
    cur_job_file = os.path.join(work_path, "cur_job.json")
    logging.debug("  Current job file: %s", cur_job_file)
    with open(cur_job_file, "r") as fp:
        cur_job = json.load(fp)
    logging.debug("  Loaded current job: %s", cur_job)
    logging.info("Current job loaded from %s", cur_job_file)

    run_tasks_ = all_task
    logging.debug("Extracting basenames from run_tasks_...")
    run_tasks = [os.path.basename(ii) for ii in run_tasks_]
    logging.debug("  Extracted basenames: %s", run_tasks)
    logging.info("Running tasks: %s", run_tasks)

    logging.info("Searching for model files in pattern 'graph*pb' under working folder %s...", work_path)
    logging.debug("  Work path: %s", work_path)
    logging.debug("  Pattern: graph*pb")
    all_models = glob.glob(os.path.join(work_path, "graph*pb"))
    logging.debug("  Found %d model files:", len(all_models))
    for model in all_models:
        logging.debug("    %s", model)

    # 记录找到的模型文件数量和具体的文件路径列表  
    logging.info("Found %d model files: %s", len(all_models), all_models)

    # 提取模型文件的文件名列表
    model_names = [os.path.basename(ii) for ii in all_models]

    # 记录提取出的模型文件名列表
    logging.info("Extracted model file names: %s", model_names)

    # 打印最终的模型文件数量和文件名列表
    logging.info("Found %d models in total: %s", len(model_names), model_names)










    

    model_devi_engine = jdata.get("model_devi_engine", "lammps")
    logging.info("Model deviation engine: %s", model_devi_engine)
    
    if model_devi_engine == "lammps":
        command = (
            "{ if [ ! -f dpgen.restart.10000 ]; then %s -i input.lammps -v restart 0; else %s -i input.lammps -v restart 1; fi }"
            % (model_devi_exec, model_devi_exec)
        )
        command = "/bin/sh -c '%s'" % command
        commands = [command]
        logging.info("LAMMPS commands: %s", commands)

        forward_files = ["conf.lmp", "input.lammps"]
        backward_files = ["model_devi.out", "model_devi.log"]
        if model_devi_merge_traj:
            backward_files += ["all.lammpstrj"]
        else:
            forward_files += ["traj"]
            backward_files += ["traj"]
        logging.info("LAMMPS forward files: %s", forward_files)
        logging.info("LAMMPS backward files: %s", backward_files)

        if use_plm:
            forward_files += ["input.plumed"]
            backward_files += ["output.plumed", "COLVAR"]
            if use_plm_path:
                forward_files += ["plmpath.pdb"]
            logging.info("PLUMED enabled, updated forward files: %s", forward_files)
            logging.info("PLUMED enabled, updated backward files: %s", backward_files)
            
    elif model_devi_engine == "dimer":
        command = mdata["model_devi_command"] 
        commands = [command]
        logging.info("Dimer commands: %s", commands)

        forward_files = ["conf.lmp", "input.lammps"]
        backward_files = ["model_devi.out", "model_devi.log"]
        if model_devi_merge_traj:
            backward_files += ["all.lammpstrj"]
        else:
            forward_files += ["traj"]
            backward_files += ["traj"]
        logging.info("Dimer forward files: %s", forward_files)
        logging.info("Dimer backward files: %s", backward_files)
        
    elif model_devi_engine == "gromacs":
        gromacs_settings = jdata.get("gromacs_settings", {})
        mdp_filename = gromacs_settings.get("mdp_filename", "md.mdp")
        topol_filename = gromacs_settings.get("topol_filename", "processed.top")
        conf_filename = gromacs_settings.get("conf_filename", "conf.gro")
        index_filename = gromacs_settings.get("index_filename", "index.raw")
        type_filename = gromacs_settings.get("type_filename", "type.raw")
        ndx_filename = gromacs_settings.get("ndx_filename", "")
        ref_filename = gromacs_settings.get("ref_filename", "em.tpr")
        deffnm = gromacs_settings.get("deffnm", "deepmd")
        maxwarn = gromacs_settings.get("maxwarn", 1)
        traj_filename = gromacs_settings.get("traj_filename", "deepmd_traj.gro")
        grp_name = gromacs_settings.get("group_name", "Other")
        trj_freq = cur_job.get("trj_freq", 10)
        logging.info("GROMACS settings: %s", gromacs_settings)

        command = "%s grompp -f %s -p %s -c %s -o %s -maxwarn %d" % (
            model_devi_exec,
            mdp_filename,
            topol_filename,
            conf_filename,
            deffnm,
            maxwarn,
        )
        command += "&& %s mdrun -deffnm %s -cpi" % (model_devi_exec, deffnm)
        if ndx_filename:
            command += f'&& echo -e "{grp_name}\\n{grp_name}\\n" | {model_devi_exec} trjconv -s {ref_filename} -f {deffnm}.trr -n {ndx_filename} -o {traj_filename} -pbc mol -ur compact -center'
        else:
            command += (
                '&& echo -e "%s\\n%s\\n" | %s trjconv -s %s -f %s.trr -o %s -pbc mol -ur compact -center'
                % (
                    grp_name,
                    grp_name,
                    model_devi_exec,
                    ref_filename,
                    deffnm,
                    traj_filename,
                )
            )
        command += "&& if [ ! -d traj ]; then \n mkdir traj; fi\n"
        command += f"python -c \"import dpdata;system = dpdata.System('{traj_filename}', fmt='gromacs/gro'); [system.to_gromacs_gro('traj/%d.gromacstrj' % (i * {trj_freq}), frame_idx=i) for i in range(system.get_nframes())]; system.to_deepmd_npy('traj_deepmd')\""
        command += f"&& dp model-devi -m ../graph.000.pb ../graph.001.pb ../graph.002.pb ../graph.003.pb -s traj_deepmd -o model_devi.out -f {trj_freq}"
        commands = [command]
        logging.info("GROMACS commands: %s", commands)

        forward_files = [
            mdp_filename,
            topol_filename,
            conf_filename,
            index_filename,
            ref_filename,
            type_filename,
            "input.json",
            "job.json",
        ]
        if ndx_filename:
            forward_files.append(ndx_filename)
        backward_files = [
            "%s.tpr" % deffnm,
            "%s.log" % deffnm,
            traj_filename,
            "model_devi.out",
            "traj",
            "traj_deepmd",
        ]
        logging.info("GROMACS forward files: %s", forward_files)
        logging.info("GROMACS backward files: %s", backward_files)
        
    elif model_devi_engine == "amber":
        commands = [
            (
                "TASK=$(basename $(pwd)) && "
                "SYS1=${TASK:5:3} && "
                "SYS=$((10#$SYS1)) && "
            )
            + model_devi_exec
            + (
                " -O -p ../qmmm$SYS.parm7 -c init.rst7 -i ../init$SYS.mdin -o rc.mdout -r rc.rst7 -x rc.nc -inf rc.mdinfo -ref init.rst7"
            )
        ]
        logging.info("AMBER commands: %s", commands)
        
        forward_files = ["init.rst7", "TEMPLATE.disang"]
        backward_files = ["rc.mdout", "rc.nc", "rc.rst7", "TEMPLATE.dumpave"]
        model_names.extend(["qmmm*.parm7", "init*.mdin"])
        logging.info("AMBER forward files: %s", forward_files)
        logging.info("AMBER backward files: %s", backward_files)
        logging.info("AMBER model names: %s", model_names)

    cwd = os.getcwd()
    logging.info("Current working directory: %s", cwd)

    user_forward_files = mdata.get("model_devi" + "_user_forward_files", [])
    forward_files += [os.path.basename(file) for file in user_forward_files]
    backward_files += mdata.get("model_devi" + "_user_backward_files", [])
    logging.info("User-defined forward files: %s", user_forward_files)
    logging.info("Updated forward files: %s", forward_files)
    logging.info("Updated backward files: %s", backward_files)
    
    api_version = mdata.get("api_version", "1.0")
    logging.info("API version: %s", api_version)

    if len(run_tasks) == 0:
        logging.error("No tasks to run for model deviation")
        raise RuntimeError(
            "run_tasks for model_devi should not be empty! Please check your files."
        )

    if Version(api_version) < Version("1.0"):
        logging.error("API version %s is no longer supported", api_version)
        raise RuntimeError(
            "API version %s has been removed. Please upgrade to 1.0." % api_version
        )

    elif Version(api_version) >= Version("1.0"):
        work_path = os.path.join(work_path, 'confs')

        logging.debug("Creating submission with the following parameters:")
        logging.debug("  model_devi_machine: %s", mdata["model_devi_machine"])
        logging.debug("  model_devi_resources: %s", mdata["model_devi_resources"])
        logging.debug("  commands: %s", commands)
        logging.debug("  work_path: %s", work_path)
        logging.debug("  run_tasks: %s", run_tasks)
        logging.debug("  group_size: %d", model_devi_group_size)
        logging.debug("  forward_common_files: %s", model_names)
        logging.debug("  forward_files: %s", forward_files)
        logging.debug("  backward_files: %s", backward_files)
        logging.debug("  outlog: model_devi.log")
        logging.debug("  errlog: model_devi.log")

        submission = make_submission(
            mdata["model_devi_machine"],
            mdata["model_devi_resources"],
            commands=commands,
            work_path=work_path,
            run_tasks=run_tasks,
            group_size=model_devi_group_size,
            forward_common_files=model_names,
            forward_files=forward_files,
            backward_files=backward_files,
            outlog="model_devi.log",
            errlog="model_devi.log",
        )

        logging.info("Submission object created: %s", submission)

        logging.info("Running submission...")
        submission.run_submission()
        logging.info("Model deviation submission completed")

