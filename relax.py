# -*- coding: utf-8 -*-
"""
Created on Thu Jul  6 14:03:27 2023

@author: Bangchen Yin

The dimer engine of model devi
"""
import os
import argparse
from ase.dimer import DimerControl, MinModeAtoms, MinModeTranslate
from ase.constraints import FixAtoms
from ase.io import read
from ase.io.trajectory import Trajectory
from deepmd.infer import calc_model_devi
from deepmd.calculator import DP
from ase.optimize import BFGS
import numpy as np
def write_lamptrj(atoms,forces,step):
    f_conf=open("conf.lmp","r")
    lines=f_conf.readlines()
    for line in lines:
        if "atom types" in line:
            n_atoms=int(line.split()[0])
    fname="./traj/"+str(step)+".lammpstrj"
   
    posname="./traj/"+str(step)+".poscar"
    f=open(fname,"w")
    f.write("ITEM: TIMESTEP\n")
    f.write(str(step)+'\n')
    f.write("ITEM: NUMBER OF ATOMS\n")
    f.write(str(len(atoms))+'\n')
    f.write("ITEM: BOX BOUNDS xy xz yz pp pp pp\n")
    o_line="0.00 "+str(atoms.cell[0][0])+" 0.00\n"
    t_line="0.00 "+str(atoms.cell[1][1])+" 0.00\n"
    th_line="0.00 "+str(atoms.cell[2][2])+" 0.00\n"
    f.write(o_line)
    f.write(t_line)
    f.write(th_line)
    print(atoms.cell)
    f.write("ITEM: ATOMS id type x y z fx fy fz\n")
    count=1
    pre=""
    for atom in atoms:
        xyz=atom.position
        f.write(str(atom.index+1)+" ")
        if(atom.symbol==pre)or(pre==""):
          f.write(str(count)+" ")
        else:
            count+=1
            f.write(str(count)+" ")
        pre=atom.symbol
        for ii in xyz:
            f.write(str(ii)+" ")
        for idf in forces[atom.index]:
            f.write(str(idf)+" ")
        f.write("\n")


def main(args):
    current_directory = os.getcwd()
    parent_directory_1 = os.path.dirname(current_directory)
    parent_directory_2 = os.path.dirname(parent_directory_1)
    pa3 = os.path.dirname(parent_directory_2)
    atoms=read(pa3+"/Data/devi/00/POSCAR")
    mask =[]
    for i in range(54):
       if i<48:
         mask.append(0)
       else:
         mask.append(1)
    if args.fix:
      fix=1
        #TODO:YOU can modify different ways to fix atoms here, if you want to fix atoms[i], then just make the mask[i]=True
     # mask=[atom.index<args.index for atom in atoms]
      #atoms.set_constraint(FixAtoms(mask=mask))
    dp=DP(model="../graph.001.pb")
    atoms.calc = dp
        #TODO: YOU can also change the ways of displacement for your own example, we basically provide the gaussian displacement.
        #More information could be found at https://wiki.fysik.dtu.dk/ase/ase/dimer.html
    with DimerControl(initial_eigenmode_method='displacement',maximum_translation=0.05,
                      displacement_method=args.method, mask=mask,dimer_separation=0.01,logfile=None, #TODO: make it adjustable
                      ) as d_control:
        d_atoms = MinModeAtoms(atoms, d_control)
        d_atoms.displace()
        with MinModeTranslate(d_atoms, trajectory='dimer_method.traj',
                              logfile='log') as dim_rlx:
            dim_rlx.run(fmax=0.02,steps=10)
            d_atoms.write('POSCAR')
            IS=read("POSCAR")
            FS=read("POSCAR")
            IS.calc=dp
            FS.calc=dp
            IS.positions+=d_atoms.eigenmodes[0]
            FS.positions-=d_atoms.eigenmodes[0]
            dyn1=BFGS(IS,trajectory="IS.traj")
            dyn2=BFGS(FS,trajectory="FS.traj")
            dyn1.run(fmax=0.05,steps=10)
            dyn2.run(fmax=0.05,steps=10)
            IS.write("IS.vasp")
            FS.write("FS.vasp")
            traj1 = Trajectory('dimer_method.traj')
            traj2=Trajectory('FS.traj')
            traj=[]
            for atoms in traj1:
                traj.append(atoms)
            for atoms in traj2:
                traj.append(atoms)
            if not os.path.exists("traj"):
              os.system("mkdir traj")
            traj=traj1
            count=0
            box=[]
            coor=[]
            atype=[]
            k=0
            pre=""
            for atom in atoms:
               if atom.symbol != pre:
                 pre=atom.symbol
                 k+=1
               atype.append(k-1)
            for atoms in traj:
                if len(box)==0:
                    box= np.array(atoms.cell).reshape([1,-1])[0]
                else:
                   box=np.append(box,np.array(atoms.cell).reshape([1,-1])[0])
                if len(coor)==0:
                    coor=np.array(atoms.positions).reshape([1,-1])[0]
                else:
                    coor=np.append(coor,np.array(atoms.positions).reshape([1,-1])[0])
              
                forces= np.zeros([len(atoms),3])
                write_lamptrj(atoms,forces,count)
                count+=1
            coor=np.array([coor])
            box=np.array([box])
            atype=np.array(atype)
            from deepmd.infer import DeepPot as DPot
            graphs = [DPot("../graph.000.pb"), DPot("../graph.001.pb")]
            model_devi = calc_model_devi(coor, box, atype, graphs,fname="model_devi.out",frequency=1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("fix", type=bool,default=False, help="Whether to fix the atoms")
    parser.add_argument("index", type=int,default=-1, help="atom's tag<index will be fixed")
    parser.add_argument("trans",type=float,default=0.1,help="max step size")
    parser.add_argument("center",type=int,default=0,help="displace center")
    parser.add_argument("radius",type=float,default=2,help="displace radius")
    parser.add_argument("method",type=str,default="gauss",help="displacement method")


    args = parser.parse_args()
    main(args)

