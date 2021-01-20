#!/bin/bash
#!/bin/bash -l
#SBATCH --nodes=1
#SBATCH --time=24:00:00
#SBATCH --exclusive
#SBATCH --constraint=cpu
#SBATCH --mem=0
#SBATCH --partition=prod
#SBATCH --account=proj83
#SBATCH --output="psp-configs-%x-%j.out"
#SBATCH --error="psp-configs-%x-%j.err"

#module load py-bluepy/0.14.6
#which python

echo LOGLEVEL=$LOGLEVEL

venv=~/.vmgr_repo/py37/bin/activate

source $venv

echo "Generate PSP configs"
python run.py $1 --output $2

