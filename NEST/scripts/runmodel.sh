MPIRANKS=$(grep -c ^processor /proc/cpuinfo)
OUTPUTDIR=../simrun
NESTINSTALLDIR=/opt/nest-vogels-sprekeler-2011/

source /opt/nest-vogels-sprekeler-2011/bin/nest_vars.sh
mkdir $OUTPUTDIR
cd $OUTPUTDIR
echo -n "Running simulation in: " `pwd`
mpiexec -n $MPIRANKS python3 ../VogelsSprekeler.py
echo "Simulation complete.."
