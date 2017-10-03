#!/usr/bin/sh

# Install required dependencies on Fedora, and install NEST.
# NOTE: Requires superuser/administrative access - i.e., the script should be run using sudo - sudo ./setup-Fedora26.sh

NESTINSTALLDIR=/opt/nest-vogels-sprekeler-2011
THISARCH=$(uname --hardware-platform)
SRC="nest-2.12.0"
TARFILE="$SRC.tar.gz"
export MODULEPATH=/etc/scl/modulefiles:/etc/scl/modulefiles:/usr/share/Modules/modulefiles:/etc/modulefiles:/usr/share/modulefiles

if [ "$(whoami)" != "root" ];
then
    echo "This script must be run using sudo to be able to install certain packages. Please run it as: sudo ./setup-Fedora26.sh"
    exit 1
fi

echo "Installing build dependencies for NEST.."
dnf install boost-devel boost-python3 openmpi-devel cmake gsl-devel readline-devel python3-Cython boost-openmpi-devel python3-devel
if [[ "x86_64" == "$THISARCH" ]]; then
    export PKG_CONFIG_PATH=/usr/lib64/openmpi/lib/pkgconfig:$PKG_CONFIG_PATH
    export LD_LIBRARY_PATH=/usr/lib64/openmpi/lib:$LD_LIBRARY_PATH
    export PATH=/usr/lib64/openmpi/bin:$PATH
    export MPI_HOME=/usr/lib64/openmpi
    export MPI_PYTHON_SITEARCH=/usr/lib64/python2.7/site-packages/openmpi
    export MPI_PYTHON2_SITEARCH=/usr/lib64/python2.7/site-packages/openmpi
    export MPI_PYTHON3_SITEARCH=/usr/lib64/python3.6/site-packages/openmpi
    export MPI_SYSCONFIG=/etc/openmpi-x86_64
else
    export PKG_CONFIG_PATH=/usr/lib/openmpi/lib/pkgconfig:$PKG_CONFIG_PATH
    export LD_LIBRARY_PATH=/usr/lib/openmpi/lib:$LD_LIBRARY_PATH
    export PATH=/usr/lib/openmpi/bin:$PATH
    export MPI_HOME=/usr/lib/openmpi
    export MPI_PYTHON_SITEARCH=/usr/lib/python2.7/site-packages/openmpi
    export MPI_PYTHON2_SITEARCH=/usr/lib/python2.7/site-packages/openmpi
    export MPI_PYTHON3_SITEARCH=/usr/lib/python3.6/site-packages/openmpi
    export MPI_SYSCONFIG=/etc/openmpi-x86
fi
export MPI_CXX_LIBRARIES=$(pkgconf --cflags ompi-cxx) 


echo "Cleaning up $NESTINSTALLDIR.."
rm -rf $NESTINSTALLDIR

echo "Setting up NEST 2.12.0.."
if [[ ! -e "$TARFILE" ]]; then
    echo "Source tar not found. Downloading from GitHub"
    wget https://github.com/nest/nest-simulator/releases/download/v2.12.0/nest-2.12.0.tar.gz 
fi

echo "Unzipping sources.."
rm -rf $SRC
tar -xf $TARFILE


echo "Building and installing NEST. Takes a while - get a coffee.."
pushd nest-2.12.0
    cmake -DCMAKE_INSTALL_PREFIX:PATH=$NESTINSTALLDIR -Dwith-python:STRING=3 -Dwith-mpi:BOOL=ON -Dwith-debug:BOOL=ON .
    make; make install; source /$NESTINSTALLDIR/bin/nest_vars.sh
popd

echo "NEST installed!"
