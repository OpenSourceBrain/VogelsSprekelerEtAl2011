#!/usr/bin/sh

# This script installs the packages required for post processing, and downloads the post processing scripts.

if [ "$(whoami)" != "root" ];
then
    echo "This script must be run using sudo to be able to install certain packages. Please run it as: sudo ./setup-Fedora26.sh"
    exit 1
fi


echo "Setting up postprocess dependencies.."
dnf install python3-numpy python3-matplotlib gnuplot python3-pandas git

echo "Fetching scripts.."
git clone https://github.com/sanjayankur31/Sinha2016-scripts.git; git checkout develop
