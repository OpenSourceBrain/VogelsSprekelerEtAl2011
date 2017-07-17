.PHONY: clean all install
CC = mpic++

DISTRO:= $(shell source /etc/os-release && echo $$NAME)
$(info $$DISTRO is [${DISTRO}])

ifeq ($(DISTRO),Fedora)
	LDFLAGS=-lboost_program_options -lboost_mpi -lboost_serialization  -L/usr/lib64/mpich/lib/ -lauryn
	CXXFLAGS= -g -pipe -Wall -Werror=format-security -Wp,-D_FORTIFY_SOURCE=2 -fexceptions -fstack-protector-strong --param=ssp-buffer-size=4 -grecord-gcc-switches -m64 -mtune=generic -std=c++11 -O3 -ffast-math -pedantic -I/usr/include -I/usr/include/mpich-x86_64
else
	CFLAGS=-std=c++11 -pipe -O3 -march=native -ffast-math -pedantic -I/usr/include
	LDFLAGS=-lboost_program_options -lboost_mpi -lboost_serialization
endif

all: vogels

vogels:
	$(CC) $(CXXFLAGS) vogels.cpp -o vogels $(LDFLAGS)

install: vogels
	mkdir -p ../run/bin/
	cp -v vogels ../run/bin/
clean:
	rm -fv vogels ../run/bin/vogels
