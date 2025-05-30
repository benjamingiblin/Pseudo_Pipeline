#CAMB Makefile

#Set FISHER=Y to compile bispectrum fisher matrix code
FISHER=

# !!! GIBLIN YOU MANUALLY ADDED THE FOLLOWING LINE TO FORCE THE COMPILER TO
# !!! TO BE gfortran ... REMOVE THIS LINE TO ALLOW THE CODE TO LOOK FOR ifort
# !!! IN WHICH CASE IT WILL SET COMPILER TO ifort BUT WILL FAIL TO COMPILE ... 
COMPILER = gfortran

#Will detect ifort/gfortran or edit for your compiler
ifneq ($(COMPILER),gfortran)
$(info $$COMPILER is [${COMPILER}])
ifortErr = $(shell which ifort >/dev/null; echo $$?)
$(info $$ifortErr is [${ifortErr}])
else
$(info $$COMPILER is [${COMPILER}])
ifortErr = 1
endif

ifeq "$(ifortErr)" "0"

#Intel compiler
# For OSX replace shared by dynamiclib
F90C     = ifort
FFLAGS = -fast -W0 -WB -fpp 
SFFLAGS = -shared -fpic
DEBUGFLAGS = -g -check all -check noarg_temp_created -traceback -fpp -fpe0

ifortVer_major = $(shell ifort -v 2>&1 | cut -d " " -f 3 | cut -d. -f 1)
ifeq ($(shell test $(ifortVer_major) -gt 15; echo $$?),0)
FFLAGS+= -qopenmp
DEBUGFLAGS+= -qopenmp
else
FFLAGS+= -openmp -vec_report0
DEBUGFLAGS+= -openmp
endif

## This is flag is passed to the Fortran compiler allowing it to link C++ if required (not usually):
F90CRLINK = -cxxlib
MODOUT = -module $(OUTPUT_DIR)
SMODOUT = -module $(DLL_DIR)
ifneq ($(FISHER),)
FFLAGS += -mkl
endif

else
gfortErr = $(shell which gfortran >/dev/null; echo $$?)
$(info $$gfortErr is [${gfortErr}])
ifeq "$(gfortErr)" "0"
COMPILER = gfortran
#Gfortran compiler:
#The options here work in v4.6+. Python wrapper needs v4.9+.
F90C     = gfortran
SFFLAGS =  -shared -fPIC

FFLAGS =  -O3 -fopenmp -ffast-math -fmax-errors=4
DEBUGFLAGS = -cpp -g -fbounds-check -fbacktrace -ffree-line-length-none -fmax-errors=4 -ffpe-trap=invalid,overflow,zero
MODOUT =  -J$(OUTPUT_DIR)
SMODOUT = -J$(DLL_DIR)

ifneq ($(shell uname -s),Darwin)
#native optimization does not work on Mac
FFLAGS+=-march=native
endif
endif
endif

IFLAG = -I

#G95 compiler
#F90C   = g95
#FFLAGS = -O2

#SGI, -mp toggles multi-processor. Use -O2 if -Ofast gives problems.
#F90C     = f90
#FFLAGS  = -Ofast -mp

#Digital/Compaq fortran, -omp toggles multi-processor
#F90C    = f90
#FFLAGS  = -omp -O4 -arch host -math_library fast -tune host -fpe1

#Absoft ProFortran, single processor:
#F90C     = f95
#FFLAGS = -O2 -cpu:athlon -s -lU77 -w -YEXT_NAMES="LCS" -YEXT_SFX="_"

#NAGF95, single processor:
#F90C     = f95
#FFLAGS = -DNAGF95 -O3

#PGF90
#F90C = pgf90
#FFLAGS = -O2 -DESCAPEBACKSLASH -Mpreprocess

#Sun V880
#F90C = mpf90
#FFLAGS =  -O4 -openmp -ftrap=%none -dalign

#Sun parallel enterprise:
#F90C     = f95
#FFLAGS =  -O2 -xarch=native64 -openmp -ftrap=%none
#try removing -openmp if get bus errors. -03, -04 etc are dodgy.

#IBM XL Fortran, multi-processor (run gmake)
#F90C     = xlf90_r
#FFLAGS  = -DESCAPEBACKSLASH -DIBMXL -qsmp=omp -qsuffix=f=f90:cpp=F90 -O3 -qstrict -qarch=pwr3 -qtune=pwr3

#Settings for building camb_fits
#Location of FITSIO and name of library
FITSDIR       ?= /usr/local/lib
FITSLIB       = cfitsio
#Location of HEALPIX for building camb_fits
HEALPIXDIR    ?= /usr/local/healpix

ifneq ($(FISHER),)
FFLAGS += -DFISHER
EXTCAMBFILES = Matrix_utils.o
else
EXTCAMBFILES =
endif

DEBUGFLAGS ?= FFLAGS
Debug: FFLAGS=$(DEBUGFLAGS)

include ./Makefile_main
