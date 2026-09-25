# Copyright (C) 2026 Neil Cornish.
# SPDX-License-Identifier: GPL-3.0-or-later

CC ?= cc
CFLAGS ?= -O3 -std=c99
GSL_CFLAGS := $(shell gsl-config --cflags)
GSL_LIBS := $(shell gsl-config --libs)

COMMON = IMRPhenomT.c IMRPhenomTHM.c PhenomTHM_TDI.c
TPHM = IMRPhenomTPHM.c IMRPhenomTPHM_Precession.c IMRPhenomTPHM_Rotation.c PhenomTPHM_TDI.c

.PHONY: all
all: PhenomTHM_TDI PhenomTPHM_TDI_test

PhenomTHM_TDI: $(COMMON) IMRPhenomT.h IMRPhenomTHM.h PhenomTHM_TDI.h LALSimIMRPhenomTHM_fits.c wdm.h
	$(CC) $(CFLAGS) $(GSL_CFLAGS) -o $@ $(COMMON) $(GSL_LIBS) -lm

PhenomTPHM_TDI_test: PhenomTPHM_TDI_test.c $(TPHM) $(COMMON) IMRPhenomTPHM.h IMRPhenomTPHM_Precession.h IMRPhenomTPHM_Rotation.h PhenomTPHM_TDI.h
	$(CC) $(CFLAGS) -DPHENOMTHM_TDI_LIBRARY $(GSL_CFLAGS) -o $@ PhenomTPHM_TDI_test.c $(TPHM) $(COMMON) $(GSL_LIBS) -lm
