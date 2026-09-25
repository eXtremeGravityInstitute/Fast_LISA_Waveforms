# Copyright (C) 2026 Neil Cornish.
# SPDX-License-Identifier: GPL-3.0-or-later
CC ?= cc
CFLAGS ?= -O3 -std=c99
GSL_CFLAGS := $(shell gsl-config --cflags)
GSL_LIBS := $(shell gsl-config --libs)

response: response.c tdi2_response.c tdi2_response.h wdm.h
	$(CC) $(CFLAGS) $(GSL_CFLAGS) -o $@ response.c tdi2_response.c $(GSL_LIBS) -lm
