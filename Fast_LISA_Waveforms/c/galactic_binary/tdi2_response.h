/* Copyright (C) 2026 Neil Cornish.
 * SPDX-License-Identifier: GPL-3.0-or-later
 */
#ifndef GB_TDI2_RESPONSE_H
#define GB_TDI2_RESPONSE_H

#include <complex.h>
#include <gsl/gsl_spline.h>

typedef int (*GBTDIPolarization)(void *userdata, double source_time,
                                 double complex *hplus, double complex *hcross);

/* One SSB-output sample of the SGS/PyTDI X2/Y2/Z2 Michelson response.
 * Positions are spacecraft-major Cartesian light-seconds. The callback
 * returns h-i h_quadrature in the same convention as the sparse AP code. */
int gb_tdi2_sample(double output_time, double sky_costheta, double sky_longitude,
                   gsl_interp_accel **position_acc, gsl_spline **position_spline,
                   GBTDIPolarization polarization, void *userdata,
                   double complex output[3]);

#endif
