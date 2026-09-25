// Copyright (C) Neil Cornish.
// SPDX-License-Identifier: GPL-3.0-or-later
//#define Nf 4096  // frequency layers
//#define Nt 4096 // time steps

#define Nf 4096  // frequency layers
// 4096 gives Tobs = Nt*Nf*dt = 3.145728e7 s for dt = 1.875 s.
#define Nt 4096 // time steps
#define dt 1.875 // time cadence
//#define dt 8.0

#define Nsf 400  // frequency steps
#define Nfd 4  // number of f-dots
#define dfdot 0.1 // fractional fdot increment

#define Bfrac 1.0  // fall-off region for frequency filter
#define nx 6.0    // filter steepness in frequency

#define mult 8  // over sampling

 /* Orbital radius of the guiding center */
#define Rgc (1.0*AU)


 /* Photon shot noise power */
#define Sps 2.25e-22
 
 /* Acceleration noise power */
#define Sacc 9.0e-30

 /* Initial azimuthal position of the guiding center */
#define kappa0 0.0

 /* Initial orientation of the LISA constellation */
#define lambda0 0.0

 /* Transfer frequency */
#define fstr 0.01908538064

 /* LISA orbital eccentricity */
#define ec 0.0048241852175

 /* LISA modulation frequency */
#define fm 3.168753575e-8
