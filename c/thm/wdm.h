// Copyright (C) Neil Cornish.
// SPDX-License-Identifier: GPL-3.0-or-later
//#define Nf 1024  // frequency layers
//#define Nt 4096 // time steps

#ifndef Nf
#define Nf 4096  // frequency layers
#endif
// 4096 gives Tobs = Nt*Nf*dt = 3.145728e7 s for dt = 1.875 s.
#ifndef Nt
#define Nt 4096 // time steps
#endif
#ifndef dt
#define dt 1.875 // time cadence
#endif

#define Nsf 400  // frequency steps
#define Nfd 4  // number of f-dots
#define dfdot 0.1 // fractional fdot increment

#define Bfrac 1.0  // fall-off region for frequency filter
#define nx 6.0    // filter steepness in frequency
#define mult 8  // over sampling
