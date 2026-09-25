# Source notices

The IMRPhenomT/THM/TPHM intrinsic C models and their Python ports descend
from the LALSimulation IMRPhenomTHM/TPHM work credited in their source headers
to Hector Estelles (2020). Neil Cornish is credited for the stripped C ports,
Python ports, and subsequent changes. `LALSimIMRPhenomTHM_fits.c` retains its
upstream and local modification notices. These files are
GPL-2.0-or-later.

The local LISA TDI and sparse-WDM response code, including the C and Python
Galactic-binary and FEW drivers, is GPL-3.0-or-later. `WDG.py` is joint work
by Neil Cornish and Noah Pearson; both are credited in that file and have
approved its GPL-3.0-or-later distribution. Preserve all file-level notices.

When the intrinsic and response code are distributed together as a combined
work, GPL-3.0-or-later terms apply. The unmodified GPLv2 and GPLv3 texts are
in `LICENSES/`. FastEMRIWaveforms, GSL, NumPy, SciPy, and Numba are external
dependencies and are not included in this source distribution; their own
licenses apply to those separately obtained packages.
