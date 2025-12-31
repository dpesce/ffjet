import numpy as np
import numpy.testing as npt

import jetfuncs as jf


def test_xyz_rtp_roundtrip_vector():
    # start with a point away from the axis so R != 0
    x, y, z = 1.2, -0.7, 2.3
    r = np.sqrt(x * x + y * y + z * z)
    R = np.sqrt(x * x + y * y)

    # arbitrary vector
    Vx, Vy, Vz = 0.2, -1.3, 0.9

    Vr, Vt, Vp = jf.xyz_to_rtp(Vx, Vy, Vz, x, y, z, r, R)
    Vx2, Vy2, Vz2 = jf.rtp_to_xyz(Vr, Vt, Vp, x, y, z, r, R)

    npt.assert_allclose([Vx2, Vy2, Vz2], [Vx, Vy, Vz], rtol=1e-12, atol=1e-12)
