import jetfuncs as jf


def test_imports():
    assert hasattr(jf, "xyz_to_rtp")


def test_xyz_to_rtp_smoke():
    Vr, Vt, Vp = jf.xyz_to_rtp(1.0, 2.0, 3.0, 1.0, 0.0, 0.0, 1.0, 1.0)
    assert Vr == 1.0


def test_tables_load_smoke():
    m = jf.JetModel(m=10.0, a=0.5, inc=60.0, mdot=1e-3, Nx=2, Ny=2, Nz=2, pretab_dir=None)
    assert m.GIx_2(1.0) >= 0.0
