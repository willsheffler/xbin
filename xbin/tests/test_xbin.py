from xbin import *
import homog
import pytest
from traitlets import TraitError
from hypothesis import given, settings
import hypothesis.strategies as hs
import hypothesis.extra.numpy as hn

try:
    import numba
    only_if_numba = lambda f: f
except ImportError:
    import pytest
    only_if_numba = pytest.mark.skip

# print(('!' * 80 + '\n') * 10, end='')
MAX_EXAMPLES = 100


def test_get_half_bt24cell_face_basic():
    assert half_bt24cell_faces.shape == (24, 4)
    assert np.all(
        homog.quat.quat_to_upper_half(half_bt24cell_faces) ==
        half_bt24cell_faces)
    assert np.min(np.linalg.norm(half_bt24cell_faces, axis=-1)) == 1
    assert np.max(np.linalg.norm(half_bt24cell_faces, axis=-1)) == 1
    assert half_bt24cell_face([0, 0, 0, 1]) == 3
    assert half_bt24cell_face([1, -1, 1, 1]) == 8
    assert half_bt24cell_face([0.5, 0.5, 0, 0]) == 12
    assert half_bt24cell_face([-0.5, 0, 0, 0.5]) == 17
    assert np.all(half_bt24cell_face(half_bt24cell_faces) == np.arange(24))
    assert np.all(half_bt24cell_face(-half_bt24cell_faces) == np.arange(24))
    q = homog.quat.rand_quat((7, 6, 5))
    x = half_bt24cell_face(q)
    assert x.shape == (7, 6, 5)
    assert np.all(x >= 0)
    assert np.all(x <= 23)


@only_if_numba
def test_numba_half_bt24cell_face():
    assert numba_half_bt24cell_face(np.array([0, 0, 0, 1], dtype='f8')) == 3
    assert numba_half_bt24cell_face(np.array([1, -1, 1, 1], dtype='f8')) == 8
    assert numba_half_bt24cell_face(np.array([0.5, 0.5, 0, 0],
                                             dtype='f8')) == 12
    assert numba_half_bt24cell_face(np.array([-0.5, 0, 0, 0.5],
                                             dtype='f8')) == 17


def test_half_bt24cell_geom():
    cellquatang = 2 * np.arccos(
        np.clip(
            np.sum(half_bt24cell_faces * half_bt24cell_faces[:, None], -1), -1,
            1))
    cellquatang = np.minimum(cellquatang, 2 * np.pi - cellquatang)
    cellcen = homog.quat.quat_to_xform(half_bt24cell_faces)
    cellcenang = homog.angle_of(homog.hinv(cellcen) @ cellcen[:, None])

    cellcenanground = np.round(cellcenang * 180 / np.pi, 5)
    # print(np.unique(cellcenanground))

    assert np.allclose(cellcenang, cellquatang)
    # np.fill_diagonal(cellcenang, 10)

    q = homog.quat.rand_quat((1000, ))
    qang = 2 * np.arccos(
        np.clip(np.abs(np.sum(q * half_bt24cell_faces[:, None], -1)), -1, 1))
    minqang = np.min(qang, 0)
    # print('Qang to closest',
    #        np.percentile(minqang * 180 / np.pi, np.arange(0, 101, 10)))
    assert np.max(minqang) < 62.81 / 180 * np.pi

    x = homog.rand_xform(1000)
    cellxform = homog.quat.quat_to_xform(half_bt24cell_faces)
    xang = homog.angle_of(x @ homog.hinv(cellxform)[:, None])
    minxang = np.min(xang, 0)
    # print('Xang to closest:',
    # np.percentile(minxang * 180 / np.pi, np.arange(0, 101, 10)))
    assert np.max(minqang) < 62.81 / 180 * np.pi


def test_XformBinner_init():
    with pytest.raises(ValueError):
        XformBinner(cart_resl=-1)
    with pytest.raises(ValueError):
        XformBinner(ori_resl=-1)
    with pytest.raises(ValueError):
        XformBinner(cart_bound=-1)
    assert XformBinner().cart_resl == 1
    assert XformBinner(cart_resl=2).cart_resl == 2


def test_xform_to_f6():
    x = homog.rand_xform(24000)
    face, f6 = xform_to_f6(x)
    assert np.all(face >= 0)
    assert np.all(face < 24)
    assert np.all(f6[..., 3:] >= 0)
    assert np.all(f6[..., 3:] <= 1)
    counts = np.sum(face == np.arange(24)[..., None], axis=-1)
    assert np.all(counts < 1200)
    assert np.all(counts > 800)

    xbt24 = homog.quat.quat_to_xform(half_bt24cell_faces)
    assert np.all(homog.is_homog_xform(xbt24))
    qbt24 = homog.quat.xform_to_quat(xbt24)
    assert np.all(homog.quat.is_valid_quat_rot(qbt24))
    assert np.allclose(qbt24, half_bt24cell_faces)
    face = half_bt24cell_face(qbt24)
    assert np.all(face == np.arange(24))


@only_if_numba
def test_numba_to_face_0():
    for i in range(1000):
        q = homog.quat.rand_quat()


@only_if_numba
def test_numba_xform_to_f6():
    for i in range(1000):
        x = homog.rand_xform(1)
        face_0, f6_0 = xform_to_f6(x)
        face_1, f6_1 = numba_xform_to_f6(x[0])
        assert face_0 == face_1
        if not np.allclose(f6_0, f6_1):
            print(i, face_0, face_1)
            print(f6_0)
            print(f6_1)
            print(hg.quat.rot_to_quat(x))
            print(hg.quat.numba_rot_to_quat(x[0]))
            assert 0


@hs.composite
def Quaternions(draw, shape=()):
    N = int(np.prod(shape))
    quat = draw(hn.arrays(np.float, (0, 4), elements=hs.floats(-1, 1)))
    while quat.shape[0] < N:
        qbox = draw(
            hn.arrays(
                dtype=np.float,
                shape=(2 * (N - quat.shape[0]), 4),
                elements=hs.floats(-1, 1)))
        norm = np.linalg.norm(qbox, axis=-1)
        qbox[norm == 0] = [1, 0, 0, 0]
        norm[norm == 0] = 1
        ok = (0.01 < norm) * (norm <= 1.0)
        qboxok = qbox[ok] / norm[ok, np.newaxis]
        quat = np.concatenate([quat, qboxok])
    return quat[:N].reshape(shape + (4, ))


@hs.composite
def Xforms(draw, shape=()):
    quat = draw(Quaternions(shape))
    x = homog.quat.quat_to_xform(quat)
    trns = draw(
        hn.arrays(np.float, shape + (3, ), elements=hs.floats(-10, 10)))
    x[..., :3, 3] = trns
    return x


@pytest.mark.skip
@given(hn.arrays(np.float, (1000, 6), elements=hs.floats(0, 1)))
@settings(max_examples=MAX_EXAMPLES)
def test_hypothesis_f6_to_xform_invertibility_face0(f6):
    assert np.all(f6[..., 3:] >= 0)
    assert np.all(f6[..., 3:] <= 1)
    f6 = f6[is_contained_in_bt24cell_face0(f6)]
    x = f6_to_xform(0, f6)
    assert np.allclose(x[..., :3, 3], f6[..., :3])
    face, f6b = xform_to_f6(x)
    assert np.sum(face != 0) < 10
    assert np.allclose(f6[face == 0], f6b[face == 0])


@given(Xforms(shape=(100, )))
@settings(max_examples=MAX_EXAMPLES)
def test_hypothesis_f6_to_xform_invertibility(xforms):
    face, f6 = xform_to_f6(xforms)
    assert np.all((0 <= face) * (face < 24))
    assert np.all(f6[..., 3:] >= 0)
    assert np.all(f6[..., 3:] <= 1)
    xhat = f6_to_xform(face, f6)
    assert np.allclose(xhat, xforms)


def test_f6_to_quat():

    assert np.allclose(
        xbin._f6_to_quat(np.array([0, 0, 0, 0.0, 0.0, 0.0])),
        [0.81251992, -0.33655677, -0.33655677, -0.33655677])
    assert np.allclose(
        xbin._f6_to_quat(np.array([0, 0, 0, 0.5, 0.5, 0.5])), [1, 0, 0, 0])
    assert np.allclose(
        xbin._f6_to_quat(np.array([0, 0, 0, 1.0, 1.0, 1.0])),
        [0.81251992, 0.33655677, 0.33655677, 0.33655677])


def test_f6_to_xform_invertibility_face0():
    nexamples = 100000
    f6 = np.random.rand(nexamples, 6)
    f6 = f6[is_contained_in_bt24cell_face0(f6)]
    assert f6.shape[0] > nexamples / 10
    assert np.all(f6[..., 3:] >= 0)
    assert np.all(f6[..., 3:] <= 1)
    x = f6_to_xform(0, f6)
    assert np.allclose(x[..., :3, 3], f6[..., :3])
    face, f6b = xform_to_f6(x)
    assert np.all(np.min(f6b[..., 3:], axis=0) < 0.1)
    assert np.all(np.max(f6b[..., 3:], axis=0) > 0.9)
    assert np.all(face == 0)
    assert np.allclose(f6, f6b)


def test_f6_to_xform_invertibility():
    nexamples = 100000
    xforms = homog.rand_xform((nexamples, ))
    face, f6 = xform_to_f6(xforms)
    assert np.all(np.min(f6[..., 3:], axis=0) < 0.1)
    assert np.all(np.max(f6[..., 3:], axis=0) > 0.9)
    assert np.all((0 <= face) * (face < 24))
    assert np.all(f6[..., 3:] >= 0)
    assert np.all(f6[..., 3:] <= 1)
    xhat = f6_to_xform(face, f6)
    assert np.allclose(xhat, xforms)


def test_too_many_bins():
    XformBinner(0.1, 3)
    with pytest.raises(ValueError):
        XformBinner(0.01, 3)
    with pytest.raises(ValueError):
        XformBinner(0.01, 10)


def test_XformBinner_covrad():
    niter = 30
    nsamp = 10000
    for i in range(niter):
        cart_resl = np.random.rand() * 10 + 0.1
        ori_resl = np.random.rand() * 50 + 2.5
        xforms = homog.rand_xform(nsamp)
        xb = XformBinner(cart_resl, ori_resl)
        idx = xb.get_bin_index(xforms)
        cen, f6 = xb.get_bin_center(idx, debug=True)
        cart_dist = np.linalg.norm(
            xforms[..., :3, 3] - cen[..., :3, 3], axis=-1)
        ori_dist = homog.angle_of(homog.hinv(cen) @ xforms)
        # if not np.all(cart_dist < cart_resl):
        # print('ori_resl', ori_resl, 'nside:', xb.ori_nside,
        # 'max(cart_dist):', np.max(cart_dist), cart_resl)
        # if not np.all(cart_dist < cart_resl):
        # print('ori_resl', ori_resl, 'nside:', xb.ori_nside,
        # 'max(ori_dist):', np.max(ori_dist))
        assert np.sum(cart_dist < cart_resl) > nsamp * 0.99
        assert np.sum(ori_dist < ori_resl / 180 * np.pi) > nsamp * 0.99


def test_xbinner_bcc6_alignment():
    cart_resl = 512
    ori_resl = 63
    xb = XformBinner(cart_resl, ori_resl)
    # print(xb.bcc6.sizes)
    bcc6_cen = xb.bcc6.get_bin_center(np.arange(len(xb.bcc6)))
    # print(bcc6_cen.shape)
    inbounds = ((xb.bcc6.lower < bcc6_cen) * (bcc6_cen < xb.bcc6.upper))
    inbounds = np.prod(inbounds, axis=-1) == 1
    # print(np.sum(inbounds))
    # print(bcc6_cen[inbounds])
    # print('nside', xb.bcc6.nside)
    assert np.all(np.abs(inbounds[0]) == 0)


@only_if_numba
def test_numba_xbin():
    cart_resl = 1.0
    ori_resl = 10.0
    xbinner = XformBinner(cart_resl, ori_resl)
    xbin_indexer = numba_xbin_indexer(cart_resl, ori_resl)

    for i in range(100):
        x = homog.rand_xform(1)
        assert xbinner.get_bin_index(x) == xbin_indexer(x[0])

    xbin_indexer = gu_xbin_indexer(cart_resl, ori_resl)

    x = homog.rand_xform(10000)
    idx0 = xbinner.get_bin_index(x)
    idx1 = xbin_indexer(x)
    assert np.all(idx0 == idx1)