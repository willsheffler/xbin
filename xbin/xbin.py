import sys
# sys.path.insert(0, '/home/sheffler/src/homog/')
# print(('!' * 80 + '\n') * 10, end='')

import functools as ft
import numpy as np
from bcc import BCC
try:
    from bcc import numba_bcc_indexer
    HAVE_NUMBA = True
except ImportError:
    HAVE_NUMBA = False
import homog as hg
from homog.util import jit, guvec, float64, int64
import operator

if HAVE_NUMBA:

    def gu_xbin_indexer(*args, **kw):
        func = _kernel_xbin_indexer(*args, **kw)
        return guvec([(float64[:, :], int64[:])], '(n,n)->()', func)

    def numba_xbin_indexer(*args, **kw):
        func0 = _kernel_xbin_indexer(*args, **kw)

        @jit
        def func(value):
            res = np.empty(1, dtype=np.int64)
            func0(value, res)
            return res[0]

        return func

    def _kernel_xbin_indexer(cart_resl,
                             ori_resl,
                             cart_bound=512.0,
                             ori_nside=None):
        spec = xbin_make_spec(cart_resl, ori_resl, cart_bound, ori_nside)
        bcc6_indexer = numba_bcc_indexer(**spec)
        assert bcc6_indexer

        @jit
        def func(xform, index):
            face, f6 = numba_xform_to_f6(xform)
            idx = bcc6_indexer(f6)
            index[0] = np.bitwise_or(np.left_shift(face, 58), idx)

        return func


class XformBinner:
    def __init__(self,
                 cart_resl=1,
                 ori_resl=15,
                 cart_bound=512.0,
                 ori_nside=None):
        self.cart_resl = cart_resl
        self.ori_resl = ori_resl
        self.cart_bound = cart_bound
        spec = xbin_make_spec(cart_resl, ori_resl, cart_bound, ori_nside)
        self.bcc6 = BCC(**spec)

    def get_bin_index(self, xform, debug=False):
        face, f6 = xform_to_f6(xform)
        # print('get_bin_index face', face)
        idx = self.bcc6.get_bin_index(f6)
        index = np.bitwise_or(np.left_shift(face, 58), idx)
        if debug: index = index, f6
        return index

    def get_bin_center(self, index, debug=False):
        face = np.right_shift(index, 58)
        if not np.all((0 <= face) * (face < 24)):
            print('get_bin_center face', face[(0 > face) + (face > 24)])
            assert np.all((0 <= face) * (face < 24))
        bcc_key = np.right_shift(np.left_shift(index, 6), 6)
        assert np.all(np.right_shift(bcc_key, 58) == 0)
        f6 = self.bcc6.get_bin_center(bcc_key)
        center = f6_to_xform(face, f6)
        if debug: center = center, f6
        return center


def xbin_make_spec(cart_resl=1, ori_resl=15, cart_bound=512.0, ori_nside=None):
    if cart_resl <= 0:
        raise ValueError('cart_resl must be > 0, not ' + str(cart_resl))
    if ori_resl <= 0:
        raise ValueError('ori_resl must be > 0, not ' + str(ori_resl))
    if cart_bound <= 0:
        raise ValueError('cart_bound must be > 0, not ' + str(cart_bound))
    if not ori_nside:
        ori_nside = int(np.sum(_xform_binner_covrad >= ori_resl) + 1)
    # 1.7 is empirically determined hack...
    sizes = ([int(1.5 * cart_bound / cart_resl)] * 3 + [int(ori_nside)] * 3)
    totsize = ft.reduce(operator.mul, [_ + 2 for _ in sizes], 1)
    if totsize > 2**58:
        raise ValueError('too many bins, increase cart_resl and/or ori_resl,'
                         ' or reduce cart_bound')
    return dict(
        sizes=sizes,
        lower=[-cart_bound] * 3 + [0] * 3,
        upper=[cart_bound] * 3 + [1] * 3)


bt24cell_width = 2 * (np.sqrt(2) - 1)
bt24cell_width_diagonal = 0.696923425058676
_R22 = np.sqrt(2) / 2
half_bt24cell_faces = np.array([
    [1.000, 0.000, 0.000, 0.000],  # 0
    [0.000, 1.000, 0.000, 0.000],  # 1
    [0.000, 0.000, 1.000, 0.000],  # 2
    [0.000, 0.000, 0.000, 1.000],  # 3
    [+0.50, +0.50, +0.50, +0.50],  # 4
    [+0.50, +0.50, +0.50, -0.50],  # 5
    [+0.50, +0.50, -0.50, +0.50],  # 6
    [+0.50, +0.50, -0.50, -0.50],  # 7
    [+0.50, -0.50, +0.50, +0.50],  # 8
    [+0.50, -0.50, +0.50, -0.50],  # 9
    [+0.50, -0.50, -0.50, +0.50],  # 10
    [+0.50, -0.50, -0.50, -0.50],  # 11
    [+_R22, +_R22, 0.000, 0.000],  # 12
    [+_R22, -_R22, 0.000, 0.000],  # 13
    [+_R22, 0.000, +_R22, 0.000],  # 14
    [+_R22, 0.000, -_R22, 0.000],  # 15
    [+_R22, 0.000, 0.000, +_R22],  # 16
    [+_R22, 0.000, 0.000, -_R22],  # 17
    [0.000, +_R22, +_R22, 0.000],  # 18
    [0.000, +_R22, -_R22, 0.000],  # 19
    [0.000, +_R22, 0.000, +_R22],  # 20
    [0.000, +_R22, 0.000, -_R22],  # 21
    [0.000, 0.000, +_R22, +_R22],  # 22
    [0.000, 0.000, +_R22, -_R22],  # 23
])
half_bt24cell_faces.flags.writeable = False


def half_bt24cell_face(quat):
    # maybe faster to do the c++ stuff:
    # template <class V4, class Index>
    # void get_cell_bt24cell_half(V4 const &quat, Index &cell) {
    #   typedef typename V4::Scalar Float;
    #   V4 const quat_pos = quat.cwiseAbs();
    #   V4 tmpv = quat_pos;

    #   Float hyperface_dist;  // dist to closest face
    #   Index hyperface_axis;  // closest hyperface-pair
    #   Float edge_dist;       // dist to closest edge
    #   Index edge_axis_1;     // first axis of closest edge
    #   Index edge_axis_2;     // second axis of closest edge
    #   Float corner_dist;     // dist to closest corner

    #   // std::cout << quat_pos.transpose() << std::endl;
    #   numeric::max2(quat_pos, hyperface_dist, edge_dist, hyperface_axis,
    #                 edge_axis_2);
    #   edge_dist = sqrt(2) / 2 * (hyperface_dist + edge_dist);
    #   corner_dist = quat_pos.sum() / 2;
    #   // std::cout << hyperface_axis << " " << edge_axis_2 << std::endl;
    #   edge_axis_1 = hyperface_axis < edge_axis_2 ? hyperface_axis : edge_axis_2;
    #   edge_axis_2 = hyperface_axis < edge_axis_2 ? edge_axis_2 : hyperface_axis;
    #   assert(edge_axis_1 < edge_axis_2);

    #   // cell if closest if of form 1000 (add 4 if negative)
    # Index facecell = hyperface_axis;  // | (quat[hyperface_axis]<0 ? 4 :
    # 0);

    #   // cell if closest is of form 1111, bitwise by ( < 0)
    #   Index bit0 = quat[0] < 0;
    #   Index bit1 = quat[1] < 0;
    #   Index bit2 = quat[2] < 0;
    #   Index cornercell = quat[3] > 0 ? bit0 | bit1 << 1 | bit2 << 2
    #                                  : (!bit0) | (!bit1) << 1 | (!bit2) << 2;

    #   // cell if closest is of form 1100
    #   Index perm_shift[3][4] = {{9, 0, 1, 2}, {0, 9, 3, 4}, {1, 3, 9, 5}};
    #   Index sign_shift = (quat[edge_axis_1] < 0 != quat[edge_axis_2] < 0) * 1 * 6;
    #   Index edgecell = sign_shift + perm_shift[edge_axis_1][edge_axis_2];

    #   // pick case 1000 1111 1100 without if statements
    #   Index swtch;
    #   util::SimpleArray<3, Float>(hyperface_dist, corner_dist, edge_dist)
    #       .maxCoeff(&swtch);
    #   cell = swtch == 0 ? facecell : (swtch == 1 ? cornercell + 4 : edgecell + 12);
    #   // this is slower !?!
    #   // Float mx = std::max(std::max(hyperface_dist,corner_dist),edge_dist);
    #   // cell2[i] = hyperface_dist==mx ? facecell : (corner_dist==mx ? cornercell+8
    #   // : edgecell+24);
    # }
    quat = np.asarray(quat)
    fullaxes = (slice(None), ) + (np.newaxis, ) * (quat.ndim - 1)
    hf = half_bt24cell_faces[fullaxes]
    dots = abs(np.sum(quat[np.newaxis] * hf, axis=-1))
    return np.argmax(dots, axis=0)


@hg.util.jit
def numba_half_bt24cell_face(quat):
    dots = np.abs(half_bt24cell_faces @ quat)
    return np.argmax(dots)


def to_face_0(quat):
    face = half_bt24cell_face(quat)
    to_0 = half_bt24cell_faces[face].copy()
    to_0[..., 0] *= -1.0  # inverse
    q0 = hg.quat.quat_multiply(to_0, quat)
    return face, hg.quat.quat_to_upper_half(q0)


@hg.util.jit
def numba_to_face_0(quat):
    face = numba_half_bt24cell_face(quat)
    to_0 = half_bt24cell_faces[face].copy()
    to_0[0] *= -1.0  # inverse
    q0 = hg.quat.numba_quat_multiply(to_0, quat)
    q0 = hg.quat.numba_quat_to_upper_half(q0)
    return face, q0


_xform_binner_covrad = np.array([
    68.907736, 38.232608, 25.860598, 19.391913, 15.461347, 13.152186,
    11.309651, 9.849257, 8.653823, 7.778357, 7.044655, 6.504872, 5.971447,
    5.631019, 5.240999, 4.830578, 4.609091, 4.373091, 4.113840, 3.855201,
    3.726551, 3.529638, 3.377918, 3.262408, 3.171638, 2.978760, 2.879508,
    2.851197, 2.672736, 2.628641, 2.515808, 2.448288
])
_xform_binner_covrad.flags.writeable = False


def _f6_to_quat(f6):
    quat = np.empty(f6.shape[:-1] + (4, ))
    quat[..., 0] = 1.0
    quat[..., 1:] = bt24cell_width * (np.clip(f6[..., 3:6], 0, 1) - 0.5)
    return quat / np.linalg.norm(quat, axis=-1)[..., np.newaxis]


def is_contained_in_bt24cell_face0(f6):
    return half_bt24cell_face(_f6_to_quat(f6)) == 0


def f6_to_xform(face, f6):
    quat = _f6_to_quat(f6)
    quat = hg.quat.quat_multiply(half_bt24cell_faces[face], quat)
    xform = hg.quat.quat_to_xform(quat)
    xform[..., :3, 3] = f6[..., :3]
    return xform


def xform_to_f6(xform):
    quat = hg.quat.rot_to_quat(xform)
    face, q0 = to_face_0(quat)
    f6 = np.empty(q0.shape[:-1] + (6, ))
    f6[..., :3] = xform[..., :3, 3]
    # TODO (w,x,y,z) why divide by xyz by w? in cell0, w close to 1.0?
    ori_params = 0.5 + q0[..., 1:4] / q0[..., 0, None] / bt24cell_width
    f6[..., 3:6] = np.clip(ori_params, 0, 1)
    return face, f6


@hg.util.jit
def numba_xform_to_f6(xform):
    quat = hg.quat.numba_rot_to_quat(xform)
    face, q0 = numba_to_face_0(quat)
    f6 = np.empty((6, ))
    f6[:3] = xform[:3, 3]
    # TODO (w,x,y,z) why divide by xyz by w? in cell0, w close to 1.0?
    ori_params = 0.5 + q0[1:4] / q0[0] / bt24cell_width
    f6[3:6] = np.minimum(1, np.maximum(0, ori_params))
    return face, f6
