import math
from copy import copy
from functools import lru_cache
from functools import partial

import igl
import numpy as np
import torch
import torch.nn.functional as F
from numpy.linalg import norm


#
@lru_cache(maxsize=12)
def get_icosahedron(level=0):
    if level == 0:
        v, f = get_base_icosahedron()
        return v, f
    # require subdivision
    v, f = get_icosahedron(level - 1)
    v, f = subdivision(v, f, 1)
    return v, f


@lru_cache(maxsize=12)
def get_unfold_icosahedron(level=0):
    if level == 0:
        unfold_v, f = get_base_unfold()
        return unfold_v, f
    # require subdivision
    unfold_v, f = get_unfold_icosahedron(level - 1)
    unfold_v, f = unfold_subdivision(unfold_v, f)
    return unfold_v, f


@lru_cache(maxsize=12)
def get_unfold_imgcoord(level=0):
    unfold_v, new_faces = get_unfold_icosahedron(level)
    distort_unfold = distort_grid(unfold_v)
    img_coord = distort_unfold_to_imgcoord(distort_unfold)
    return img_coord


@lru_cache(maxsize=12)
def get_weight_alpha(level=0):
    v, f = get_icosahedron(level)
    img_coord = get_unfold_imgcoord(level)
    weight = calculate_weight(img_coord, v)
    return weight


def get_base_icosahedron():
    t = (1.0 + 5.0 ** .5) / 2.0
    vertices = [-1, t, 0, 1, t, 0, 0, 1, t, -t, 0, 1, -t, 0, -1, 0, 1, -t, t, 0, -1, t, 0,
                1, 0, -1, t, -1, -t, 0, 0, -1, -t, 1, -t, 0]
    faces = [0, 2, 1, 0, 3, 2, 0, 4, 3, 0, 5, 4, 0, 1, 5,
             1, 7, 6, 1, 2, 7, 2, 8, 7, 2, 3, 8, 3, 9, 8, 3, 4, 9, 4, 10, 9, 4, 5, 10, 5, 6, 10, 5, 1, 6,
             6, 7, 11, 7, 8, 11, 8, 9, 11, 9, 10, 11, 10, 6, 11]

    # make every vertex have radius 1.0
    vertices = np.reshape(vertices, (-1, 3)) / (np.sin(2 * np.pi / 5) * 2)
    faces = np.reshape(faces, (-1, 3))

    # Rotate vertices so that v[0] = (0, -1, 0), v[1] is on yz-plane
    ry = -vertices[0]
    rx = np.cross(ry, vertices[1])
    rx /= np.linalg.norm(rx)
    rz = np.cross(rx, ry)
    R = np.stack([rx, ry, rz])
    vertices = vertices.dot(R.T)
    return vertices, faces


def subdivision(v, f, level=1):
    for _ in range(level):
        # subdivision
        v, f = igl.upsample(v, f)
        # normalize
        v /= np.linalg.norm(v, axis=1)[:, np.newaxis]
    return v, f


def get_base_unfold():
    v, f = get_base_icosahedron()
    unfold_v = {i: [] for i in range(12)}

    # edge length
    edge_len = 1 / np.sin(2 * np.pi / 5)
    # height
    h = 3 ** 0.5 * edge_len / 2

    # v0
    for i in range(5):
        unfold_v[0].append([i * edge_len, 0])

    # v1
    for _ in range(5):
        unfold_v[1].append([-0.5 * edge_len, h])
    unfold_v[1][1] = [-0.5 * edge_len + 5 * edge_len, h]
    unfold_v[1][4] = [-0.5 * edge_len + 5 * edge_len, h]

    # v2-v5
    for i in range(2, 6):
        for _ in range(5):
            unfold_v[i].append([(0.5 + i - 2) * edge_len, h])

    # v6
    for _ in range(5):
        unfold_v[6].append([-edge_len, 2 * h])
    unfold_v[6][1] = [-edge_len + 5 * edge_len, 2 * h]
    unfold_v[6][2] = [-edge_len + 5 * edge_len, 2 * h]
    unfold_v[6][4] = [-edge_len + 5 * edge_len, 2 * h]

    # v7-v10
    for i in range(7, 11):
        for _ in range(5):
            unfold_v[i].append([(i - 7) * edge_len, 2 * h])

    # v11
    for i in range(5):
        unfold_v[11].append([(-0.5 + i) * edge_len, 3 * h])

    # to numpy
    for i in range(len(unfold_v)):
        unfold_v[i] = np.array(unfold_v[i])
    return unfold_v, f


class UnfoldVertex(object):
    def __init__(self, unfold_v):
        self.unfold_v = unfold_v
        self.reset()

    def __getitem__(self, item):
        pos = self.unfold_v[item][self.cnt[item]]
        self.cnt[item] += 1
        return pos

    def reset(self):
        self.cnt = {key: 0 for key in self.unfold_v.keys()}


class VertexIdxManager(object):
    def __init__(self, unfold_v):
        self.reg_v = {}
        self.next_v_index = len(unfold_v)

    def get_next(self, a, b):
        if a > b:
            a, b = b, a
        key = f'{a},{b}'
        if key not in self.reg_v:
            self.reg_v[key] = self.next_v_index
            self.next_v_index += 1
        return self.reg_v[key]


def unfold_subdivision(unfold_v, faces):
    v_idx_manager = VertexIdxManager(unfold_v)

    new_faces = []
    new_unfold = copy(unfold_v)
    v_obj = UnfoldVertex(unfold_v)
    for (a, b, c) in faces:
        a_pos = v_obj[a]
        b_pos = v_obj[b]
        c_pos = v_obj[c]

        new_a = v_idx_manager.get_next(a, b)
        new_b = v_idx_manager.get_next(b, c)
        new_c = v_idx_manager.get_next(c, a)

        new_a_pos = (a_pos + b_pos) / 2
        new_b_pos = (b_pos + c_pos) / 2
        new_c_pos = (c_pos + a_pos) / 2

        # new faces
        new_faces.append([a, new_a, new_c])
        new_faces.append([b, new_b, new_a])
        new_faces.append([new_a, new_b, new_c])
        new_faces.append([new_b, c, new_c])

        # new vertex
        indices = [new_a, new_b, new_c]
        poses = [new_a_pos, new_b_pos, new_c_pos]
        for (idx, pos) in zip(indices, poses):
            if idx not in new_unfold:
                new_unfold[idx] = []
            for _ in range(3):
                new_unfold[idx].append(pos)
    return new_unfold, new_faces


def distort_grid(unfold_v):
    np_round = partial(np.round, decimals=9)

    # calculate transform matrix
    new_x = unfold_v[2][0] - unfold_v[0][0]
    edge_len = np.linalg.norm(new_x)
    new_x /= edge_len
    new_y = np.cross([0, 0, 1], np.append(new_x, 0))[:2]
    R = np.stack([new_x, new_y])

    a = unfold_v[2][0] - unfold_v[0][0]
    b = unfold_v[1][0] - unfold_v[0][0]
    skew = np.eye(2)
    skew[0, 1] = -1 / np.tan(np.arccos(a.dot(b) / norm(a) / norm(b)))
    skew[0] /= norm(skew[0])

    T = skew.dot(R)
    # scale adjust
    scale = np.linalg.det(skew) * edge_len
    T /= scale

    # to numpy array for efficient computation
    # np_round to alleviate numerical error when sorting
    distort_unfold = copy(unfold_v)
    five_neighbor = [distort_unfold[i] for i in range(12)]
    five_neighbor = np.array(five_neighbor)
    # Transform
    five_neighbor = np_round(five_neighbor.dot(T.T))

    # the same procedure for six_neighbor if len(unfold_v) > 12
    if len(unfold_v) > 12:
        six_neighbor = [distort_unfold[i] for i in range(12, len(unfold_v))]
        six_neighbor = np.array(six_neighbor)
        six_neighbor = np_round(six_neighbor.dot(T.T))

    # to original shape
    distort_unfold = {}
    cnt = 0
    for it in five_neighbor:
        distort_unfold[cnt] = it
        cnt += 1
    if len(unfold_v) > 12:
        for it in six_neighbor:
            distort_unfold[cnt] = it
            cnt += 1
    return distort_unfold


def get_rect_idxs(x, y):
    rect_idxs = []
    for i in range(5):
        x_min = i
        x_max = x_min + 1
        y_min = -i
        y_max = y_min + 2
        if x_min <= x <= x_max and y_min <= y <= y_max:
            rect_idxs.append(i)
    return rect_idxs


def distort_unfold_to_imgcoord(distort_unfold, drop_NE=True):
    """
    Parameters
    ----------
    distort_unfold :
        distorted unfold
    drop_NE : bool
        drop north and east as in [1]

    References
    ----------
    [1] orientation-aware semantic segmentation on icosahedron spheres, ICCV2019

    """
    vertex_num = len(distort_unfold)
    level = round(math.log((vertex_num - 2) // 10, 4))

    width = 2 ** level + 1
    height = 2 * width - 1

    unfold_pts_set = set()  # (vertex_id, x, y)

    # remove duplicate
    for key, arr in distort_unfold.items():
        for val in arr:
            unfold_pts_set.add((key, val[0], val[1]))

    # sort
    unfold_pts_set = sorted(unfold_pts_set, key=lambda x: (x[1], x[2]))

    # to image coorinate
    img_coord = {}
    for (vertex_id, x, y) in unfold_pts_set:
        rect_idxs = get_rect_idxs(x, y)
        for key in rect_idxs:
            if key not in img_coord:
                img_coord[key] = []
            img_coord[key].append(vertex_id)

    # to numpy
    for key in img_coord:
        img_coord[key] = np.array(img_coord[key]).reshape(width, height).T

    if drop_NE:
        # orientation-aware semantic segmentation on icosahedron spheres form
        for key in img_coord:
            img_coord[key] = img_coord[key][1:, :-1]

    return img_coord


def unfold_padding(arr, cval=0, only_NE=False):
    """
    Parameters
    ----------
    arr : dict {0-4: array}
        array
    cval : int or float
        initial padding value

    only_NE : bool
        drop north and east as in [1]

    References
    ----------
    [1] orientation-aware semantic segmentation on icosahedron spheres, ICCV2019
    """
    is_ndarray = False
    if isinstance(arr[0], np.ndarray):
        is_ndarray = True
        # to torch tensor
        arr = copy(arr)
        for i in range(5):
            arr[i] = torch.from_numpy(arr[i].copy())
            if arr[i].ndim == 3:
                # H x W x C -> C x H x W
                arr[i] = arr[i].permute(2, 0, 1)
            elif arr[i].ndim == 2:
                # H x W  -> 1 x H x W
                arr[i] = arr[i].unsqueeze(0)
            # Add batch dimension
            arr[i] = arr[i].unsqueeze(0)

    h, w = arr[0].size(2), arr[0].size(3)

    arr_with_pad = []
    pad_w = (0, 1, 1, 0) if only_NE else (1, 1, 1, 1)
    for key in range(5):
        arr_with_pad.append(F.pad(arr[key], pad_w, value=cval))

    for key in range(5):
        tgt = (key + 1) % 5
        # north
        arr_with_pad[key][..., 0, pad_w[0] + 1:] = arr[tgt][..., :w, 0]
        # east
        arr_with_pad[key][..., 1:w + 1, -1] = arr[tgt][..., w:, 0]
        arr_with_pad[key][..., w + 1:-1 - pad_w[3], -1] = arr[tgt][..., -1, 1:]

        if not only_NE:
            tgt = (key - 1) % 5
            # some Indices look like shifted but if you check the connectivility, it's fine
            # west
            arr_with_pad[key][..., 1:w + 1, 0] = arr[tgt][..., 0, :]
            arr_with_pad[key][..., w + 1:-1, 0] = arr[tgt][..., :w, -1]
            # south
            arr_with_pad[key][..., -1, :-1] = arr[tgt][..., w - 1:, -1]

    if is_ndarray:
        for i in range(5):
            arr_with_pad[i] = arr_with_pad[i].permute(0, 2, 3, 1).squeeze().numpy()

    return arr_with_pad


def calculate_weight(img_coord, vertices):
    """ calculate weight alpha

    References
    ----------
    [1] orientation-aware semantic segmentation on icosahedron spheres, ICCV2019
    """
    arr_with_pad = unfold_padding(img_coord, cval=0, only_NE=False)

    key = 0
    vi_idx = arr_with_pad[key][1:-1, 1:-1]
    vn1_idx = arr_with_pad[key][1:-1, 0:-2]
    vn6_idx = arr_with_pad[key][0:-2, 1:-1]

    vi = vertices[vi_idx]
    vn1 = vertices[vn1_idx]
    vn6 = vertices[vn6_idx]

    # vector from vi to north pole
    north_pole = np.array([0, -1, 0])
    to_north_pole = north_pole - vi

    # unit vector from vi to neighbor
    vn1vi = vn1 - vi
    vn1vi /= norm(vn1vi, axis=-1, keepdims=True)
    vn6vi = vn6 - vi
    vn6vi /= norm(vn6vi, axis=-1, keepdims=True)

    # face normal
    face_n = np.cross(vn1vi, vn6vi)
    face_n /= norm(face_n, axis=-1, keepdims=True)

    # to north pole on tangent plane
    proj_vec = np.sum(to_north_pole * face_n, axis=-1, keepdims=True) * face_n
    np_tangent_plane = to_north_pole - proj_vec
    np_tangent_plane /= norm(np_tangent_plane, axis=-1, keepdims=True)

    # calculate cost
    psi = np.arccos(np.sum(vn1vi * np_tangent_plane, axis=-1))
    tmp_vals = np.sum(vn6vi * np_tangent_plane, axis=-1)
    tmp_vals[0, 0] = np.clip(tmp_vals[0, 0], -1, 1)  # make sure value is between -1 and 1
    phi = np.arccos(tmp_vals)
    weight = phi / (psi + phi)

    return weight
