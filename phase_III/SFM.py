from operator import itemgetter
import numpy as np


def calc_TFL_dist(prev_container, curr_container, focal: float, pp: np.ndarray):
    norm_prev_pts, norm_curr_pts, R, foe, tZ = prepare_3D_data(prev_container, curr_container, focal, pp)

    if abs(tZ) < 10e-6:
        print('tz = ', tZ)

    elif norm_prev_pts.size == 0:
        print('no prev points')

    elif norm_prev_pts.size == 0:
        print('no curr points')

    else:
        curr_container.corresponding_ind, curr_container.traffic_lights_3d_location, curr_container.valid = \
            calc_3D_data(norm_prev_pts, norm_curr_pts, R, foe, tZ)

    return curr_container


def prepare_3D_data(prev_container, curr_container, focal: float, pp: np.ndarray):
    norm_prev_pts = normalize(prev_container.traffic_light, focal, pp)
    norm_curr_pts = normalize(curr_container.traffic_light, focal, pp)
    R, foe, tZ = decompose(np.array(curr_container.EM))

    return norm_prev_pts, norm_curr_pts, R, foe, tZ


def calc_3D_data(norm_prev_pts: np.ndarray, norm_curr_pts: np.ndarray, R: np.ndarray, foe: np.ndarray, tZ: float) -> \
        (list, np.array, list):

    norm_rot_pts = rotate(norm_prev_pts, R)
    pts_3D = []
    corresponding_ind = []
    valid_vec = []

    for p_curr in norm_curr_pts:
        corresponding_p_ind, corresponding_p_rot = find_corresponding_points(p_curr, norm_rot_pts, foe)
        Z = calc_dist(p_curr, corresponding_p_rot, foe, tZ)
        valid = (Z > 0)

        if not valid:
            Z = 0
        valid_vec.append(valid)
        P = Z * np.array([p_curr[0], p_curr[1], 1])
        pts_3D.append((P[0], P[1], P[2]))
        corresponding_ind.append(corresponding_p_ind)

    return corresponding_ind, np.array(pts_3D), valid_vec


def normalize(pts: np.ndarray, focal: float, pp: np.ndarray) -> np.ndarray:
    return (pts - pp) / focal


def unnormalize(pts: np.ndarray, focal: float, pp: np.ndarray) -> np.ndarray:
    return pts * focal + pp


def decompose(EM: np.ndarray) -> (np.ndarray, float):
    R = EM[:3, :3]
    t = EM[:3, 3]
    tx, ty, tz = t[0], t[1], t[2]
    foe = np.array([tx/tz, ty/tz])

    return R, foe, tz


def rotate(pts: np.ndarray, R: np.ndarray) -> [np.ndarray]:
    ones = np.ones((len(pts[:, 0]), 1), int)
    rotate_mat = np.dot(R, (np.hstack([pts, ones])).T)

    return (rotate_mat[:2]/rotate_mat[2]).T


def find_corresponding_points(p: np.array, norm_pts_rot: list, foe: np.array) -> (int, np.ndarray):
    x, y = 0, 1
    m = (foe[y]-p[y])/(foe[x]-p[x])
    n = (p[y]*foe[x] - foe[y]*p[x])/(foe[x]-p[x])

    list_ = [[abs((m * pts[x] + n - pts[y]) / np.sqrt(pow(m, 2) + 1)), i] for i, pts in enumerate(norm_pts_rot)]
    min_dist, i_min = min(list_, key=itemgetter(0))

    return i_min, norm_pts_rot[i_min]


def calc_dist(p_curr: np.ndarray, p_rot: np.ndarray, foe: np.array, tZ: float) -> float:
    dis_x = tZ * (foe[0] - p_rot[0]) / (p_curr[0] - p_rot[0])
    dis_y = tZ * (foe[1] - p_rot[1]) / (p_curr[1] - p_rot[1])
    m = abs(p_curr[1]-p_rot[1]) / abs((p_curr[0] - p_rot[0]))

    if m < 1:
        return (1-m)*dis_x + m*dis_y
    return dis_y 

