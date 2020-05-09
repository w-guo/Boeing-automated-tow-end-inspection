import math
import numpy as np
from sklearn.neighbors import KernelDensity
from scipy.signal import argrelextrema
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

# --------------- Filter line segments with major orientation ---------------


def get_orientation(line):
    '''get orientation of a line
    '''
    if line[0] > line[2]:
        line[0], line[2] = line[2], line[0]
        line[1], line[3] = line[3], line[1]
    orientation = math.atan2((line[3] - line[1]), (line[2] - line[0]))

    return math.degrees(orientation)


def lineMagnitude(line):
    'Get line (aka vector) length'
    lineMagnitude = math.hypot(line[2] - line[0], line[3] - line[1])
    return lineMagnitude


def perp_distance(a_line, b_line):
    'Perpendicular distance between two parallel line segments'
    px, py = a_line[:2]
    x1, y1, x2, y2 = b_line

    LineMag = lineMagnitude(b_line)

    u1 = (((px - x1) * (x2 - x1)) + ((py - y1) * (y2 - y1)))
    u = u1 / (LineMag * LineMag)

    ix = x1 + u * (x2 - x1)
    iy = y1 + u * (y2 - y1)
    perp_dist = lineMagnitude([px, py, ix, iy])

    return perp_dist


def is_aligned(a_line, b_line, dist, tol_angle):
    'check if two parallel lines almost align'

    if dist < 10:  # if the two lines are close enough
        return True
    elif perp_distance(a_line, b_line) / dist < math.sin(
            math.pi / 180 * tol_angle):
        return True
    else:
        return False


def find_major_orientations(lines):
    orit = []
    l_mag = []
    for l in lines:
        line_mag = lineMagnitude(l)
        orientation = get_orientation(l)
        l_mag.append(line_mag)
        orit.append(orientation)
    # 1D clustering
    kde = KernelDensity(bandwidth=2,
                        kernel='gaussian').fit(np.array(orit).reshape(-1, 1))
    # make the range a little wider than [-90,90]
    s = np.linspace(-91, 91, 183)
    e = kde.score_samples(s.reshape(-1, 1))
    #plt.plot(s, e)
    mi, ma = argrelextrema(e, np.less)[0], argrelextrema(e, np.greater)[0]

    # special case
    # if nearly vertical lines are caterigorized into two clusters
    if s[ma][0] < -85 and s[ma][-1] > 85:
        # we consider the two clusters as one cluster. Thus, the number of groups is len(mi)
        groups = [[] for x in range(len(mi))]
        line_idx = [[] for x in range(len(mi))]
        for i in range(len(orit)):
            if orit[i] > s[mi][-1]:
                groups[0].append(l_mag[i])
                line_idx[0].append(i)
            else:
                for j in range(len(mi)):
                    if orit[i] < s[mi][j]:
                        groups[j].append(l_mag[i])
                        line_idx[j].append(i)
                        break
    else:
        # common case
        groups = [[] for x in range(len(mi) + 1)]
        line_idx = [[] for x in range(len(mi) + 1)]
        if len(mi) > 0:
            for i in range(len(orit)):
                if orit[i] > s[mi][-1]:
                    groups[-1].append(l_mag[i])
                    line_idx[-1].append(i)
                else:
                    for j in range(len(mi)):
                        if orit[i] < s[mi][j]:
                            groups[j].append(l_mag[i])
                            line_idx[j].append(i)
                            break
        else:
            for i in range(len(orit)):
                groups[0].append(l_mag[i])
                line_idx[0].append(i)

    # determine the major orientations based on the total length of line segments from nearly the same oritation
    line_mag_sum = np.zeros(len(groups))
    for i in range(len(groups)):
        line_mag_sum[i] = np.sum(np.array(groups[i]))

    # the total length of line segments should not be too small (1/6 is just an estimate)
    group_idx = np.where(
        line_mag_sum > np.sum(line_mag_sum) * 1 / 6)[0].tolist()
    group_idx_all = []
    # find the oritation of tow ends and tow edges along the boungdary of the top layer
    # (or subtop layer because the total length of tow ends or tow edges that
    # consists of the boungdary of the top layer is not necessarily the largest)
    top_group_idx = []
    top_1_group_idx = np.argsort(line_mag_sum)[-1].tolist()
    top_group_idx.append(top_1_group_idx)

    if len(group_idx) > 1:
        for i in group_idx:
            if abs(abs(s[ma][i] - s[ma][top_1_group_idx]) - 90) < 5:
                top_group_idx.append(i)
                break
    group_idx_all = top_group_idx
    # just in case that a subtop layer exists, find the oritation of tow ends and
    # tow edges along the boungdary of the subtop layer
    sub_group_idx = list(set(group_idx) - set(top_group_idx))
    if len(sub_group_idx) > 0:
        sub_top_group_idx = []
        sub_top_1_group_idx = sub_group_idx[np.argsort(
            line_mag_sum[sub_group_idx])[-1]]
        sub_top_group_idx.append(sub_top_1_group_idx)
        if len(sub_group_idx) > 1:
            for i in sub_group_idx:
                if abs(abs(s[ma][i] - s[ma][sub_top_1_group_idx]) - 90) < 5:
                    sub_top_group_idx.append(i)
                    break
        group_idx_all += sub_top_group_idx

    idx_all = []
    for i in group_idx_all:
        idx_all += line_idx[i]

    filter_lines = np.array(lines)[idx_all]

    return filter_lines.tolist()


# --------------------------- Merge line segments ---------------------------

# modified from https://stackoverflow.com/questions/45531074/how-to-merge-lines-after-houghlinesp


def group_lines(lines):
    'Clusterize (group) lines'
    groups = []  # all lines groups are here
    # Parameters to play with
    tol_distance_to_merge = 25
    tol_angle_to_merge = 22.5
    # first line will create new group every time
    groups.append([lines[0]])
    # if line is different from existing gropus, create a new group
    for line_new in lines[1:]:
        if check_similaririty(line_new, groups, tol_distance_to_merge,
                              tol_angle_to_merge):
            groups.append([line_new])

    return groups


def check_similaririty(line_new, groups, tol_distance_to_merge,
                       tol_angle_to_merge):
    '''Check if line have enough distance and angle to be count as similar
    '''
    for group in groups:
        # walk through existing line groups
        for line_old in group:
            orit_new = get_orientation(line_new)
            orit_old = get_orientation(line_old)
            l_dist = get_distance(line_new, line_old)
            # special case:
            # if two line segements are nearly parallel
            # for merging, since we often deal with short lines, we allow comparatively
            # larger angles to check parallel and alignment conditions
            if abs(orit_new - orit_old) < 10:
                # if two line segements almost align, we allow a larger distance (40 pix) to merge
                if is_aligned(line_new, line_old, l_dist, 20) and l_dist < 40:
                    group.append(line_new)
                    return False
                elif l_dist > 15 and perp_distance(
                        line_new, line_old) / l_dist > math.sin(
                            math.pi / 180 * 60):
                    continue
        # common case
        # check the angle between lines
            if abs(orit_new - orit_old) < tol_angle_to_merge:
                # if all is ok -- line is similar to others in group
                if l_dist < tol_distance_to_merge:
                    group.append(line_new)
                    return False
    # if it is totally different line
    return True


def get_distance(a_line, b_line):
    """Get all possible distances between each dot of two lines and second line
    return the shortest
    """
    dist1 = DistancePointLine(a_line[:2], b_line)
    dist2 = DistancePointLine(a_line[2:], b_line)
    dist3 = DistancePointLine(b_line[:2], a_line)
    dist4 = DistancePointLine(b_line[2:], a_line)

    return min(dist1, dist2, dist3, dist4)


def DistancePointLine(point, line):
    """Get distance between point and line
    http://local.wasp.uwa.edu.au/~pbourke/geometry/pointline/source.vba
    """
    px, py = point
    x1, y1, x2, y2 = line

    LineMag = lineMagnitude(line)

    u1 = (((px - x1) * (x2 - x1)) + ((py - y1) * (y2 - y1)))
    u = u1 / (LineMag * LineMag)

    if (u < 1e-5) or (u > 1):
        # closest point does not fall within the line segment, take the shorter distance
        # to an endpoint
        ix = lineMagnitude([px, py, x1, y1])
        iy = lineMagnitude([px, py, x2, y2])
        DistancePointLine = np.amin([ix, iy])
    else:
        # Intersecting point is on the line, use the formula
        ix = x1 + u * (x2 - x1)
        iy = y1 + u * (y2 - y1)
        DistancePointLine = lineMagnitude([px, py, ix, iy])

    return DistancePointLine


def sort_lines(lines):
    """Sort lines cluster and return first and last coordinates
    """
    orientation = get_orientation(lines[0])

    # special case
    if (len(lines) == 1):
        return [lines[0][:2], lines[0][2:]]

    # [[1,2,3,4],[]] to [[1,2],[3,4],[],[]]
    points = []
    for line in lines:
        points.append(line[:2])
        points.append(line[2:])
    if orientation < 22.5:
        points = sorted(points, key=lambda point: point[0])
    else:
        points = sorted(points, key=lambda point: point[1])

    # return first and last point in sorted group
    # [[x,y],[x,y]]
    return [points[0], points[-1]]


def merge_lines(lines):

    lines_g1 = []
    lines_g2 = []

    for line in lines:
        orientation = get_orientation(line)
        if orientation > 85:
            orientation -= 180  # let the nearly vertical lines be in the same group
        # Since the range of line oritations is [-90,90], and the oritation of
        # tow ends and tow edges is around either -45 and 45 (or vice versa) or
        # 0 and 90. Thus, a threshold of 22.5 will safely seperate groups of
        # tow ends and tow edges
        if orientation < 22.5:
            lines_g1.append(line)
        else:
            lines_g2.append(line)

    lines_g1 = sorted(lines_g1, key=lambda line: line[0])
    lines_g2 = sorted(lines_g2, key=lambda line: line[1])

    merged_lines_all = []

    # for each cluster in group 1 and group 2 lines leave only one line
    for i in [lines_g1, lines_g2]:
        if len(i) > 0:
            groups = group_lines(i)
            merged_lines = []
            for group in groups:
                merged_lines.append(sort_lines(group))

            merged_lines_all.extend(merged_lines)

    merged_lines_all = [
        endpoint[0] + endpoint[1] for endpoint in merged_lines_all
    ]

    return merged_lines_all


# ---------------------------------------------------------------------------


def remove_isolated_lines(lines):
    'Remove lines not "connected" to others'
    n_l = len(lines)
    if n_l > 1:
        dist_m = np.zeros((n_l, n_l))
        orit_diff_m = np.zeros((n_l, n_l))
        # adjacency matrix to find connected components
        adj_m_1 = np.zeros((n_l, n_l))
        adj_m_2 = np.zeros((n_l, n_l))

        for i in range(n_l):
            for j in range(i + 1, n_l):
                dist_m[i, j] = get_distance(lines[i], lines[j])
                orit_i = get_orientation(lines[i])
                orit_j = get_orientation(lines[j])
                orit_diff_m[i, j] = abs(orit_i - orit_j)
                # Case 1: find line segments within a distance and form about 90 deg
                # condition for connection
                if dist_m[i, j] < 50 and abs(orit_diff_m[i, j] - 90) < 10:
                    adj_m_1[i, j] = 1
                # Case 2: find vertical or horizontal line segments almost align
                # condition for connection
                if orit_diff_m[i, j] < 5 and (abs(orit_i) > 85
                                              or abs(orit_i) < 5):
                    if is_aligned(lines[i], lines[j], dist_m[i, j], 5):
                        adj_m_2[i, j] = 1

        # find line segements that are "connnected"
        adj_1_n_components, adj_1_labels = connected_components(
            csgraph=csr_matrix(adj_m_1), directed=False, return_labels=True)
        adj_2_n_components, adj_2_labels = connected_components(
            csgraph=csr_matrix(adj_m_2), directed=False, return_labels=True)
        counts_1 = np.unique(adj_1_labels, return_counts=True)[1]
        counts_2 = np.unique(adj_2_labels, return_counts=True)[1]

        # keep polylines that are composed of more than one line segment
        polylines_1 = np.where(counts_1 > 1)[0]
        polylines_2 = np.where(counts_2 > 1)[0]

        connected_lines_idx_all = []
        if len(polylines_1) > 0:
            for idx_1 in polylines_1:
                connected_lines_idx_all.append(
                    np.where(adj_1_labels == idx_1)[0])
        if len(polylines_2) > 0:
            for idx_2 in polylines_2:
                connected_lines_idx_all.append(
                    np.where(adj_2_labels == idx_2)[0])

        if len(connected_lines_idx_all) > 0:
            filter_lines = np.array(lines)[np.concatenate(
                connected_lines_idx_all).ravel()].tolist()
        else:
            # if all the line segments are isolated from each other, then we keep the longest line segment
            filter_lines = lines[0]
            for line in lines[1:]:
                if lineMagnitude(line) > lineMagnitude(filter_lines):
                    filter_lines = line
            filter_lines = [filter_lines]

        return filter_lines
    else:
        return lines
