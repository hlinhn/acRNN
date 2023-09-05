from read_bvh import parse_frames, get_pos_joints_index
from read_bvh_hierarchy import read_bvh_hierarchy
import sys
import numpy as np
import math
from transforms3d.quaternions import qmult, mat2quat, qconjugate
from visualize import plot_motion
from rotation2xyz import get_skeleton_position
from transforms3d.euler import euler2mat, euler2quat


class BVHSkeleton:
    def __init__(self, sample):
        self.skeleton, self.non_end_bones = read_bvh_hierarchy(sample)
        sample_data = parse_frames(sample)
        self.joint_indices = get_pos_joints_index(sample_data[0], self.non_end_bones, self.skeleton)
        self.chain_name = [['hip', 'abdomen', 'chest', 'neck', 'head'],
                           ['hip', 'rButtock', 'rThigh', 'rShin', 'rFoot'],
                           ['hip', 'lButtock', 'lThigh', 'lShin', 'lFoot'],
                           ['chest', 'rCollar', 'rShldr', 'rForeArm', 'rHand'],
                           ['chest', 'lCollar', 'lShldr', 'lForeArm', 'lHand']]
        self.lookup_indices = {}
        for index, bone in enumerate(self.non_end_bones):
            self.lookup_indices[bone] = 6 + index * 3
        self.lookup_indices['hip'] = 3
        self.offsets = {}
        for bone in self.skeleton.keys():
            self.offsets[bone] = self.skeleton[bone]['offsets']
        self.chain_link = ['hip', 'abdomen', 'chest', 'neck', 'head', 'rButtock', 'rThigh', 'rShin', 'rFoot',
                           'lButtock', 'lThigh', 'lShin', 'lFoot', 'rCollar', 'rShldr', 'rForeArm', 'rHand',
                           'lCollar', 'lShldr', 'lForeArm', 'lHand']
        self.chain_index = [self.lookup_indices[x] for x in self.chain_link]
        self.parent_index = [-1, 0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 2, 13, 14, 15, 2, 17, 18, 19]
        self.offsets_arr = np.zeros((len(self.chain_index), 3))
        for i, link in enumerate(self.chain_link):
            self.offsets_arr[i] = self.offsets[link]
        self.anim_index = [[self.chain_link.index(x) for x in y] for y in self.chain_name]

    def homogeneous(self, rot, vec, hip=False):
        arr = np.zeros((4, 4))
        arr[0:3, 3] = vec
        arr[3, 3] = 1
        rot = [math.radians(x) for x in rot]
        if hip:
            rotation = euler2mat(rot[2], rot[1], rot[0], 'sxyz')
        else:
            rotation = euler2mat(rot[2], rot[1], rot[0], 'syxz')
        arr[0:3, 0:3] = rotation
        return arr

    def dual(self, rot, vec, hip=False):
        rot = [math.radians(x) for x in rot]
        if hip:
            qr = euler2quat(*rot[::-1], 'sxyz')
        else:
            qr = euler2quat(*rot[::-1], 'syxz')
        qt = [0, *vec]
        qd = 0.5 * qmult(qt, qr)
        return np.concatenate((qr, qd))

    def dual_mult(self, q1, q2):
        qr = qmult(q1[:4], q2[:4])
        qd = qmult(q1[:4], q2[4:]) + qmult(q1[4:], q2[:4])
        return np.concatenate((qr, qd))

    def homogeneous_to_dual(self, H):
        qr = mat2quat(H[0:3, 0:3])
        qt = [0, H[0, 3], H[1, 3], H[2, 3]]
        qd = 0.5 * qmult(qt, qr)
        return [*qr, *qd]

    def from_rotation_to_dual(self, data):
        transformed = []
        global_trans = []
        duals = []
        for frame in data:
            frame_trans = np.zeros((len(self.chain_index), 4, 4))
            dual_frame = np.zeros((len(self.chain_index), 8))
            for j, ind in enumerate(self.chain_index):
                # local = self.homogeneous(frame[ind:ind+3], self.offsets_arr[j], hip=(j==0))
                local = self.dual(frame[ind:ind+3], self.offsets_arr[j], hip=(j==0))
                if self.parent_index[j] < 0:
                    # frame_trans[j] = local
                    dual_frame[j] = local
                else:
                    # frame_trans[j] = np.dot(frame_trans[self.parent_index[j]], local)
                    dual_frame[j] = self.dual_mult(dual_frame[self.parent_index[j]], local)
                # dual_frame[j] = self.homogeneous_to_dual(frame_trans[j])
            global_trans.append(np.array(frame[0:3]))
            duals.append(dual_frame)
        duals_np = np.stack(duals, axis=0)
        return duals_np, global_trans

    # two steps: from dual quaternion to rotation
    # current frame to position
    def from_dual_to_position(self, current, global_trans):
        absolute_pos = []
        for i, frame in enumerate(current):
            root_position = global_trans[i]
            arr = np.zeros((len(self.chain_link), 3))
            for j, c in enumerate(frame):
                d = 2 * qmult(c[4:], qconjugate(c[:4]))
                arr[j, :] = d[1:] + root_position
            absolute_pos.append(arr)

        np_position = np.stack(absolute_pos, axis=2)
        return np_position, self.anim_index

    def from_dict_to_array(self, pos_dict):
        arr = np.zeros((len(pos_dict), 3))
        for k in pos_dict.keys():
            ind = self.joint_indices[k]
            for i in range(3):
                arr[ind][i] = pos_dict[k][i]
        return arr

    def from_rotation_to_position(self, data):
        chain_index = [[self.joint_indices[y] for y in x] for x in self.chain_name]
        positions = []
        origin = None
        for i in range(len(data)):
            position = get_skeleton_position(data[i], self.non_end_bones, self.skeleton)
            position_arr = self.from_dict_to_array(position)
            positions.append(position_arr)

        np_positions = np.stack(positions, axis=2)
        return np_positions, chain_index


# numbers to numbers, not differentiable yet
# order zyx
def posrot2dual(p, r):
    qrx = [math.cos(r[0]/2.0), math.sin(r[0]/2.0), 0, 0]
    qry = [math.cos(r[1]/2.0), 0, math.sin(r[1]/2.0), 0]
    qrz = [math.cos(r[2]/2.0), 0, 0, math.sin(r[2]/2.0)]
    qr = qmult(qrz, qry)
    qr = qmult(qr, qrx)
    qt = [0, p[0], p[1], p[2]]
    qd = qmult(qt, qr) / 2.0
    return qr, qd


def dual2posrot(d):
    pass


def main():
    import time
    start = time.time()
    skeleton = BVHSkeleton(sys.argv[1])
    data = parse_frames(sys.argv[2])
    # print(time.time() - start)
    # start = time.time()
    # pos, chain = skeleton.from_rotation_to_position(data)
    # print(pos)
    # plot_motion(pos, chain, interval=500)

    converted, global_trans = skeleton.from_rotation_to_dual(data)
    # test_convert, indices = skeleton.from_dual_to_position(converted, global_trans)
    print(time.time() - start)
    # start = time.time()
    # plot_motion(test_convert, indices, interval=50, save_path="/home/halinh/convert.gif")
    # print(time.time() - start)


if __name__ == '__main__':
    main()
