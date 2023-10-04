from read_bvh import parse_frames, get_pos_joints_index
from read_bvh_hierarchy import read_bvh_hierarchy
import sys
import numpy as np
import math
from transforms3d.quaternions import qmult, mat2quat, qconjugate
from visualize import plot_motion
from rotation2xyz import get_skeleton_position
from transforms3d.euler import euler2mat, euler2quat
from tqdm import tqdm
import os
from pathlib import Path


def normalize(dual):
    rot = dual[:4]
    trans = dual[4:]
    rot_mag = np.sqrt(np.sum(rot ** 2))
    print(rot_mag)
    dot = np.dot(rot, trans)
    print(dot)
    dual[:4] /= rot_mag
    dual[4:] = trans / rot_mag - dual[:4] * dot / rot_mag ** 2
    return dual


def pick_closer(dual, target):
    pass


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
        self.chain_link = []
        for chain in self.chain_name:
            for link in chain:
                if link not in self.chain_link:
                    self.chain_link.append(link)
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
        global_trans = []
        duals = []
        for frame in data:
            frame_trans = np.zeros((len(self.chain_index), 4, 4))
            dual_frame = np.zeros((len(self.chain_index), 8))
            for j, ind in enumerate(self.chain_index):
                local = self.dual(frame[ind:ind+3], self.offsets_arr[j], hip=(j==0))
                if self.parent_index[j] < 0:
                    dual_frame[j] = local
                else:
                    dual_frame[j] = self.dual_mult(dual_frame[self.parent_index[j]], local)
            global_trans.append(np.array(frame[0:3]))
            duals.append(dual_frame)
        duals_np = np.stack(duals, axis=0)
        global_trans_np = np.stack(global_trans, axis=0)
        return duals_np, global_trans_np

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


def convert_data(folder):
    if not os.path.isfile(os.path.join(folder, "standard.bvh")):
        print("Standard file does not exist")
        return
    path = Path(folder)
    output_path = os.path.join(path.parent, "train_dual_quaternion")
    os.makedirs(output_path, exist_ok=True)
    for f in os.listdir(folder):
        if not os.path.isdir(os.path.join(folder, f)):
            continue
        print(f"Processing {f}")
        child_folder = os.path.join(output_path, f)
        os.makedirs(child_folder, exist_ok=True)
        for bvh in tqdm(os.listdir(os.path.join(folder, f))):
            filename = os.path.join(folder, f, bvh)
            skeleton = BVHSkeleton(filename)
            data = parse_frames(filename)
            dual, trans = skeleton.from_rotation_to_dual(data)
            saved_file = os.path.join(child_folder, bvh[:-4])
            np.savez(saved_file, dual, trans)


def test_visualization(filename):
    skeleton = BVHSkeleton(filename)
    data = parse_frames(filename)
    dual, trans = skeleton.from_rotation_to_dual(data)
    for i in range(trans.shape[0]):
        trans[i][1] = 0
    converted, index = skeleton.from_dual_to_position(dual, trans)
    plot_motion(converted, index, interval=500)


def test_load(filename):
    data = np.load(filename)
    dual = data['arr_0']
    glob = data['arr_1']
    dual = dual.reshape(dual.shape[0], -1)
    composite = np.hstack((glob, dual))
    test_data = dual[1000, 5*8:6*8]
    print(test_data)
    normed = normalize(test_data)
    print(normed)
    print(test_data)


def main():
    # convert_data(sys.argv[1])
    test_load(sys.argv[1])
    # test_visualization(sys.argv[1])


if __name__ == '__main__':
    main()
