import numpy as np
import sys
sys.path.append("/home/halinh/projects/HumanML3D")
from paramUtil import t2m_raw_offsets, t2m_kinematic_chain
from common.skeleton import Skeleton
import torch


def extract_rotation(data):
    pass


def rotation_to_position(positions, rotations):
    # from HumanML3D repo
    root_pos = positions[:, 0]
    skeleton = Skeleton()
    new_joints = skeleton.forward_kinematics_np(rotation, root_pos)
    return new_joints


def extract_position(data):
    pass


def compare_rot_pos(data):
    rotation = extract_rotation(data)
    rotpos = rotation_to_position(data, rotation)
    position = extract_position(data)

    difference = rotpos - position
    mag = np.linalg.norm(difference)
    max_diff = np.max(np.abs(difference))
    print(mag)
    print(max_diff)


def ground_truth_test(data):
    # test HumanML3D data first, expect all diffs to be 0
    pass


def main():
    skeleton = Skeleton(torch.from_numpy(t2m_raw_offsets), t2m_kinematic_chain, "cpu")


if __name__ == "__main__":
    main()
