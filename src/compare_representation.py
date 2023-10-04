import numpy as np
import sys
sys.path.append("/home/halinh/projects/HumanML3D")
from paramUtil import t2m_raw_offsets, t2m_kinematic_chain
from common.skeleton import Skeleton
import torch
from common.quaternion import quaternion_to_cont6d, qrot, qinv


# data: [r_vel, l_vel(2), root_y, pos(21 * 3), rot(21 * 6), local_vel(22* 3), feet_l(2), feet_r(2)]

joints_num = 22

# from HumanML3D
def recover_root_rot_pos(data):
    rot_vel = data[..., 0]
    r_rot_ang = torch.zeros_like(rot_vel).to(data.device)
    '''Get Y-axis rotation from rotation velocity'''
    r_rot_ang[..., 1:] = rot_vel[..., :-1]
    r_rot_ang = torch.cumsum(r_rot_ang, dim=-1)

    r_rot_quat = torch.zeros(data.shape[:-1] + (4,)).to(data.device)
    r_rot_quat[..., 0] = torch.cos(r_rot_ang)
    r_rot_quat[..., 2] = torch.sin(r_rot_ang)

    r_pos = torch.zeros(data.shape[:-1] + (3,)).to(data.device)
    r_pos[..., 1:, [0, 2]] = data[..., :-1, 1:3]
    '''Add Y-axis rotation to root position'''
    r_pos = qrot(qinv(r_rot_quat), r_pos)

    r_pos = torch.cumsum(r_pos, dim=-2)

    r_pos[..., 1] = data[..., 3]
    return r_rot_quat, r_pos


def extract_rotation(data, skeleton):
    rotation = data[..., 67:193]
    r_rot_quat, r_pos = recover_root_rot_pos(data)
    r_rot_cont6d = quaternion_to_cont6d(r_rot_quat)
    rotation = torch.cat([r_rot_cont6d, rotation], dim=-1)
    rotation = rotation.view(-1, joints_num, 6)
    new_joints = skeleton.forward_kinematics_cont6d(rotation, r_pos)
    return new_joints


def extract_position(data):
    r_rot_quat, r_pos = recover_root_rot_pos(data)
    positions = data[..., 4:67]
    positions = positions.view(positions.shape[:-1] + (-1, 3))
    positions = qrot(qinv(r_rot_quat[..., None, :]).expand(positions.shape[:-1] + (4,)), positions)
    positions[..., 0] += r_pos[..., 0:1]
    positions[..., 2] += r_pos[..., 2:3]
    positions = torch.cat([r_pos.unsqueeze(-2), positions], dim=-2)
    return positions


def compare_rot_pos(data, skeleton):
    rotpos = extract_rotation(data, skeleton)
    position = extract_position(data)

    difference = (rotpos - position).numpy()
    mag = np.linalg.norm(difference)
    max_diff = np.max(np.abs(difference))
    print(mag)
    print(max_diff)


def ground_truth_test(filename, example_file):
    data = np.load(filename)
    data = torch.from_numpy(data).unsqueeze(0).float()
    skeleton = Skeleton(torch.from_numpy(t2m_raw_offsets), t2m_kinematic_chain, "cpu")
    target_offsets = skeleton.get_offsets_joints(get_offsets(example_file))
    skeleton.set_offset(target_offsets)
    compare_rot_pos(data, skeleton)


def get_offsets(example):
    example_data = np.load(example)
    example_data = example_data.reshape(len(example_data), -1, 3)
    example_data = torch.from_numpy(example_data)
    return example_data[0]


def main():
    ground_truth_test(sys.argv[1], "/home/halinh/projects/HumanML3D/HumanML3D/new_joints/012314.npy")


if __name__ == "__main__":
    main()
