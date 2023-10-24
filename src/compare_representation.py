import numpy as np
import sys
sys.path.append("/home/halinh/projects/HumanML3D")
from paramUtil import t2m_raw_offsets, t2m_kinematic_chain
from common.skeleton import Skeleton
import torch
from common.quaternion import quaternion_to_cont6d, qrot, qinv
from visualize import plot_motion
import os
import glob


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


def bones_length(data, chain):
    total = 0
    for c in chain:
        for i in range(len(c) - 1):
            diff = np.linalg.norm(data[i+1] - data[i])
            total += diff
    return total


def compare_rot_pos(data, skeleton):
    position = extract_position(data)
    rotpos = extract_rotation(data, skeleton)

    difference = (rotpos - position[0]).numpy()
    mag = np.linalg.norm(difference)
    max_diff = np.max(np.abs(difference))

    ref_offset = skeleton._offset
    print(ref_offset)
    bone_ref = np.sum([np.linalg.norm(x) for x in ref_offset])
    print(bone_ref)
    rotpos_bones = []
    for d in rotpos:
        rotpos_bones.append(bones_length(d, t2m_kinematic_chain))

    pos_bones = []
    for d in position[0]:
        pos_bones.append(bones_length(d, t2m_kinematic_chain))

    # why are my rotpos bones not constant?
    rotpos_bones = np.asarray(rotpos_bones)
    pos_bones = np.asarray(pos_bones)
    print(pos_bones.shape)

    print(np.mean(rotpos_bones))
    print(np.std(rotpos_bones))
    print(np.mean(pos_bones))
    print(np.std(pos_bones))

    print(mag / len(rotpos))
    print(max_diff)

    plot = False
    if plot:
        plot_position = position[0].numpy().transpose(1, 2, 0)
        plot_rotation = rotpos.numpy().transpose(1, 2, 0)
        plot_motion(plot_position, t2m_kinematic_chain, interval=100, save_path="from_position.gif")
        plot_motion(plot_rotation, t2m_kinematic_chain, interval=100, save_path="from_rotation.gif")


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


def random_check_sequence(sequence, size=50):
    # within sequence the variance is small as expected
    L = len(sequence)
    idx = np.random.randint(L, size=size)
    offset_values = []
    for i in idx:
        data = sequence[i]
        skeleton = Skeleton(torch.from_numpy(t2m_raw_offsets), t2m_kinematic_chain, "cpu")
        offsets = skeleton.get_offsets_joints(data)
        offset_values.append(offsets)

    bones_length = len(offset_values[0])
    cv = []
    diff = []
    for i in range(bones_length):
        collected = [np.linalg.norm(offset_bones[i]) for offset_bones in offset_values]
        std = np.std(collected)
        mean = np.mean(collected)
        bone_diff = abs(collected - mean)
        if mean > 0:
            cv.append(std/mean)
            diff.append(max(bone_diff) / mean)
        # if std > 1e-6:
            # print(f"Bones {i} mean {mean} std {std}")
    # print(min(diff))
    # print(max(diff))
    print(max(cv))


def to_joints(data, convert="joint"):
    if convert == "joint":
        data = data.reshape(len(data), -1, 3)
    elif convert == "all":
        # data = data[..., 4:67]
        # r_pos = np.zeros((len(data), 3))
        # data = np.hstack([r_pos, data])
        # data = data.reshape(len(data), -1, 3)
        data = extract_position(torch.from_numpy(data))
        data = data.numpy()
    else:
        position_data = extract_position(torch.from_numpy(data))
        skeleton = Skeleton(torch.from_numpy(t2m_raw_offsets), t2m_kinematic_chain, "cpu")
        target_offsets = skeleton.get_offsets_joints(position_data[0])
        skeleton.set_offset(target_offsets)
        data = extract_rotation(torch.from_numpy(data), skeleton)
        data = data.numpy()
    return data



def check_across_sequence(folder, size=50, convert="rot"):
    # this also passed
    file_list = [x for x in glob.glob(f"{folder}/*.npy")]
    idx = np.random.randint(len(file_list), size=size)
    offset_values = []
    for i in idx:
        data = np.load(os.path.join(folder, file_list[i]))
        data = to_joints(data, convert)
        index = np.random.randint(len(data))
        data = torch.from_numpy(data)[index] # this is okay because within sequence it's correct
        skeleton = Skeleton(torch.from_numpy(t2m_raw_offsets), t2m_kinematic_chain, "cpu")
        offsets = skeleton.get_offsets_joints(data)
        offset_values.append(offsets)

    bones_length = len(offset_values[0])
    cv = []
    for i in range(bones_length):
        collected = [np.linalg.norm(offset_bones[i]) for offset_bones in offset_values]
        std = np.std(collected)
        mean = np.mean(collected)
        if mean > 0:
            cv.append(std/mean)
        # if std > 1e-6:
        #     mean = np.mean(collected)
        #     print(f"Bones {i} mean {mean} std {std}")
    print(min(cv))
    print(max(cv))


def random_check(folder, size=50, convert="rot"):
    file_list = [x for x in glob.glob(f"{folder}/*.npy")]
    idx = np.random.randint(len(file_list), size=size)
    for i in idx:
        print(f"Checking {file_list[i]}")
        data = np.load(os.path.join(folder, file_list[i]))
        data = to_joints(data, convert)
        data = torch.from_numpy(data)
        random_check_sequence(data)


def main():
    # random_check(sys.argv[1], convert=True)
    # check_across_sequence(sys.argv[1], convert="rot")
    ground_truth_test(sys.argv[1], "/home/halinh/projects/HumanML3D/HumanML3D/new_joints/012314.npy")


if __name__ == "__main__":
    main()
