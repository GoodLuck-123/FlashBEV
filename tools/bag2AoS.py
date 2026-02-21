#!/usr/bin/env python3

# 数据处理任务1.2
# python实现离线解包工具：ROS1 Bag -> AoS Binary
# 思考：有没有哪些数据处理可以提前在这一步完成，减轻后续 C++ 端的负担？
#      比如说滤除明显的硬件噪点（x=y=z=0），或者直接在这里做个简单的坐标变换？

import os
import sys
import struct
import argparse
from rosbags.rosbag1 import Reader
from rosbags.typesys import Stores, get_typestore, get_types_from_msg

def parse_args():
    parser = argparse.ArgumentParser(description="FlashBEV 离线解包工具: ROS1 Bag -> AoS Binary (为 float4 准备)")
    parser.add_argument('-i', '--input_bag', type=str, required=True, help="输入的 R3LIVE .bag 文件路径")
    parser.add_argument('-o', '--out_dir', type=str, default="./data_aos_bin", help="输出的 AoS .bin 文件目录")
    parser.add_argument('-t', '--topic', type=str, default="/livox/lidar", help="雷达话题名")
    return parser.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    print(f"[*] 正在暴力拆解 Bag: {args.input_bag}")
    print(f"[*] 目标格式: AoS 交织内存 (16-byte Aligned for CUDA float4)")

    typestore = get_typestore(Stores.ROS1_NOETIC)

    frame_count = 0
    with Reader(args.input_bag) as reader:
        connections = [x for x in reader.connections if x.topic == args.topic]
        if not connections:
            print(f"[!] 没找到话题 {args.topic}。")
            sys.exit(1)

        # 动态注册 Livox 自定义消息 (兼容你的系统环境)
        add_types = {}
        for conn in connections:
            raw_msgdef = conn.msgdef.data if hasattr(conn.msgdef, 'data') else conn.msgdef
            clean_msgtype = conn.msgtype.replace('/msg/', '/')
            add_types.update(get_types_from_msg(raw_msgdef, clean_msgtype))
        typestore.register(add_types)

        print("[*] 类型注册完毕，开始提取交织数据 (AoS)...")

        for connection, timestamp, rawdata in reader.messages(connections=connections):
            msg = typestore.deserialize_ros1(rawdata, connection.msgtype)
            
            if msg.point_num == 0:
                continue

            # 扁平化数组，按照 x, y, z, i 依次塞入
            aos_flat_array = []

            for point in msg.points:
                # 过滤硬件噪点
                if point.x == 0.0 and point.y == 0.0 and point.z == 0.0:
                    continue
                # 严格按照 16 字节 (4个 float) 交织排列
                aos_flat_array.extend([
                    point.x, 
                    point.y, 
                    point.z, 
                    float(point.reflectivity)
                ])

            # 实际有效点数 = 数组长度 / 4
            actual_points = len(aos_flat_array) // 4
            if actual_points == 0:
                continue

            # AoS 写入
            bin_path = os.path.join(args.out_dir, f"frame_{frame_count:06d}.bin")
            with open(bin_path, 'wb') as f:
                # 1. 头部 4 字节：总点数 (无符号 32 位整数)
                f.write(struct.pack('<I', actual_points))
                # 2. 连续写入交织数据: x1, y1, z1, i1, x2, y2...
                # < 代表小端序，保证和 C++ 内存完全一致
                f.write(struct.pack(f'<{len(aos_flat_array)}f', *aos_flat_array))
            
            frame_count += 1
            if frame_count % 100 == 0:
                print(f"[*] 已转换 {frame_count} 帧...")

    print(f"[+] 搞定！共生成 {frame_count} 帧 AoS 格式数据，保存在 {args.out_dir} 中。")

if __name__ == "__main__":
    main()