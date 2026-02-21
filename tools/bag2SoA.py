#!/usr/bin/env python3
import os
import sys
import struct
import argparse
from rosbags.rosbag1 import Reader
from rosbags.typesys import Stores, get_typestore, get_types_from_msg

def parse_args():
    parser = argparse.ArgumentParser(description="FlashBEV 离线解包工具: ROS1 Bag -> SoA Binary")
    parser.add_argument('-i', '--input_bag', type=str, required=True, help="输入的 R3LIVE .bag 文件路径")
    parser.add_argument('-o', '--out_dir', type=str, default="./data_bin", help="输出的 .bin 文件目录")
    parser.add_argument('-t', '--topic', type=str, default="/livox/lidar", help="雷达话题名")
    return parser.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    print(f"[*] 正在暴力拆解 Bag: {args.input_bag}")

    # 1. 初始化 Typestore (使用 ROS 1 Noetic 的基础类型)
    typestore = get_typestore(Stores.ROS1_NOETIC)

    frame_count = 0
    with Reader(args.input_bag) as reader:
        connections = [x for x in reader.connections if x.topic == args.topic]
        if not connections:
            print(f"[!] 没找到话题 {args.topic}。")
            sys.exit(1)

        # 2. 从 bag 的连接头中动态提取并注册 Livox 的自定义消息结构
        add_types = {}
        for conn in connections:
            # 如果是个对象，就提取它的 .data 属性；如果是老版本，就直接用
            raw_msgdef = conn.msgdef.data if hasattr(conn.msgdef, 'data') else conn.msgdef
            
            # 有些 bag 里的 msgtype 会带有 'msg/' 路径，做个清理
            clean_msgtype = conn.msgtype.replace('/msg/', '/')
            
            add_types.update(get_types_from_msg(raw_msgdef, clean_msgtype))
        typestore.register(add_types)
        print("[*] 自定义消息类型注册完毕！开始提取...")

        # 3. 遍历并反序列化
        for connection, timestamp, rawdata in reader.messages(connections=connections):
            # 采用全新的 typestore.deserialize_ros1 API
            msg = typestore.deserialize_ros1(rawdata, connection.msgtype)
            
            num_points = msg.point_num
            if num_points == 0:
                continue

            x_arr, y_arr, z_arr, i_arr = [], [], [], []

            for point in msg.points:
                if point.x == 0.0 and point.y == 0.0 and point.z == 0.0:
                    continue
                x_arr.append(point.x)
                y_arr.append(point.y)
                z_arr.append(point.z)
                i_arr.append(float(point.reflectivity))

            actual_points = len(x_arr)
            if actual_points == 0:
                continue

            # 4. 极致的 SoA 内存对齐写入
            bin_path = os.path.join(args.out_dir, f"frame_{frame_count:06d}.bin")
            with open(bin_path, 'wb') as f:
                f.write(struct.pack('<I', actual_points))
                f.write(struct.pack(f'<{actual_points}f', *x_arr))
                f.write(struct.pack(f'<{actual_points}f', *y_arr))
                f.write(struct.pack(f'<{actual_points}f', *z_arr))
                f.write(struct.pack(f'<{actual_points}f', *i_arr))
            
            frame_count += 1
            if frame_count % 100 == 0:
                print(f"[*] 已转换 {frame_count} 帧...")

    print(f"[+] 共提取 {frame_count} 帧至 {args.out_dir}")

if __name__ == "__main__":
    main()