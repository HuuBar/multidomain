import hnswlib
import numpy as np
import json
import os

def build_subgraph_indices(topk_dir, index_dir, dim=128):
    """为每个维度的 Top nk 节点构建 HNSW 索引"""
    print(f"开始为每个维度的 Top nk 节点构建 HNSW 索引...")

    # 创建索引目录
    print(f"正在创建索引目录: {index_dir}")
    os.makedirs(index_dir, exist_ok=True)
    print(f"索引目录 {index_dir} 创建完成。")

    # 获取目录中的所有文件
    print(f"正在获取目录 {topk_dir} 中的所有文件...")
    files = os.listdir(topk_dir)
    print(f"目录 {topk_dir} 中的文件获取完成。")

    # 遍历每个文件
    for index, filename in enumerate(files):
        if filename.endswith(".json"):
            print(f"正在处理文件 {filename} ({index + 1}/{len(files)})")

            # 恢复原始标签名
            tag = filename[:-5].replace('_', '/')
            print(f"原始标签名恢复完成: {tag}")

            # 读取文件内容
            print(f"正在读取文件内容: {os.path.join(topk_dir, filename)}")
            with open(os.path.join(topk_dir, filename), 'r') as f:
                movies = json.load(f)
            print(f"文件 {filename} 内容读取完成。")

            # 提取节点ID和分数作为向量
            print(f"正在提取节点ID和分数作为向量...")
            item_ids = list(movies.keys())
            scores = np.array(list(movies.values()), dtype=np.float32).reshape(-1, 1)
            print(f"节点ID和分数提取完成。")

            # 构建HNSW索引
            print(f"开始构建 HNSW 索引，维度: {dim}, 元素个数: {len(scores)}")
            index = hnswlib.Index(space='l2', dim=dim)
            index.init_index(max_elements=len(scores), ef_construction=200, M=16)
            index.add_items(scores, np.arange(len(scores)))
            print(f"HNSW 索引构建完成。")

            # 保存索引和节点ID映射
            safe_tag = tag.replace('/', '_')  # 替换斜杠为下划线以确保路径有效
            index_file = os.path.join(index_dir, f"{safe_tag}.bin")
            ids_file = os.path.join(index_dir, f"{safe_tag}_ids.json")

            print(f"正在保存 HNSW 索引到文件: {index_file}")
            index.save_index(index_file)
            print(f"HNSW 索引保存完成。")

            print(f"正在保存节点ID映射到文件: {ids_file}")
            with open(ids_file, 'w') as f:
                json.dump(item_ids, f)
            print(f"节点ID映射保存完成。")

    print("所有文件处理完成，HNSW 索引构建和保存完成。")

if __name__ == "__main__":
    build_subgraph_indices(
        topk_dir="data/topk_subgraphs",
        index_dir="data/subgraph_indices",
        dim=1  # 单维度分数
    )
