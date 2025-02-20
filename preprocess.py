import pandas as pd
import json
import os

def precompute_topk_per_dimension(tag_scores_path, output_dir, nk=1000):
    """为每个维度预计算 Top nk 节点并保存"""
    print(f"开始预计算每个维度的 Top {nk} 节点...")

    # 加载标签分数数据
    print(f"正在加载标签分数数据: {tag_scores_path}")
    tag_scores = pd.read_csv(tag_scores_path)
    print("标签分数数据加载完成。")

    # 按标签分组，取每个标签下分数最高的前nk个电影
    print(f"开始处理标签数据以获取每个标签的 Top {nk} 电影...")
    topk_per_tag = {}
    unique_tags = tag_scores['tag'].unique()
    total_tags = len(unique_tags)
    for index, tag in enumerate(unique_tags):
        print(f"正在处理标签 {tag} ({index + 1}/{total_tags})")
        tag_data = tag_scores[tag_scores['tag'] == tag]
        topk_movies = tag_data.nlargest(nk, 'score')[['item_id', 'score']]
        topk_per_tag[tag] = topk_movies.set_index('item_id')['score'].to_dict()
        print(f"标签 {tag} 处理完成，Top {nk} 电影已获取。")
    print("所有标签的 Top 电影获取完成。")

    # 保存到文件
    print(f"开始将结果保存到输出目录: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)
    for tag, movies in topk_per_tag.items():
        safe_tag = tag.replace('/', '_')  # 替换斜杠为下划线以确保路径有效
        filename = os.path.join(output_dir, f"{safe_tag}.json")
        print(f"正在保存标签 {tag} 的 Top {nk} 电影到文件: {filename}")
        with open(filename, 'w') as f:
            json.dump(movies, f)
        print(f"标签 {tag} 的 Top {nk} 电影保存完成。")
    print("所有标签的 Top 电影保存完成，预计算完成。")

if __name__ == "__main__":
    precompute_topk_per_dimension(
        tag_scores_path="data/scores/tagdl.csv",
        output_dir="data/topk_subgraphs",
        nk=1000
    )
