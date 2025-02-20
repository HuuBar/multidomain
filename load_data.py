import pandas as pd
import json
from sklearn.preprocessing import MinMaxScaler

def load_tag_scores(file_path):
    """加载电影-标签相关性分数"""
    return pd.read_csv(file_path)

def load_tags(file_path):
    """加载标签信息"""
    with open(file_path, 'r') as f:
        tags = [json.loads(line) for line in f]
    return pd.DataFrame(tags)

def load_ratings(file_path):
    """加载用户评分数据"""
    with open(file_path, 'r') as f:
        ratings = [json.loads(line) for line in f]
    return pd.DataFrame(ratings)

def preprocess_data(tag_scores, tags):
    """构建电影-标签矩阵并归一化"""
    movie_tag_matrix = tag_scores.pivot(index='item_id', columns='tag', values='score').fillna(0)
    scaler = MinMaxScaler()
    movie_tag_matrix_normalized = pd.DataFrame(
        scaler.fit_transform(movie_tag_matrix),
        columns=movie_tag_matrix.columns,
        index=movie_tag_matrix.index
    )
    return movie_tag_matrix_normalized