import os
import json
import logging

logging.basicConfig(level=logging.DEBUG)

class DynamicTopKQuery:
    def __init__(self, index_dir):
        self.index_dir = index_dir
        self.indices = {}

    def load_index(self, label):
        safe_label = label.replace('/', '_')  # 替换斜杠为下划线以确保路径有效
        index_path = os.path.join(self.index_dir, f"{safe_label}.json")
        if os.path.exists(index_path):
            with open(index_path, 'r') as f:
                self.indices[label] = json.load(f)
            logging.debug(f"Index for label {label} loaded from {index_path}")
        else:
            logging.error(f"Index for label {label} not found at path: {index_path}")

    def load_indices(self, labels):
        for label in labels:
            self.load_index(label)

    def query(self, query_weights, k=10):
        # 假设这里有一些查询逻辑
        results = {}
        for label, weight in query_weights.items():
            if label in self.indices:
                # 获取标签对应的TopK电影及其分数
                tag_results = self.indices[label]
                for item_id, score in tag_results.items():
                    if item_id not in results:
                        results[item_id] = {}
                    results[item_id][label] = score
                    results[item_id]['weighted_score'] = results[item_id].get('weighted_score', 0) + score * weight
        
        # 获取前k个结果
        sorted_results = sorted(results.items(), key=lambda x: x[1]['weighted_score'], reverse=True)
        logging.debug(f"Query results: {sorted_results[:k]}")
        return sorted_results[:k]
