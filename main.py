from query import DynamicTopKQuery

def main():
    # 用户查询（示例：浪漫和动作的权重）
    query_weights = {"romantic": 0.6, "action": 0.4}
    
    # 提取用户查询的标签
    query_labels = list(query_weights.keys())
    
    # 初始化查询器
    query_engine = DynamicTopKQuery(index_dir="data/topk_subgraphs")
    
    # 根据用户查询的标签加载相应的索引
    query_engine.load_indices(query_labels)
    
    # 获取查询结果
    topk_results = query_engine.query(query_weights, k=10)
    
    # 打印TopK结果，包含每个领域的原始分数
    print("TopK Results:")
    for item_id, scores in topk_results:
        print(f"Item ID: {item_id}, Weighted Score: {scores['weighted_score']:.4f}")
        for label in query_labels:
            if label in scores:
                print(f"  {label}: {scores[label]:.4f}")
            else:
                print(f"  {label}: 0.0000")

if __name__ == "__main__":
    main()
