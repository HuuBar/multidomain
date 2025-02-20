from sklearn.metrics import ndcg_score

def evaluate_recommendations(topk_results, user_ratings, user_id):
    """评估推荐结果的准确性"""
    true_preferences = user_ratings[user_id].values  # 假设 user_ratings 是用户-电影评分矩阵
    predicted_scores = [score for _, score in topk_results]
    ndcg = ndcg_score([true_preferences], [predicted_scores])
    return ndcg