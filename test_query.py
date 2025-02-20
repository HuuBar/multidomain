import os
import json
import unittest
from query import DynamicTopKQuery

# 创建测试数据
test_data_dir = "test_data/topk_subgraphs"
os.makedirs(test_data_dir, exist_ok=True)

romantic_json = {
    "item_id1": 0.9,
    "item_id2": 0.85,
    "item_id3": 0.8,
    "item_id4": 0.75,
    "item_id5": 0.7,
    "item_id6": 0.65,
    "item_id7": 0.6,
    "item_id8": 0.55,
    "item_id9": 0.5,
    "item_id10": 0.45
}

action_json = {
    "item_id1": 0.8,
    "item_id2": 0.75,
    "item_id3": 0.7,
    "item_id4": 0.65,
    "item_id5": 0.6,
    "item_id6": 0.55,
    "item_id7": 0.5,
    "item_id8": 0.45,
    "item_id9": 0.4,
    "item_id10": 0.35
}

with open(os.path.join(test_data_dir, "romantic.json"), 'w') as f:
    json.dump(romantic_json, f)

with open(os.path.join(test_data_dir, "action.json"), 'w') as f:
    json.dump(action_json, f)

class TestDynamicTopKQuery(unittest.TestCase):
    def setUp(self):
        self.query_engine = DynamicTopKQuery(index_dir=test_data_dir)

    def test_load_indices(self):
        query_labels = ["romantic", "action"]
        self.query_engine.load_indices(query_labels)
        self.assertIn("romantic", self.query_engine.indices)
        self.assertIn("action", self.query_engine.indices)

    def test_query(self):
        query_weights = {"romantic": 0.6, "action": 0.4}
        expected_results = [
            ('item_id1', 0.86),
            ('item_id2', 0.81),
            ('item_id3', 0.76),
            ('item_id4', 0.71),
            ('item_id5', 0.66),
            ('item_id6', 0.61),
            ('item_id7', 0.56),
            ('item_id8', 0.51),
            ('item_id9', 0.46),
            ('item_id10', 0.41)
        ]
        query_labels = list(query_weights.keys())
        self.query_engine.load_indices(query_labels)
        topk_results = self.query_engine.query(query_weights, k=10)
        self.assertEqual(topk_results, expected_results)

if __name__ == "__main__":
    unittest.main()
