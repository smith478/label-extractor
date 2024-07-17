import unittest
from unittest.mock import Mock, patch
import numpy as np
from rag_retriever import build_retriever_client, evaluate_retrieval, main

class TestRetrievalFunctions(unittest.TestCase):

    # To run the unit tests in this file, run `python -m unittest test_rag_retriever.py`

    def test_build_retriever_client(self):
        with patch('your_module_name.QdrantClient') as mock_client:
            mock_client.return_value.add.return_value = None
            labels = ["label1", "label2"]
            result = build_retriever_client(labels, "test_collection", 3, "test_vectorizer")
            self.assertIsInstance(result, QdrantRM)
            mock_client.return_value.set_model.assert_called_once_with("test_vectorizer")
            mock_client.return_value.add.assert_called_once()

    def test_evaluate_retrieval(self):
        mock_retriever = Mock()
        mock_retriever.forward.return_value = [{'long_text': 'label1'}, {'long_text': 'label2'}]
        
        reports = ["report1", "report2"]
        ground_truth = [["label1"], ["label3"]]
        
        results, mrr, recall = evaluate_retrieval(reports, ground_truth, mock_retriever)
        
        self.assertEqual(len(results), 2)
        self.assertAlmostEqual(mrr, 0.75)  # (1/1 + 1/3) / 2
        self.assertAlmostEqual(recall, 0.5)  # 1 out of 2 labels found

    @patch('your_module_name.build_retriever_client')
    @patch('your_module_name.evaluate_retrieval')
    @patch('builtins.open', new_callable=unittest.mock.mock_open)
    @patch('csv.DictWriter')
    def test_main(self, mock_csv_writer, mock_open, mock_evaluate, mock_build_client):
        mock_build_client.return_value = Mock()
        mock_evaluate.return_value = ([], 0.5, 0.6)
        
        main()
        
        self.assertEqual(mock_build_client.call_count, 5)  # Once for each vectorizer
        self.assertEqual(mock_evaluate.call_count, 5)  # Once for each vectorizer
        self.assertEqual(mock_open.call_count, 2)  # Once for each CSV file
        self.assertEqual(mock_csv_writer.return_value.writeheader.call_count, 2)  # Once for each CSV file

if __name__ == '__main__':
    unittest.main()