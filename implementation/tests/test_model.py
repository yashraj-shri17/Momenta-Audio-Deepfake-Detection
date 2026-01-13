
import unittest
import torch
import sys
import os

# Add implementation to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model import CNN_GRU_Model

class TestModel(unittest.TestCase):
    def test_model_structure(self):
        model = CNN_GRU_Model()
        # Input: (Batch, Channels, Length) -> (1, 1, 16000) for 1 sec audio
        dummy_input = torch.randn(1, 1, 16000)
        output = model(dummy_input)
        
        self.assertEqual(output.shape, (1, 1))
        # Sigmoid output should be in [0, 1]
        self.assertTrue(0 <= output.item() <= 1)

if __name__ == '__main__':
    unittest.main()
