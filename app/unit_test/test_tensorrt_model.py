# Model loading Unit Test

import unittest
import tensorrt as trt
import os

class TestTensorRTModel(unittest.TestCase):
    def setUp(self):
        """Setup the test environment, including the TensorRT logger."""
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.model_path = "/home/dp/lisa/main/models/bedsheet_v11.engine"  # Update with your .engine file path

    def test_model_file_exists(self):
        """Check if the model file exists."""
        self.assertTrue(os.path.exists(self.model_path), f"Model file not found: {self.model_path}")

    def test_model_loading(self):
        """Verify that the model can be loaded by TensorRT."""
        with open(self.model_path, 'rb') as f, trt.Runtime(self.logger) as runtime:
            engine = runtime.deserialize_cuda_engine(f.read())
            self.assertIsNotNone(engine, "Failed to load the TensorRT model.")
            # Optionally, check the number of bindings (inputs and outputs)
            num_bindings = engine.num_bindings
            self.assertGreater(num_bindings, 0, "Model has no input/output bindings.")

    def test_model_execution_context(self):
        """Verify that an execution context can be created."""
        with open(self.model_path, 'rb') as f, trt.Runtime(self.logger) as runtime:
            engine = runtime.deserialize_cuda_engine(f.read())
            self.assertIsNotNone(engine, "Failed to load the TensorRT model.")
            context = engine.create_execution_context()
            self.assertIsNotNone(context, "Failed to create an execution context for the model.")

if __name__ == '__main__':
    unittest.main()
