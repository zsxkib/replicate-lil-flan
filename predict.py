from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from cog import BasePredictor, Input

CACHE_DIR = "/src/models"

class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        self.model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small", cache_dir=CACHE_DIR)
        self.tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small", cache_dir=CACHE_DIR)

    def predict(
        self,
        prompt: str = Input(description="Prompt for language model"),
    ) -> str:
        """Run a single prediction on the model"""
        inputs = self.tokenizer(prompt, return_tensors="pt").input_ids
        outputs = self.model.generate(inputs)
        return self.tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]