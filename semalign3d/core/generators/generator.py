import torch


class Generator:

    def __init__(self, verbose=False):
        self.verbose = verbose

    def generate(self, category: str):
        """Generate data for a specific category."""
        raise NotImplementedError("Subclasses should implement this method.")

    def _log(self, *msg):
        """Log a message"""
        if self.verbose:
            print(*msg)

    def process_categories(self, categories):
        for category in categories:
            try:
                torch.cuda.empty_cache()  # Clear GPU memory before processing each category
                print(f"Processing category: {category}")
                self.generate(category)
            except Exception as e:
                print(f"Error processing category {category}: {e}")
        print("All categories processed.")
