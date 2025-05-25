class ModelFactory:
    def __init__(self, model_class, **kwargs):       
        self.model_class = model_class
        self.kwargs = kwargs

    def build(self):       
        return self.model_class(**self.kwargs)
