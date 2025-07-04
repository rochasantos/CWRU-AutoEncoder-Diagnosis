class ModelFactory:
    def __init__(self, model_class, classifier_layer=None, **kwargs):       
        self.model_class = model_class
        self.classifier_layer = classifier_layer
        self.kwargs = kwargs

    def build(self):
        model = self.model_class(**self.kwargs)
        
        if self.classifier_layer is not None:
            if hasattr(model, 'classifier'):
                model.classifier = self.classifier_layer
            elif hasattr(model, 'fc'):
                model.fc = self.classifier_layer
            else:
                raise ValueError("Model instance does not have a classifier or fc attribute to replace.")
        
        return model
