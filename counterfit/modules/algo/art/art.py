import numpy as np
from counterfit.module import CFAlgo
from counterfit.targets import CFTarget


from art.estimators.classification import PyTorchClassifier
from art.utils import compute_success_array


class ArtInferenceAttack(CFAlgo):
    def __init__(self, framework):
        if framework == "PyTorch":
            self.classifier_cls =  PyTorchClassifier
        else:
            raise Exception(framework, "unsupported")
        super(ArtInferenceAttack, self).__init__()

    def run(
        self,
        target,
        y,
        input_shape,
        num_classes,
        loss,
        x=None,
    ) -> bool:

        """
        Build the attack.
        """

        target_classifier = self.classifier_cls(target, input_shape=input_shape, nb_classes=num_classes, loss=loss)
        attack = self.attack_cls(target_classifier)

        self.pre_attack_processing()
    
        y_onehot = None
        # create one-hot
        if type(y) == int:
            a = np.array([y])
            y_onehot = np.zeros((a.size, num_classes))
            y_onehot[np.arange(a.size),a] = 1
        elif len(y[0]) != num_classes:
            y_onehot = np.zeros((y.size, num_classes))
            y_onehot[np.arange(y.size),y] = 1
        else:
            y_onehot = y

        if x:
            self.results = attack.infer(x=x, y=y_onehot)
        else:
            self.results = attack.infer(None, y=y_onehot)
        return True

    def pre_attack_processing(self):
        pass

    def post_attack_processing(self):
        pass

    def inference_success(self, cfattack):
        pass

    def set_parameters(self) -> None:
        # ART has its own set_params function. Use it.
        attack_params = {}
        for k, v in self.options.attack_parameters.items():
            attack_params[k] = v["current"]
        self.attack.set_params(**attack_params)

class ArtEvasionAttack(CFAlgo):
    def run(
        self,
        target: CFTarget,
        x,
        y=None,
        params=None,
        channels_first: bool = False,
        clip_values: tuple = (0, 1),
    ) -> bool:

        """
        Build the attack.
        """

        # Build the blackbox classifier
        target_classifier = BlackBoxClassifierNeuralNetwork(
            target.predict_wrapper,
            target.input_shape,
            len(target.output_classes),
            channels_first=channels_first,
            clip_values=clip_values,
        )

        attack = self.attack_cls(target_classifier)

        self.pre_attack_processing()
        if y:
            self.results = attack.generate(x=np.array(x, dtype=np.float32), y=y)
        else:
            self.results = attack.generate(x)
        return True

    def pre_attack_processing(self):
        pass

    def post_attack_processing(self):
        pass

    def evasion_success(self, cfattack):
        if cfattack.options.__dict__.get("targeted") == True:
            labels = cfattack.options.target_labels
            targeted = True
        else:
            labels = cfattack.initial_labels
            targeted = False

        success = compute_success_array(
            cfattack.attack._estimator,
            cfattack.samples,
            labels,
            cfattack.results,
            targeted,
        )

        final_outputs, final_labels = cfattack.target.get_sample_labels(
            cfattack.results
        )

        cfattack.final_labels = final_labels
        cfattack.final_outputs = final_outputs
        return success

    def set_parameters(self) -> None:
        # ART has its own set_params function. Use it.
        attack_params = {}
        for k, v in self.options.attack_parameters.items():
            attack_params[k] = v["current"]
        self.attack.set_params(**attack_params)