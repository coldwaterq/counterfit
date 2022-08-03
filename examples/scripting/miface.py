from hashlib import new
from PIL import Image
from counterfit.targets import CFTarget
from datasets import load_dataset
import torch
import io
import numpy as np
import os
import torchvision

from counterfit.modules.algo.art.inference.miface import CFMIFace


class MyTarget(CFTarget):
    def __init__(self):
        # self.input_shape = (640, 480, 3)
        self.output_classes = [i for i in range(0, 1000)]

        # Load the model
        dataset = load_dataset("huggingface/cats-image")

        self.image = np.array(dataset["test"]["image"][0]).astype(np.float32)

        self.input_shape = self.image.shape

        self.feature_extractor = AutoFeatureExtractor.from_pretrained(
            "microsoft/resnet-50"
        )

        self.endpoint = ResNetForImageClassification.from_pretrained(
            "microsoft/resnet-50"
        )

    def predict(self, x):

        # Convert to x to images so we can use the feature extractor.
        imgs = []
        for i in x:
            new_img = Image.fromarray(np.array(i).astype(np.uint8), "RGB")

            # Rebuild the batch from images.
            new_x = (
                self.feature_extractor(new_img, return_tensors="pt")
                .get("pixel_values")
                .numpy()
            )[0]

            imgs.append(new_x)

        # Keep this because I don't want to lose it.
        # membuf = io.BytesIO()
        # img.save(membuf, format="png")

        try:
            x = torch.from_numpy(np.array(imgs))
        except:
            pass

        with torch.no_grad():
            logits = self.endpoint(x).logits

        return logits.numpy()


if __name__ == "__main__":
    # Test the load and predict
    print("creating target")
    target  = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)
    input_shape = (3,256,256)
    num_classes = 1000
    print("running MIFace")
    mif = CFMIFace(framework="PyTorch")
    x = None
    y = 1
    for i in range(10):
        mif.run(target, x=x, y=y, input_shape=input_shape, num_classes=num_classes)
        x = mif.results
        os.rename(f"/results/{y}.png",f"/results/i{i}.png")
    # Moved the following lines to art.py so that it would write out images as it 
    # discovered them instead of all at the end
    # print(len(mif.results))
    # print(mif.results[0].shape)
    # for i in range(len(mif.results)):
    #     result = mif.results[i]*255
    #     result = np.array(result, dtype=np.uint8)
    #     result = np.moveaxis(result, 0, -1)
    #     image = Image.fromarray(result)
    #     image.save(f"/results/{i}.png")


    print("done")
    # Image.fromarray((im_a * 255).astype(np.uint8), "RGB").save("./maybe.png")
