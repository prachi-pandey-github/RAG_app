import os
import torch
import clip
import numpy as np
from PIL import Image
from sklearn.linear_model import LogisticRegression

DATASET_PATH = "dataset"

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load CLIP
model, preprocess = clip.load("ViT-B/32", device=device)

features = []
labels = []

classes = ["animated", "real"]

for label, cls in enumerate(classes):

    folder = os.path.join(DATASET_PATH, cls)

    for file in os.listdir(folder):

        path = os.path.join(folder, file)

        try:
            image = preprocess(Image.open(path)).unsqueeze(0).to(device)

            with torch.no_grad():
                embedding = model.encode_image(image)

            embedding = embedding.cpu().numpy().flatten()

            features.append(embedding)
            labels.append(label)

        except:
            pass

features = np.array(features)
labels = np.array(labels)

print("Dataset size:", len(features))

# Train classifier
clf = LogisticRegression(max_iter=1000)

clf.fit(features, labels)

print("Training complete!")

# Save model
import pickle
pickle.dump(clf, open("classifier.pkl","wb"))