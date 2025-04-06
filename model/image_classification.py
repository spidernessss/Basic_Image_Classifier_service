import os
import time
import pickle
import cv2
import numpy as np
from keras.applications.regnet import preprocess_input
from model.model_util import DeepModel


class ImageClassifier:
    def __init__(self):
        print("Loading ImageClassifier")
        print()
        self.all_skus = {}
        self.model = DeepModel()
        self.predict_time = 0
        self.time_search = 0
        self.count_frame = 0
        self.top_k = 5
        # Categories
        self.classes_average_vectors = {}

    def extract_features(self, image):
        """Extracts features from an image using the MobileNet model."""
        print("Extracting features from image: ", image)
        image = cv2.resize(image, (224, 224))
        image = preprocess_input(image)
        image = np.expand_dims(image, axis=0)
        feature = self.model.extract_feature(image)
        print("Extracted features are: ", feature)
        return feature

    def predict(self, image):
        """Predicts the closest matching class for an input image using cosine similarity."""
        print("Doing model.predict ")
        print()
        self.count_frame += 1
        before_time = time.time()
        target_features = self.extract_features(image)
        self.predict_time += time.time() - before_time

        best_match = None
        max_similarity = -float('inf')
        for class_name, class_vector in self.classes_average_vectors.items():
            print("The image_features are ", target_features)
            similarity = self.model.distance(target_features, class_vector)
            print("For class_name ", class_name, " For class_vector ", class_vector, " The distance is ", similarity)
            if similarity > max_similarity:
                max_similarity = similarity
                best_match = class_name
        print("Best_match is ", best_match, " Max_similarity is ", max_similarity)
        return best_match, max_similarity

    def add_class(self, class_name, image_folder):
        """Calculates the average feature vector for a class from all images in the folder."""
        class_vectors = []
        for filename in os.listdir(image_folder):
            image_path = os.path.join(image_folder, filename)
            image = cv2.imread(image_path)
            features = self.extract_features(image)
            class_vectors.append(features)
        self.classes_average_vectors[class_name] = np.mean(class_vectors, axis=0)
        print("Added classes: ", self.classes_average_vectors)

    # FOR MILVUS
    def get_class_vectors(self, class_name, image_folder):
        """Calculates the average feature vector for a class from all images in the folder."""
        class_vectors = []
        for filename in os.listdir(image_folder):
            image_path = os.path.join(image_folder, filename)
            image = cv2.imread(image_path)
            features = self.extract_features(image)
            class_vectors.append(features)
        return np.mean(class_vectors, axis=0)


    def add_img(self, image_path, id_image):
        image = cv2.imread(image_path)
        cur_image = image
        feature = self.extract_features(cur_image)
        if id_image not in self.all_skus:
            self.all_skus[id_image] = []
        self.all_skus[id_image].append(feature)
        return feature

    def remove_by_id(self, id_image):
        if id_image in self.all_skus:
            self.all_skus.pop(id_image)

    def remove_all(self):
        self.all_skus.clear()

    def add_img_from_pickle(self, id_image, pickle_path):
        res = pickle.load(open(pickle_path, 'rb'))
        self.all_skus[id_image] = res

    def get_additional_info(self):
        json_res = {}
        json_res["Extract features, time"] = self.predict_time
        json_res["Find nearest, time"] = self.time_search
        json_res["Count frame"] = self.count_frame
        json_res["RPS"] = self.count_frame / (self.predict_time + self.time_search)
        return json_res

    def draw_label(self, frame0, x0, y0, w0, h0, label0, colour):
        """Draws a rectangle around an object and writes its classification label on the frame."""
        cv2.putText(frame0, label0, (x0 + 10, y0 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.7, colour, 2)
        return frame0
