import os
import sys
import argparse
import time
import itertools
import numpy as np
import pandas as pd


class KnnClassifier:
    def __init__(self, k: int, p: float):
        """
        Constructor for the KnnClassifier.

        :param k: Number of nearest neighbors to use.
        :param p: p parameter for Minkowski distance calculation.
        """
        self.k = k
        self.p = p

        # TODO - Place your student IDs here. Single submitters please use a tuple like so: self.ids = (123456789,)
        self.ids = (212699581, 211709597)

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        This method trains a k-NN classifier on a given training set X with label set y.

        :param X: A 2-dimensional numpy array of m rows and d columns. It is guaranteed that m >= 1 and d >= 1.
            Array datatype is guaranteed to be np.float32.
        :param y: A 1-dimensional numpy array of m rows. it is guaranteed to match X's rows in length (|m_x| == |m_y|).
            Array datatype is guaranteed to be np.uint8.
        """

        # TODO - your code here

        self.X = X
        self.Y = y

    

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        This method predicts the y labels of a given dataset X, based on a previous training of the model.
        It is mandatory to call KnnClassifier.fit before calling this method.

        :param X: A 2-dimensional numpy array of m rows and d columns. It is guaranteed that m >= 1 and d >= 1.
            Array datatype is guaranteed to be np.float32.
        :return: A 1-dimensional numpy array of m rows. Should be of datatype np.uint8.
        """

        # TODO - your code here


        data_matrix = np.matrix(X)
        train_matrix = np.matrix(self.X).T
        mulMatrix = (data_matrix * train_matrix)
        labelMatrix = np.repeat(self.Y, data_matrix.shape[0])

        mat = np.zeros((A.shape[0], A.shape[1], 2))
        mat[:, :, 0] = A
        # mat[:, :, 1] = 

        A = np.array(A)
        return list(map(self.maxOccur, A))

        # ret = np.zeros(len(X))
        
        # for i, e in enumerate(X):
        #     k_neig = sorted(list(zip(self.X[:self.k], self.Y[:self.k])),
        #                 key=lambda point: self.dis(point[0], e))

        #     for x, y in zip(self.X, self.Y):
        #         d1 = self.dis(x, e) # distance of current test point from current train point
        #         d2 = self.dis(k_neig[-1][0], e) # distance of current test point from furthest
        #         if d1 < d2 or (d1 == d2 and y < k_neig[-1][1]):
        #             k_neig[-1] = (x, y)
        #             k_neig = sorted(k_neig, key=lambda point: self.dis(point[0], e))

        #     labels = [e[1] for e in k_neig]

        #     if self.allEqual(labels):
        #         ret[i] = labels[0]
        #     else:
        #         ret[i] = max(labels, key=labels.count)

        # return ret



        ### Example code - don't use this:
        # return np.random.randint(low=0, high=2, size=len(X), dtype=np.uint8)

    def maxOccur(self, li):
        return max(list(li), key=list(li).count)
    
    def allEqual(self, li: list):
        t = li.count(li[0])
        for e in li:
            if li.count(e) != t:
                return False
            t = li.count(e)
        return True

    def dis(self, a: np.ndarray, b: np.ndarray):
        t = a - b
        return sum([e**self.p for e in t])**(1/self.p)

def main():

    print("*" * 20)
    print("Started HW1_ID1_ID2.py")
    # Parsing script arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('csv', type=str, help='Input csv file path')
    parser.add_argument('k', type=int, help='k parameter')
    parser.add_argument('p', type=float, help='p parameter')
    args = parser.parse_args()

    print("Processed input arguments:")
    print(f"csv = {args.csv}, k = {args.k}, p = {args.p}")

    print("Initiating KnnClassifier")
    model = KnnClassifier(k=args.k, p=args.p)
    print(f"Student IDs: {model.ids}")
    print(f"Loading data from {args.csv}...")
    data = pd.read_csv(args.csv, header=None)
    print(f"Loaded {data.shape[0]} rows and {data.shape[1]} columns")
    X = data[data.columns[:-1]].values.astype(np.float32)
    y = pd.factorize(data[data.columns[-1]])[0].astype(np.uint8)

    print("Fitting...")
    model.fit(X, y)
    print("Done")
    print("Predicting...")
    y_pred = model.predict(X)
    print("Done")
    accuracy = np.sum(y_pred == y) / len(y)
    print(f"Train accuracy: {accuracy * 100 :.2f}%")
    print("*" * 20)


if __name__ == "__main__":
    main()
