import csv
import cv2
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import random


def main():
    Question2Clustering()

def Question2Clustering():
    # Define k as number of clusters
    k = 2

    # Preprocessing
    # Data input
    fileName = "diabetes.csv"
    X_unstandard,Y = readCSVfile(fileName)

    # Standardizes features
    X, featureMeans, featureStdDevs = standardizeFeatures(X_unstandard)

    # Runs the k means algo
    myKMeans(X, Y, k)


def myKMeans(X, Y, k):
    colors = ["b", "g", "r", "c", "m", "y", "k"]
    threeDim = True

    # If 2 Dimensions
    if len(X[0]) == 2:
        threeDim = False

    # If 3 Dimensions
    elif len(X[0]) == 3:
        pass

    # If 3+ Dimensions, do PCA
    elif len(X[0]) > 3:
        X = myPCA(X)

    clustering(X, Y, k, colors, threeDim)


def clustering(X, Y, k, colors, threeDim):
    #seeding to 0
    random.seed(0)
    randomindices = random.sample(range(0, len(X)), k)

    # initializing random first reference vectors
    referenceVectors = X[randomindices]

    epi = 100
    iter = 0
    # begining the k clusters algo (need to add wrapping loop)
    title = "K_{}.avi".format(k)
    out = cv2.VideoWriter(title, cv2.VideoWriter_fourcc(*"DIVX"), 1, (640,480))

    # | --
    while epi >= 2**-23:
        iter +=1
        Xassoc = [-1]*len(X)

        # print(X)

        # figures out which refer vectors it belongs to
        for i in range(0, len(X)):
            minDist = 1000000000000
            for j in range(0, len(referenceVectors)):
                dist = euclDistance(X[i], referenceVectors[j])
                if dist < minDist:
                    minDist = dist
                    Xassoc[i] = j

        # print("ref",referenceVectors)
        avgPurity = calcPurity(Xassoc, Y, k)

        if threeDim:
            # plot in 3d
            fig = plt.figure()
            ax = plt.axes(projection='3d')
            ax.set_xlabel('X Label')
            ax.set_ylabel('Y Label')
            ax.set_zlabel('Z Label')

            # plot reference vector
            for v in range(0, len(referenceVectors)):
                ax.scatter3D(referenceVectors[v][0], referenceVectors[v][1], referenceVectors[v][2], c=colors[v], marker="o")

            # plot other points
            # print(Xassoc)
            for u in range(0, len(X)):
                ax.scatter3D(X[u][0], X[u][1], X[u][2], c=colors[Xassoc[u]], marker="x")

            plt.title('Iteration {} Purity = {}'.format(iter, avgPurity))
            fig.show()

        else:
            # plot in 2d
            fig = plt.figure()
            ax = plt.axes()
            ax.set_xlabel('X Label')
            ax.set_ylabel('Y Label')

            for v in range(0, len(referenceVectors)):
                ax.scatter(referenceVectors[v][0], referenceVectors[v][1], c=colors[v], marker="o")
            for u in range(0, len(X)):
                ax.scatter(X[u][0], X[u][1], c=colors[Xassoc[u]], marker="x")

            plt.title('Iteration {} Purity = {}'.format(iter, avgPurity))
            fig.show()

        newReferenceVectors = calcNewReferenceVectors(X, Xassoc,k)
        epi = calcEpi(referenceVectors, newReferenceVectors)
        referenceVectors = newReferenceVectors

        # converting to np array so i can save it with cv2 - from source code example
        data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))

        out.write(cv2.cvtColor(data, cv2.COLOR_RGB2BGR))
        # --- |

    # too do:
    # add title, (iteration and purity)
    # create loops wrapping loop that repeats this and makes new reference vectors


def euclDistance(currX,referenceVector):
    return np.linalg.norm(currX - referenceVector)

def calcEpi(referenceVectors,newReferenceVectors):
    sumDist = 0
    referenceVectors = np.asarray(referenceVectors).astype("float")
    newReferenceVectors = np.asarray(newReferenceVectors).astype("float")

    # print("referenceVectors: ",referenceVectors)
    # print("newReferenceVectors: ",newReferenceVectors)
    for i in range(0, len(referenceVectors)):
        #dist = pdist(referenceVectors[i], newReferenceVectors[i], metric='cityblock')
        dist = np.abs(referenceVectors[i] - newReferenceVectors[i]).sum()
        sumDist += dist
    # print("sumDist: ",sumDist)
    return sumDist


def myPCA(X):
    # does PCA for more than 3 variables
    covFaces = np.cov(np.transpose(X), ddof=1)
    U, S, V = np.linalg.svd(covFaces)

    eigenValues = S
    eigenVectors = V

    # uncomment to see eigenvalues
    # print("EigenValues:\n", eigenValues)
    # print("EigenVectors:\n", eigenVectors)

    # the three largest are just the first three since the function orders it

    eVec1 = np.array(eigenVectors[0])
    eVec2 = np.array(eigenVectors[1])
    eVec3 = np.array(eigenVectors[2])

    # uncomment to check norms
    # print(np.linalg.norm(eVec1))
    # print(np.linalg.norm(eVec2))

    W = np.array([eVec1, eVec2, eVec3]).transpose()

    Z = np.matmul(X, W)

    return Z


def calcPurity(Xassoc, Y, k):
    purity = []
    supervisedLabels = np.unique(Y)

    for i in range(0,k):  # loops through each cluster
        currCluster = i
        # print("currCluster: ", currCluster)

        clusterIndexs = np.where((np.asarray(Xassoc) == currCluster))[0]

        # print("clusterIndexes: ", clusterIndexs)

        Nij = []
        for j in range(0, len(supervisedLabels)):
            currSupervisedLabel = supervisedLabels[j]
            instancesOfJ = 0
            # print("current Supervised Label: ", currSupervisedLabel)

            for cIndex in clusterIndexs: # in current cluster, how many have label j
                if Y[cIndex] == currSupervisedLabel:
                    instancesOfJ += 1
            Nij.append(instancesOfJ)
            # print("Nij: ", Nij)

        purity.append(max(Nij)/len(clusterIndexs))
    return np.mean(purity)


def calcNewReferenceVectors(X , Xassoc, k):
    newReferenceVectors = []

    for i in range(0,k): # loops through each cluster
        currCluster = i
        clusterIndexs = np.where((np.asarray(Xassoc) == currCluster))[0]

        clusterVector = X[clusterIndexs]
        newVect = []

        for j in range(0,len(X[0])):
            newMean = np.mean(clusterVector.transpose()[j])

            newVect.append(newMean)

        newReferenceVectors.append(newVect)

    # print("newReferenceVectors", newReferenceVectors)
    return newReferenceVectors






def readCSVfile(name):
    with open(name, newline='') as csvfile:
        data = np.asarray(list(csv.reader(csvfile))).astype(np.float)


    X_unstandard = data[:, 1:]
    Y = data[:, 0]

    return X_unstandard, Y

def standardizeFeatures(X):
    # features are columns
    # array to standardize
    featureMeans = []
    featureStdDevs = []

    for i in range(len(X[0])):  # loops over each column
        currFeature = []

        # need to get the mean and std dev for each pixel
        for j in range(len(X)):  # loops over each observation
            currFeature.append(X[j][i])

        meancurrFeature = np.mean(currFeature)
        stdDevcurrFeature = np.std(currFeature, ddof=1)

        featureMeans.append(meancurrFeature)
        featureStdDevs.append(stdDevcurrFeature)

        # now we can standardize
        for k in range(len(X)):  # loops for each image
            X[k][i] = (X[k][i] - meancurrFeature) / stdDevcurrFeature

    return X, featureMeans, featureStdDevs


if __name__ == "__main__":
    main()
