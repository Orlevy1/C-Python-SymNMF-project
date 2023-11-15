import numpy as np
import pandas as pd
import symnmfmodule as sym
from sklearn.metrics import silhouette_score
import sys

np.random.seed(0)
def initialize_H(n, k,W):
    m = sum(sum(row) for row in W) / (len(W) * len(W[0]))
    # Calculate the range for random initialization
    lower_bound = 0
    upper_bound = 2 * np.sqrt(m / k)
    # Randomly initialize matrix H
    H = np.random.uniform(lower_bound, upper_bound, size=(n, k))
    H_res= H.tolist()
    return H_res

#Reads input file and returns a data matrix
def read_input(file_name):
    data = pd.read_csv(file_name, sep=',', header=None)
    return data.values.tolist()

# Function to print output in the specified format
def print_output(goal, result):
    if goal == 'symnmf':
        final_H = result
        for row in final_H:
            print(','.join("%.4f" % x for x in row))

    elif goal == 'sym' or goal == 'ddg' or goal == 'norm':
        for row in result:
            print(','.join("%.4f" % x for x in row))

##kmeans algo
def kmeans():
    epsilon=0.0001
    iter = 300
    K = sys.argv[1]
    file = str(sys.argv[2])
    N, numOfdims, data = init_file(file)

    cheaking(K,N)
    i=0
    iter = int(iter)
    K=int(K)
    centroids=[data[i] for i in range (K)]
    while (i<iter):
        clusters= [[] for i in range (K)]
        clusterAssignments = []  # New list to track cluster assignments
        new_Centroids = []
        for point in data:
            closest=cheak_centroids(point,centroids)
            clusters[closest].append(point)
            clusterAssignments.append(closest)  # Assign the cluster index to the data point

        for cluster in clusters:
            if cluster == []:
                new_Centroids.append([0.0 for i in range(numOfdims)])
            else:
                new_Centroids.append(calc_average(cluster,numOfdims))  # append the mean of the cluster to the new list of centroids

        for i in range(K):
            if distance(new_Centroids[i],centroids[i]) > epsilon:
                break
            if i == K - 1:
                return clusterAssignments,centroids
        centroids = new_Centroids
        i += 1
    return clusterAssignments,centroids


def distance(point1, point2):
    dist = 0
    i = 0
    while i < len(point1):
        dist += round((point1[i] - point2[i]) ** 2,4)
        i += 1
    return round(dist ** 0.5,4)

def init_file(inputdata):
    with open(inputdata, 'r') as f:
        data = []
        for line in f:
            row = line.strip().split(',')
            if row == ['']:
                continue
            row = [float(x) for x in row]
            data.append(row)
        numOfdims = len(data[0])
        numOfPoints = len(data)
    return (numOfPoints, numOfdims, data)

def cheaking(K,N):
    if int(K)>N or int(K)<=0 or not K.isnumeric():
        sys.exit("Invalid number of clusters!")


def cheak_centroids(point,centroids):
    distances=[]
    for i in range(len(centroids)):
        dis=distance(point,centroids[i])
        distances.append(dis)
        closest=distances.index(min(distances))
    return closest

def calc_average(cluster,dims):
    sum=[0 for i in range (dims)]
    size=len(cluster)
    for i in range(size):
        for j in range(dims):
            sum[j]+=round(cluster[i][j],4)
    for i in range(dims):
        sum[i]=round(sum[i]/size,4)

    return sum

def print_Kmeans(centroids):
    for vec in centroids:
        for j in range(len(vec)):
            if j == len(vec)-1:
                print(round(vec[j], 4), end = "\n")
            else:
                print(round(vec[j], 4), end = ",")


#end of kmeans

#analysis algo on H symnmf
def analysis_H_symnmf (H):
    H_np = np.array(H)
    num_elements, num_clusters = H_np.shape
    cluster_assignments = []
    for element_index in range(num_elements):
        cluster_index = np.argmax(H[element_index])
        cluster_assignments.append(cluster_index)
    return cluster_assignments


def main():
    user_input = sys.argv
    input_file = str(user_input[2])
    try:
        X = read_input(input_file)
    except:
        print("An Error Has Occured!")
        exit()
    try:
        k = int(user_input[1])
    except:
        print("Invalid Input!")
        exit()

    # Initialize H when goal is symnmf
    W = sym.norm(X)
    H = initialize_H(len(X), k, W)
    # Call C extension function to perform symNMF and get the final H
    final_H =  sym.symnmf(H,W)

    #get the output and get array of clusters for symnmf
    points = analysis_H_symnmf(final_H)
    average_silhouette_score_mat = silhouette_score(X,np.array(points))

    # get the output and get array of clusters for kmeans
    clusters, cent = kmeans()
    data_np = np.array(clusters)
    average_silhouette_score = silhouette_score(X, data_np)

    # Print the silhouette scores
    print("nmf:", np.round(average_silhouette_score_mat, 4))
    print("kmeans:", np.round(average_silhouette_score, 4))

if __name__ == "__main__":
    main()