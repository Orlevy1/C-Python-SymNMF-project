import numpy as np
import pandas as pd
import symnmfmodule as sym
import sys

np.random.seed(0)
##Goals dictionary for user input
goals = {
    'symnmf': sym.symnmf,
    'sym': sym.sym,
    'ddg': sym.ddg,
    'norm': sym.norm,
}

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


def main():
    user_input = sys.argv

    if len(user_input) != 4:
        print("Invalid Input!")
        exit()

    else:
        try:
            k = int(user_input[1])
        except:
            print("Invalid Input!")
            exit()

        goal = str(user_input[2])
        if goal not in goals:
            print("Invalid Input!")
            exit()

        input_file = str(user_input[3])
        try:
            X = read_input(input_file)
        except:
            print("An Error Has Occurred:")
            exit()


    if goal == 'symnmf':
        # Initialize H when goal is symnmf
        W = sym.norm(X)
        H = initialize_H(len(X), k,W)
        # Call your C extension function to perform symNMF and get the final H
        final_H =  goals['symnmf'](H,W)
        # Output final_H in the required format
        print_output(goal, final_H)

    elif goal == 'sym':
        # Call your C extension function to calculate the similarity matrix
        similarity_matrix = sym.sym(X)
        # Output similarity_matrix in the required format
        print_output(goal, similarity_matrix)

    elif goal == 'ddg':
        # Call your C extension function to calculate the diagonal degree matrix
        diagonal_degree_matrix = sym.ddg(X)
        # Output diagonal_degree_matrix in the required format
        print_output(goal, diagonal_degree_matrix)

    elif goal == 'norm':
        # Call your C extension function to calculate the normalized similarity matrix
        normalized_similarity_matrix = sym.norm(X)
        # Output normalized_similarity_matrix in the required format
        print_output(goal, normalized_similarity_matrix)

    else:
        print("Invalid Input!")
        exit()

if __name__ == "__main__":
    main()
