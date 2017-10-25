from __future__ import print_function

import numpy as np
import os
import subprocess

from scipy import sparse

from uncurl.sparse_utils import sparse_create_plda_file

PLDA_FOLDER = "/home/yjzhang/plda"
eps=1e-10

# Contains methods to process input and output files for PLDA.

# Generates an input file for PLDA from the matrix
# Assumes "matrix" is a Numpy array containing only integers, with dimensions (genes x cells)
def create_plda_file(matrix, filename):
    if sparse.issparse(matrix):
        sparse_create_plda_file(matrix, filename)
        return
    f = open(filename, "w")
    (r,c) = matrix.shape
    strings = []
    # PLDA input format requires one line per document (cell). Each line contains a sparse
    # representation of the counts of the words (genes) present. Example:
    # G1 12 G2 4 G5 6
    # (G1 appears 12 times, G2 appears 4 times, G5 appears 6 times)
    for i in range(c):
        for j in range(r):
            strings.append("G" + str(j) + " "  + str(int(matrix[j,i])) + " ")
        strings.append("\n")
    f.write("".join(strings))


# Parses the "model file" outputted by PLDA into a
# (word x topic) (or gene x archetype) matrix.
def parse_model_file(model_file, word_topic=None, included_genes=None):
    f = open(model_file, "r")
    lines = f.readlines()
    num_words = len(lines)  # There's 1 line for each word
    num_topics = len(lines[1].split()) - 1

    if word_topic is None:
        word_topic = np.zeros((num_words, num_topics))
    if included_genes is None:
        included_genes = np.arange(num_words)
    for line in lines:
        tokens = line.split()
        gene_number = int(tokens[0][1:])
        original_row_number = included_genes[gene_number]
        word_topic[original_row_number, :] = np.array(map(float, tokens[1:]))
    return word_topic


# Parse the "inference result" matrix outputed by PLDA into a
# (topic x document) (or archetype x cell) matrix.
def parse_result_file(result_file):
    document_topic = np.loadtxt(result_file, dtype="float")
    return document_topic.T



# Given a PLDA input file (each line is a "document", with each word followed by
# its count), return a corresponding data matrix.
def parse_plda_input(input_file, num_columns):
    f = open(input_file, "r")
    lines = f.readlines()
    num_lines = len(lines)
    matrix = np.zeros((num_lines, num_columns))
    row = 0

    for line in lines:
        tokens = line.split()
        i = 1
        while i < len(tokens):
            gene_number = int(tokens[i-1][1:])
            matrix[row, gene_number] = int(tokens[i])
            i += 2
        row += 1
    return matrix


# Given a PLDA input file, runs PLDA to find the M/W matrices.
# Note: please call "create_plda_file()" beforehand to create a PLDA input
# file from your matrix.
def plda_estimate_state(data, k, threads=4, num_iterations=150, plda_folder=None):
    if plda_folder is None:
        plda_folder = PLDA_FOLDER
    data_mean = np.array(data.mean(0)).flatten()
    try:
        os.mkdir('plda')
    except:
        pass
    filename = os.path.join(os.getcwd(), 'plda', 'data.txt')
    create_plda_file(data, filename)
    print("Training PLDA")
    train_args = ("mpiexec", "-n", str(threads), os.path.join(plda_folder, "mpi_lda"),
                  "--num_topics", str(k), "--alpha", "0.1",
                  "--beta", "0.01", "--training_data_file", filename,
                  "--model_file", "model.txt", "--burn_in_iterations", "100", "--total_iterations", str(num_iterations))
    subprocess.call(train_args) #, stdout=subprocess.PIPE)

    print("TRAINED")

    inference_args = (os.path.join(plda_folder, "infer"), "--alpha", "0.1", "--beta",
                      "0.01", "--inference_data_file", filename, "--inference_result_file",
                      "result.txt", "--model_file", "model.txt", "--total_iterations",
                      "50", "--burn_in_iterations", "20")
    subprocess.call(inference_args) #, stdout=subprocess.PIPE)

    M = parse_model_file("model.txt")
    W = parse_result_file("result.txt")
    M *= (data_mean / np.mean(M))
    W = W/W.sum(axis=0, keepdims=1)
    return M, W

