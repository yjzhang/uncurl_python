import numpy as np
import os
import errno
import subprocess
from scipy import sparse
from uncurl.sparse_utils import sparse_create_libsvm_file

LIGHTLDA_FOLDER = "/home/yjzhang/lightlda-warm-start"
eps = 1e-10

# Contains methods to process input and output files for LightLDA.


# Writes the given matrix to a file in libsvm (sparse matrix) format.
# Assumes "matrix" is a Numpy array containing only integers, with dimensions (genes x cells)
def create_libsvm_file(matrix, filename):
    f = open(filename, "w")
    (r,c) = matrix.shape
    print("Create_libsvm_file")
    print(filename)
    # NOTE: I assume the "label" attribute of the LibSVM file isn't used
    # (LightLDA is unsupervised), so all labels are set to the same class.
    # This might be an incorrect assumption!!!

    # Each line represents a document (cell), and contans a sparse
    # representation of the counts of the words (genes) present. Example:
    # 1 0:12 2:4 5:6
    # (document is class 1. Gene 0 appears 12 times, G2 appears 4 times, G5 appears 6 times)
    strings = []
    for i in range(c):
        strings.append("1\t")
        for j in range(r):
            if matrix[j,i] != 0:
                strings.append(str(j))
                strings.append(":")
                strings.append(str(int(matrix[j,i])))
                strings.append(" ")
        strings.append("\n")
    f.write("".join(strings))

# Writes the given matrix to a file in libsvm (sparse matrix) format.
# Assumes "matrix" is a Numpy array containing only integers or floats.
# Dimensions are (cell x archetype) or (gene x archetype)
def create_model_file(filename, matrix):
    f = open(filename, "w")
    (r,c) = matrix.shape

    # NOTE: I assume the "label" attribute of the LibSVM file isn't used
    # (LightLDA is unsupervised), so all labels are set to the same class.
    # This might be an incorrect assumption!!!

    # Each line represents a document (cell), and contans a sparse
    # representation of the topics (archetypes) present.. Example:
    # 1 0:12 2:4 5:6
    # (cell is class 1. archetype 0 has value 12, archetype 2 has value 4,
    # archetype 5 has value 6
    strings = []
    for i in range(r):
        strings.append(str(i) + " ")
        for j in range(c):
            if matrix[i, j] >= 0.0:
                strings.append(str(j))
                strings.append(":")
                strings.append(str(matrix[i, j]))
                strings.append(" ")
        strings.append("\n")
    f.write("".join(strings))




# Create a dictionary file relating each "index" to its "word" (in this case, simply
# G0, G1, etc.) I don't think this is used.
def create_dict_file(num_genes, filename):
    f = open(filename, "w")
    for i in range(num_genes):
        f.write(str(i) + "\tG" + str(i) + "\t" + str(i+1) + "\n")
        


# Parses the "model file" outputted by LightLDA into a
# (word x topic) (or gene x archetype) matrix.
def parse_model_file(model_file, num_topics, num_words):
    f = open(model_file, "r")
    lines = f.readlines()
    word_topic = np.zeros((num_words, num_topics))

    for line in lines:
        tokens = line.split()
        word_id = tokens[0]
        for i in range(1, len(tokens)):
            topic_id, count = tokens[i].split(":")
            word_topic[int(word_id), int(topic_id)] = float(count)
    return word_topic


# Parse the "inference result" matrix outputed by LightLDA into a
# (topic x document) (or archetype x cell) matrix.
def parse_result_file(result_file, num_topics):
    f = open(result_file, "r")
    lines = f.readlines()
    num_docs = len(lines)
    matrix = np.zeros((num_topics, num_docs))
    for line in lines:
        tokens = line.split()
        doc_id = tokens[0]
        for i in range(1, len(tokens)):
            topic_id, count = tokens[i].split(":")
            matrix[int(topic_id), int(doc_id)] = float(count)
    return matrix



def poisson_objective(X, m, w):
    """
    Creates an objective function and its derivative for M, given W and X
    Args:
        w (array): clusters x cells
        X (array): genes x cells
        selected_genes (array): array of ints - genes to be selected
    """
    clusters, cells = w.shape
    genes = X.shape[0]
    #m = m.reshape((X.shape[0], w.shape[0]))
    d = m.dot(w)+eps
    #temp = X/d
    #w_sum = w.sum(1)
    #w2 = w.dot(temp.T)
    #deriv = w_sum - w2.T
    return np.sum(d - X*np.log(d))/genes #, deriv.flatten()/genes


def lightlda_estimate_state(data, k, input_folder="data1/LightLDA_input", threads=8, max_iters=250, prepare_data=True, init_means=None, init_weights=None, lightlda_folder=None, data_capacity=1000):
    """
    Runs LDA on the given dataset (can be an 2-D array of any form - sparse
    or dense, as long as it can be indexed). If the data has not already been
    prepared into LDA format, set "prepare_data" to TRUE. If "prepare_data" is
    FALSE, the method assumes that the data has already been preprocessed into
    LightLDA format and is located at the given "input_folder".
    """
    if lightlda_folder is None:
        lightlda_folder = LIGHTLDA_FOLDER
    if prepare_data:
        prepare_lightlda_data(data, input_folder, lightlda_folder)

    # Check if initializations for M/W were provided.
    if ((init_means is not None) and (init_weights is None)) or ((init_means is None) and (init_weights is not None)):
        raise ValueError("LightLDA requires that either both M and W be initialized, or neither. You initialized one but not the other.")
    
    warm_start = False

    # If we have initial M/W matrices, write to the model and doc-topic files
    if (init_means is not None) and (init_weights is not None):
        warm_start = True
        init_means = init_means/init_means.sum(0)
        init_weights = init_weights/init_weights.sum(0)
        create_model_file("server_0_table_0.model", init_means)
        create_model_file("doc_topic.0", init_weights.T)
        print(init_means)
        print("init_means")

    # Run LightLDA
    print("TRAINING")
    # TODO: argument for data capacity
    train_args = (os.path.join(lightlda_folder, "bin/lightlda"), "-num_vocabs", str(data.shape[0]), "-num_topics",
                  str(k), "-num_iterations", str(max_iters), "-alpha", "0.05", "-beta", "0.01", "-mh_steps", "2",
                  "-num_local_workers", str(threads), "-num_blocks", "1", "-max_num_document", str(data.shape[1]),
                  "-input_dir", input_folder, "-data_capacity", str(data_capacity))
    if warm_start:
        print("warm start")
        train_args = train_args + ("-warm_start",)

    # Call LightLDA
    subprocess.call(train_args)
    
    # Parse final model and doc-topic files to obtain M/W
    print("data shape")
    print(data.shape)

    M = parse_model_file("server_0_table_0.model", k, data.shape[0])
    W = parse_result_file("doc_topic.0", k)
    
    # Not sure if normalization is correct
    M = M * (np.mean(data) / np.mean(M))
    W = W/W.sum(0)
    print("shapes")
    print(M.shape)
    print(W.shape)
    # TODO: poisson_objective doesn't work for sparse matrices
    if sparse.issparse(data):
        ll = 0
    else:
        ll = poisson_objective(data, M, W)
    #M = M * (5./np.mean(M))
    return M, W, ll


# Converts matrix to LightLDA format and dumps it into the input folder.
def prepare_lightlda_data(data, input_folder, lightlda_folder):
    print("Preparing LightLDA data")

    # Create the input directory if it doesn't exist
    try:
        os.makedirs(input_folder)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise

    libsvm_file = os.path.join(input_folder, "input.libsvm")
    print('create libsvm file')
    sparse_create_libsvm_file(data, libsvm_file)
    print("libsvm file created")
 
    # Produce metadata file
    metadata_args = (os.path.join(lightlda_folder, "example/get_meta.py"),
                     libsvm_file,
                     os.path.join(input_folder, "genes.word_id.dict"))
    subprocess.call(metadata_args)

    #create_dict_file(num_words, os.path.join(LIGHTLDA_FOLDER, "input/word.dict"))

    # Convert the libsvm file into LightLDA input
    pid = str(0)
    convert_args = (os.path.join(lightlda_folder, "bin/dump_binary"),
                    libsvm_file,
                    os.path.join(input_folder, "genes.word_id.dict"),
                    input_folder, pid)
    subprocess.call(convert_args)

    print("Done preparing LightLDA data")

