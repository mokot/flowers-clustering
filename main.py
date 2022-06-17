import sys
import glob
from matplotlib import pyplot as plt
import numpy as np
import torch
from torchvision import transforms
from PIL import Image
from scipy import spatial
import random

NUM_ITERATIONS = 100
K_MEDOIDS_MAX_ITERATIONS = np.inf


def cluster_display(data, cluster, clusters, path, index):
    """
    Display all images for specific group of images.
    """
    figure = plt.figure(figsize=(10, 5))
    row_num = int(np.ceil(len(cluster) / 5))

    plt.suptitle("Cluster: {:d}".format(index), fontsize=25)

    # calculate silhouettes
    dtype = np.dtype([("image_name", np.unicode_, 25), ("silhouette", np.float64)])
    silhouettes = np.sort(
        np.array([(el, silhouette(el, clusters, data)) for el in cluster], dtype=dtype),
        order=["silhouette", "image_name"],
    )[::-1]

    for index_i, (image_name, silhouette_temp) in enumerate(silhouettes):
        image = Image.open(path + "/" + str(image_name))
        ax = figure.add_subplot(row_num, 5, index_i + 1)

        # set image title
        ax.set_title("{:.2f}".format(silhouette_temp))

        # remove the axis labels
        ax.set_xticks([])
        ax.set_yticks([])

        # plot the image
        plt.imshow(image)

    # plt.show()  # show figure
    plt.savefig("cluster_{:d}.jpg".format(index))  # save figure

    return


def calculate_distance(data, medoids):
    """
    Calculate distance between all points and medoids.
    """
    # calculate distance to every medoid
    return [
        [cosine_dist(data[key], data[medoids[index]]) for index in range(len(medoids))]
        for key in data.keys()
    ]


def assign_cluster(data):
    """
    Assign each point to the closest medoid.
    """
    # assign cluster (minimize distance (cos(0) = 0))
    return np.argmin(data, axis=1)


def get_cluster(data, medoids):
    """
    Get cluster for each point.
    """
    clusters = [[] for _ in range(len(medoids))]
    data_keys = list(data.keys())
    clusters_index = assign_cluster(calculate_distance(data, medoids))
    for index in range(len(clusters_index)):
        clusters[clusters_index[index]].append(data_keys[index])

    return clusters


def read_data(path):
    """
    Read all images from path, convert them into embedding and return them as a 
    dictionary.

    docs: https://pytorch.org/hub/pytorch_vision_squeezenet/
    """

    data = {}
    dir_images = sorted(glob.glob(path + "/*"))

    model = torch.hub.load("pytorch/vision:v0.10.0", "squeezenet1_1", pretrained=True)
    model.eval()

    preprocess = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    for image_name in dir_images:
        input_image = Image.open(image_name)
        input_tensor = preprocess(input_image)
        input_batch = input_tensor.unsqueeze(0)

        # forward propagation
        with torch.no_grad():
            output = model(input_batch)

        # probabilities = torch.nn.functional.softmax(output[0], dim=0)
        data[image_name.split("/")[-1]] = output[0]

    return data


def cosine_dist(d1, d2):
    """
    Calculate cosine distance between two vectors.

    wiki: https://en.wikipedia.org/wiki/Cosine_similarity
    cos(fi) = (d1 @ d2) / (||d1|| * ||d2||)
    """

    return spatial.distance.cosine(d1, d2)
    # return 1 - np.dot(d1, d2) / (np.linalg.norm(d1) * np.linalg.norm(d2))
    # return 1 - np.dot(d1, d2) / (np.sqrt(np.sum(np.array(np.square(d1)))) *
    #                          np.sqrt(np.sum(np.array(np.square(d2)))))


def k_medoids(data, medoids):
    """
    Perform k-medoids clustering.

    wiki: https://en.wikipedia.org/wiki/K-medoids
    """

    # associate each data point to the closest medoid
    clusters = None

    iteration = 1
    # while the cost of the configuration decreases
    while iteration <= K_MEDOIDS_MAX_ITERATIONS:
        # print("Iteration: {:d}".format(iteration))
        iteration += 1

        medoids_old = np.copy(medoids)
        clusters = assign_cluster(calculate_distance(data, medoids))

        # for each medoid m, and for each non-medoid data point o
        for cluster_index in set(clusters):
            cluster_points_key = np.array(list(data.keys()))[clusters == cluster_index]
            cost = np.sum(
                [
                    cosine_dist(data[medoids[cluster_index]], data[cluster_temp_key])
                    for cluster_temp_key in cluster_points_key
                ]
            )

            for point_key in cluster_points_key:
                new_cost = np.sum(
                    [
                        cosine_dist(data[point_key], data[cluster_temp_key])
                        for cluster_temp_key in cluster_points_key
                    ]
                )

                # if the cost change is the current best, remember this m and o combination
                if new_cost < cost:
                    cost = new_cost
                    medoids[cluster_index] = point_key

        # perform the best swap of m_best and o_best, if it decreases the cost function
        # otherwise, the algorithm terminates
        if (medoids_old == medoids).all():
            break

    return get_cluster(data, medoids)


def silhouette(el, clusters, data):
    """
    Calculate silhouette for each point.

    wiki: https://en.wikipedia.org/wiki/Silhouette_(clustering)
    """
    cluster = [cluster_temp for cluster_temp in clusters if el in cluster_temp][0]
    cluster_other = [
        cluster_temp for cluster_temp in clusters if el not in cluster_temp
    ]

    # if cluster has one element, silhouette is equal to 0
    if len(cluster) == 1:
        return 0

    # cohesion
    ai = np.sum([cosine_dist(data[el], data[el_temp]) for el_temp in cluster]) / (
        len(cluster) - 1
    )

    # separation
    bi = np.min(
        [
            np.sum([cosine_dist(data[el], data[el_temp]) for el_temp in cluster_temp])
            / len(cluster_temp)
            for cluster_temp in cluster_other
        ]
    )

    # silhouette value (si)
    return (bi - ai) / np.max((ai, bi))


def silhouette_average(data, clusters):
    """
    Za podane podatke (slovar vektorjev) in skupine (seznam seznamov nizov:
    ključev v slovarju data) vrni povprečno silhueto.
    """
    return np.average([silhouette(el, clusters, data) for el in list(data.keys())])


if __name__ == "__main__":
    K = 5
    path = "./images"
    if len(sys.argv) == 3:
        K = sys.argv[1]
        path = sys.argv[2]

    data = read_data(path)
    medoids_best = -1
    clusters_best = -1
    silhouette_best = -np.inf

    random.seed(1)
    # repeat the procces for NUM_ITERATIONS times
    for iteration in range(NUM_ITERATIONS):
        # select K of the N (= 50) data points as the medoids
        medoids = random.sample(list(data.keys()), K)
        clusters = k_medoids(data, medoids)
        silhouette_avg = silhouette_average(data, clusters)

        if silhouette_avg > silhouette_best:
            silhouette_best = silhouette_avg
            medoids_best = medoids
            clusters_best = clusters

    # display clusters and silhouettes
    for index, cluster in enumerate(clusters_best):
        cluster_display(data, cluster, clusters_best, path, index + 1)
