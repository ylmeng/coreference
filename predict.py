import argparse
import time
import os
import multiprocessing
import sys
import numpy as np
import pickle
import shutil
import copy
from collections import defaultdict
from keras.models import load_model

from build_data import build_dataFrame, DataGen, group_data
import scipy.spatial.distance as distance
from scipy.cluster.hierarchy import linkage, inconsistent, fcluster

def clustering(pair_results, binarize=False, cutoff=1.7):
    def distance(e1, e2):
        e1 = tuple(e1.astype(int))
        e2 = tuple(e2.astype(int))
        if e1 == e2:
            return 1.0  # This is the minumum distance
        if (e1, e2) in pair_results:
            similarity = max(pair_results[(e1, e2)], 1e-3)
            # dist = 1 - pair_results[(e1, e2)] #+ 1e-4
            dist = min(1.0 / (similarity), 10.0)
            # dist = (10 * (1 - pair_results[(e1, e2)])) ** 2
        else:
            # dist = 0.9
            dist = 10.0
        if binarize:
            dist = np.round(dist)

        return dist

    # distance has no direction
    pairs = pair_results.keys()
    for key in pairs:
        pair_results[(key[1], key[0])] = pair_results[key]

    x = [key[0] for key in pair_results]
    x = list(set(x))
    x.sort(key=lambda x: x[0])
    x = np.array(x)

    clusters, Z = fclusterdata(x, cutoff, criterion='distance', metric=distance, depth=2, method='single')
    return x, clusters, Z


def fclusterdata(X, t, criterion='inconsistent',
                     metric='euclidean', depth=2, method='single', R=None):
    """
    This is adapted from scipy fclusterdata.
    https://github.com/scipy/scipy/blob/v1.0.0/scipy/cluster/hierarchy.py#L1809-L1878
    """
    X = np.asarray(X, order='c', dtype=np.double)

    if type(X) != np.ndarray or len(X.shape) != 2:
        print(type(X), X.shape)
        raise TypeError('The observation matrix X must be an n by m numpy '
                        'array.')

    Y = distance.pdist(X, metric=metric)
    Z = linkage(Y, method=method)
    if R is None:
        R = inconsistent(Z, d=depth)
    else:
        R = np.asarray(R, order='c')
    T = fcluster(Z, criterion=criterion, depth=depth, R=R, t=t)
    return T, Z


class TriadEvaluator(object):
    def __init__(self, model, test_input_gen, file_batch=10):
        self.model = model
        self.test_input_gen = test_input_gen
        self.data_q_store = multiprocessing.Queue(maxsize=5)

    def write_results(self, df, dest_path, n_iterations, cluster_cutoff=1.7, save_dendrograms=True, prune=True):
        """Perform evaluation on all test data, write results"""
        # assert self.data_available
        print("# files: %d" % n_iterations)

        # all_pairs_true = []
        all_pairs_pred = []
        processed_docs = set([])
        discarded = 0
        for i in range(n_iterations):
            # test_data_q = self.data_q_store.get()
            test_data_q = self.test_input_gen.next()
            assert len(test_data_q) == 1  # only process one file
            X, y, index_map = test_data_q[0]
            doc_id = index_map.keys()[0][0]
            if doc_id in processed_docs:
                print("%s already processed before!" % doc_id)
                continue
            processed_docs.add(doc_id)

            pred = []
            # true = []
            for X, y in group_data([X, y], 40, batch_size=None):
                pred.append(self.model.predict(X))
                # true.append(y)
            pred = np.concatenate(pred)
            pred = np.reshape(pred, [-1, 3])  # in case there are batches
            # true = np.concatenate(y)
            # true = np.reshape(true, [-1, 3])

            pair_results = defaultdict(list)
            # pair_true = {}
            for key in index_map:
                if sum(np.round(pred[index_map[key]])) == 2:  # skip illogical triads
                    discarded += 1
                pair_results[(key[1], key[2])].append(pred[index_map[key]][0])
                pair_results[(key[2], key[3])].append(pred[index_map[key]][1])
                pair_results[(key[3], key[1])].append(pred[index_map[key]][2])

                # pair_true[(key[1], key[2])] = true[index_map[key]][0]
                # pair_true[(key[2], key[3])] = true[index_map[key]][1]
                # pair_true[(key[3], key[1])] = true[index_map[key]][2]

            pair_results_mean = {}
            for key, value in pair_results.items():
                mean_value = np.mean(value)
                # mean_value = TriadEvaluator.median(value)
                pair_results_mean[key] = mean_value
                all_pairs_pred.append(mean_value)
                # all_pairs_true.append(pair_true[key])

            locs, clusters, linkage = clustering(pair_results_mean, cutoff=cluster_cutoff, binarize=False)
            # _, clusters_true, linkage_true = clustering(pair_true, binarize=False)

            if prune:
                clusters = TriadEvaluator.prune_clusters(clusters, threshold=1)
            if save_dendrograms:
                np.save(os.path.join(dest_path, 'linkages', doc_id.split('/')[-1]+'.npy'), linkage)
                # np.save(os.path.join(dest_path, 'true-linkages', doc_id.split('/')[-1] + '.npy'), linkage_true)

            doc_df = df.loc[df.doc_id == doc_id]
            length = len(doc_df)
            # print("Saving %s results..." % doc_id)
            sys.stdout.write("Saving results %d / %d\r" % (i + 1, n_iterations))
            sys.stdout.flush()
            corefs = ['-' for _ in range(length)]
            for loc, cluster in zip(locs, clusters):
                if cluster != 0:  # 0 represents the pruned clusters
                    start, end = loc
                    if corefs[start] == '-':
                        corefs[start] = '(' + str(cluster)
                    else:
                        corefs[start] += '|(' + str(cluster)

                    if corefs[end] == '-':
                        corefs[end] = str(cluster) + ')'
                    elif start == end:
                        corefs[end] += ')'
                    else:
                        corefs[end] += '|' + str(cluster) + ')'
            with open(os.path.join(dest_path, 'responses', doc_id.split('/')[-1]) + '.txt', 'w') as f:
                # f.write('#begin document (%s);\n' % doc_id)
                for ind, coref in enumerate(corefs):
                    word = doc_df.iloc[ind].word.decode('utf8')
                    pos = doc_df.iloc[ind].pos
                    f.write('\t'.join((doc_id, word.encode('utf8'), pos, coref)) + '\n')
                # f.write('\n#end document\n')

        print("Completed saving results!")
        # print("Pairwise evaluation:")
        # print("True histogram", np.histogram(all_pairs_true, bins=4))
        print("Prediction histogram", np.histogram(all_pairs_pred, bins=4))
        # print(classification_report(all_pairs_true, np.round(all_pairs_pred), digits=3))
        print("Discarded triads:", discarded)

    @staticmethod
    def top_n_mean(values, n):
        values.sort(reverse=True)
        if len(values) >= n:
            values = values[:n]
        return np.mean(values)

    @staticmethod
    def median(values):
        values.sort(reverse=True)
        return values[len(values)/2]

    @staticmethod
    def nonlinear_mean(values):
        return np.mean(np.round(values))

    @staticmethod
    def prune_clusters(clusters, threshold=1):
        sizes = defaultdict(int)
        for item in clusters:
            sizes[item] += 1
        remove_list = [x for x in sizes if sizes[x] <= threshold]

        new_clusters = copy.copy(clusters)
        for i, item in enumerate(clusters):
            if item in remove_list:
                new_clusters[i] = 0  # use 0 as dummy label
        return new_clusters


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("model_dir",
                        help="Directory containing the trained model")

    parser.add_argument("test_dir",
                        help="Directory containing test files")

    parser.add_argument("result_dir",
                        help="Directory to write result files")

    parser.add_argument("--triad",
                        action='store_true',
                        default=False,
                        help="use triads")

    parser.add_argument("--prune",
                        action='store_true',
                        default=False,
                        help="prune small clusters")

    parser.add_argument("--cutoff",
                        type=float,
                        default=1.7,
                        help="cutoff threashold for clustering")

    args = parser.parse_args()


    with open(os.path.join(args.model_dir, 'word_indexes.pkl')) as f:
        word_indexes = pickle.load(f)
    with open(os.path.join(args.model_dir, 'pos_tags.pkl')) as f:
        pos_tags = pickle.load(f)

    df = build_dataFrame(args.test_dir, threads=1, suffix='txt')
    model = load_model(os.path.join(args.model_dir, 'model.h5'))
    print("Loaded model")

    n_files = len(df.doc_id.unique())
    test_gen = DataGen(df, word_indexes, pos_tags)
    test_input_gen = test_gen.generate_triad_input(looping=True, test_data=True, threads=4)
    evaluator = TriadEvaluator(model, test_input_gen)
    evaluator.write_results(df, args.result_dir, cluster_cutoff=args.cutoff, n_iterations=n_files, prune=args.prune)


if __name__ == "__main__":
    main()