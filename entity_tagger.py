import numpy as np
import re
import os
from build_data import build_dataFrame, get_entities
from word2vec import build_vocab
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from keras.preprocessing.sequence import pad_sequences
import nltk
import pickle
import glob

EMBEDDING_DIM = 300
# LABELS = ['-', '(', ')', '()']
LABELS = ['O', 'B', 'I', '_START_TAG_', '_END_TAG_', '_PAD_']

class DataGen(object):
    def __init__(self, path, word_indexes={}, pos_tags=[]):
        self.df = build_dataFrame(path, threads=4, suffix='gold_conll')
        self.pos_tag_indexes = {}
        if not word_indexes:
            self.get_embedding_matrix()
        else:
            self.word_indexes = word_indexes
        if not pos_tags:
            self.get_pos_tags()
        else:
            self.pos_tags = pos_tags

    def generate_sentences(self, looping=True):
        doc_ids = self.df.doc_id.unique()
        prefix = re.compile('\(\d+')
        suffix = re.compile('\d+\)')
        while True:
            np.random.shuffle(doc_ids)
            for doc_id in doc_ids:
                doc_df = self.df.loc[self.df.doc_id == doc_id]
                entity_started = False
                counts = 0
                # word_list = ['_S_START_']  # start of a sentence
                # pos_list = ['_S_START_']
                # label_list = ['_START_TAG_']
                word_list = []
                pos_list = []
                label_list = []
                for index, row in doc_df.iterrows():
                    word = row.word
                    pos = row.pos
                    coref = row.coref
                    starts = prefix.findall(coref)
                    ends = suffix.findall(coref)
                    counts += len(starts)
                    counts -= len(ends)

                    # if not starts and not ends:
                    #     label = 0
                    # elif starts and not entity_started and counts > 0:  # only count biggest entities
                    #     label = 1
                    #     entity_started = True
                    # elif ends and entity_started and counts == 0:
                    #     label = 2
                    #     entity_started = False
                    # elif counts == 0:
                    #     label = 3
                    # else:
                    #     label = 0

                    if starts and not entity_started:
                        label = 1  # B
                        entity_started = True
                        if counts == 0:
                            entity_started = False # (0)
                    elif ends:
                        label = 2  # I
                        if counts == 0:
                            entity_started = False
                    elif counts > 0:
                        label = 2
                    elif counts == 0:
                        label = 0
                        entity_started = False
                    else:
                        raise ValueError("Wrong label")

                    word_list.append(word)
                    pos_list.append(pos)
                    label_list.append(label)

                    if pos == '.':
                        # word_list.append('_S_END_')
                        # pos_list.append('_S_END_')
                        # label_list.append('_END_TAG_')
                        yield word_list, pos_list, label_list

                        # word_list = ['_S_START_']  # start of a sentence
                        # pos_list = ['_S_START_']
                        # label_list = ['_START_TAG_']
                        word_list = []
                        pos_list = []
                        label_list = []
            if not looping:
                break

    def generate_input(self, batch=32):
        counter = 0
        X0 = []
        X1 = []
        y = []
        for word_list, pos_list, label_list in self.generate_sentences(): # one sentence each time
            X0.append(self.get_word_indexes(word_list))
            X1.append(self.get_pos_indexes(pos_list))
            y.append(label_list)
            counter += 1
            if counter % batch == 0:
                X0 = pad_sequences(X0, maxlen=None, dtype='int32', padding='post', truncating='post', value=0)
                X1 = pad_sequences(X1, maxlen=None, dtype='int32', padding='post', truncating='post', value=0)
                y = pad_sequences(y, maxlen=None, dtype='int32', padding='post', truncating='post', value=LABELS.index('_PAD_'))
                yield [X0, X1], y
                X0 = []
                X1 = []
                y = []

    def get_word_indexes(self, word_list):
        return [self.word_indexes[word] if word in self.word_indexes else self.word_indexes['UKN'] for word in word_list]

    def get_pos_indexes(self, pos_list):
        return [self.pos_tags.index(pos) + 1 if pos in self.pos_tags else self.pos_tags.index('UKN') for pos in pos_list]

    def get_embedding_matrix(self, word_vectors=None):
        if word_vectors is None:
            print('Loading word embeddings...')
            glove_path = 'glove.840B.300d.txt'
            word_vectors = build_vocab(self.df.word.unique(), glove_path, K=200000)
            word_vectors['_S_START_'] = np.ones(EMBEDDING_DIM)
            word_vectors['_S_END_'] = - np.ones(EMBEDDING_DIM)
            word_vectors['UKN'] = np.random.uniform(-0.5, 0.5, EMBEDDING_DIM)

        word_indexes = {}
        embedding_matrix = np.random.uniform(low=-0.5, high=0.5, size=(len(word_vectors) + 1, EMBEDDING_DIM))
        for index, word in enumerate(sorted(word_vectors.keys())):
            word_indexes[word] = index + 1  # starts with 1
            embedding_vector = word_vectors.get(word, None)
            embedding_matrix[index + 1] = embedding_vector
        embedding_matrix[0] = np.zeros(EMBEDDING_DIM)  # used for mask/padding

        self.embedding_matrix = embedding_matrix
        self.word_indexes = word_indexes

    def get_pos_tags(self):
        all_pos_tags = self.df.pos.unique()
        all_pos_tags.sort()
        print("%d pos tags found" % len(all_pos_tags))
        print(all_pos_tags)
        self.pos_tags = np.append(all_pos_tags, ['_START_POS_', '_END_POS_', 'UKN']).tolist()
        for i, tag in enumerate(self.pos_tags):
            self.pos_tag_indexes[tag] = i + 1  # starts with 1


### copied from http://pytorch.org/tutorials/beginner/nlp/advanced_tutorial.html ###
def to_scalar(var):
    # returns a python float
    return var.view(-1).data.tolist()[0]


def argmax(vec):
    # return the argmax as a python int
    _, idx = torch.max(vec, 1)
    return to_scalar(idx)


# def prepare_sequence(seq, to_ix):
#     idxs = [to_ix[w] for w in seq]
#     tensor = torch.LongTensor(idxs)
#     return autograd.Variable(tensor)

def get_variable(data, dtype=float):
    if dtype == int:
        if type(data) == list:
            return [autograd.Variable(torch.LongTensor(x).cuda()) for x in data]
        else:
            return autograd.Variable(torch.LongTensor(data).cuda())

    if type(data) == list:
        return [autograd.Variable(torch.Tensor(x).cuda()) for x in data]
    return autograd.Variable(torch.Tensor(data).cuda())

# Compute log sum exp in a numerically stable way for the forward algorithm
# it is a smooth approximation of maximum
def log_sum_exp(vec):
    max_score = vec[0, argmax(vec)]
    max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
    return max_score + \
        torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))

class BiLSTM_CRF(nn.Module):

    def __init__(self, vocab_size, pos_size, word_embeddings=None):
        super(BiLSTM_CRF, self).__init__()
        # self.hidden_dim = 128
        self.vocab_size = vocab_size
        self.pos_size = pos_size
        self.tagset_size = len(LABELS)
        self.mask = None

        self.word_embeds = nn.Embedding(self.vocab_size, EMBEDDING_DIM)
        if word_embeddings is not None:
            self.word_embeds.weight = nn.Parameter(torch.from_numpy(word_embeddings).type(torch.cuda.FloatTensor))
        print("word embedding size:", self.word_embeds.weight.size())
        self.word_lstm = nn.LSTM(EMBEDDING_DIM, 128, num_layers=1, batch_first=True, bidirectional=True)
        self.word_hidden = nn.Linear(256, 128)

        self.pos_embeds = nn.Embedding(self.pos_size + 1, self.pos_size + 1)
        self.pos_embeds.weight = nn.Parameter(torch.eye(self.pos_size + 1).type(torch.cuda.FloatTensor))
        self.pos_lstm = nn.LSTM(self.pos_size + 1, 8, num_layers=1, batch_first=True, bidirectional=True)
        self.pos_hidden = nn.Linear(16, 8)

        # Maps the output of the LSTM into tag space.
        self.hidden2tag = nn.Linear(136, self.tagset_size)

        # Matrix of transition parameters.  Entry i,j is the score of
        # transitioning *to* i *from* j.
        self.transitions = nn.Parameter(torch.zeros(self.tagset_size, self.tagset_size).cuda())

        # These two statements enforce the constraint that we never transfer
        # to the start tag and we never transfer from the stop tag
        # LABELS = ['O', 'B', 'I', '_START_TAG_', '_END_TAG_', '_PAD_']
        self.transitions.data[LABELS.index('_START_TAG_'), :] = -100
        self.transitions.data[LABELS.index('I'), LABELS.index('O')] = -100
        self.transitions.data[LABELS.index('I'), LABELS.index('_START_TAG_')] = -100
        self.transitions.data[LABELS.index('B'), LABELS.index('_START_TAG_')] = 0.5
        self.transitions.data[LABELS.index('O'), LABELS.index('_START_TAG_')] = 0.5
        self.transitions.data[:, LABELS.index('_END_TAG_')] = -100

        # self.hidden = self.init_hidden()

    def init_hidden(self):
        return (autograd.Variable(torch.randn(2, 1, self.hidden_dim // 2)),
                autograd.Variable(torch.randn(2, 1, self.hidden_dim // 2)))

    def _forward_alg(self, feats_batch):
        transitions = F.softmax(self.transitions.permute(1, 0)).permute(1, 0)
        transitions.data[:, LABELS.index('_END_TAG_')] = -100

        alphas = []
        sentence_lens = self.mask.sum(dim=1).data.cpu().numpy()
        # print(sentence_lens)
        for batch_index, feats in enumerate(feats_batch):  # over batches
            max_len = sentence_lens[batch_index]
            # Do the forward algorithm to compute the partition function
            init_alphas = torch.cuda.FloatTensor(1, self.tagset_size).fill_(-100.)
            # START_TAG has all of the score.
            init_alphas[0][LABELS.index('_START_TAG_')] = 0.

            # Wrap in a variable so that we will get automatic backprop
            forward_var = autograd.Variable(init_alphas)

            # Iterate through the sentence
            for feat_index, feat in enumerate(feats):
                if feat_index == max_len:
                    break
                alphas_t = []  # The forward variables at this timestep
                for next_tag in range(self.tagset_size):
                    # broadcast the emission score: it is the same regardless of
                    # the previous tag
                    emit_score = feat[next_tag].view(1, -1).expand(1, self.tagset_size)  # fetch one score and make a 1x5 tensor

                    # the ith entry of trans_score is the score of transitioning to
                    # next_tag from i
                    trans_score = transitions[next_tag].contiguous().view(1, -1)
                    # The ith entry of next_tag_var is the value for the
                    # edge (i -> next_tag) before we do log-sum-exp

                    next_tag_var = forward_var + trans_score + emit_score

                    # The forward variable for this tag is log-sum-exp of all the
                    # scores.
                    alphas_t.append(log_sum_exp(next_tag_var))
                forward_var = torch.cat(alphas_t).view(1, -1)
            terminal_var = forward_var + self.transitions[LABELS.index('_END_TAG_')]
            alphas.append(log_sum_exp(terminal_var))
        alphas = torch.cat(alphas, 0)
        # print(alphas)
        # print(alphas.requires_grad)
        return alphas

    def _get_lstm_features(self, words, pos_tags, mask_value=0):
        # self.hidden = self.init_hidden()
        self.mask = (words != mask_value)
        word_embeds = self.word_embeds(words)
        word_embeds = F.dropout(word_embeds, p=0.5)
        word_lstm_out, _ = self.word_lstm(word_embeds)
        word_hidden_out = F.relu(self.word_hidden(word_lstm_out))  # pytorch lstm output is a sequence be default

        pos_embeds = self.pos_embeds(pos_tags)
        pos_embeds = F.dropout(pos_embeds, p=0.3)
        pos_lstm_out, _ = self.pos_lstm(pos_embeds)
        pos_hidden_out = F.relu(self.pos_hidden(pos_lstm_out))

        # print("sizes", word_hidden_out.size(), pos_hidden_out.size())
        cat_out = torch.cat((word_hidden_out, pos_hidden_out), -1)
        cat_out = F.dropout(cat_out, p=0.5)
        lstm_feats = self.hidden2tag(cat_out)  # (batch, seq, categories)
        return F.relu(lstm_feats)

        # # change shape to (categories, batch, seq) and perform softmax
        # lstm_feats = F.softmax(lstm_feats.permute(2, 1, 0))  # softmax works on the first dimension only
        #
        # return lstm_feats.permute(2, 1, 0)

    def _score_sentence(self, feats_batch, tags_batch):
        transitions = F.softmax(self.transitions.permute(1, 0)).permute(1, 0)
        transitions.data[:, LABELS.index('_END_TAG_')] = -100
        scores = []
        for feats, tags, in zip(feats_batch, tags_batch):
            # Gives the score of a provided tag sequence
            score = autograd.Variable(torch.cuda.FloatTensor([0]))
            tags = torch.cat((torch.cuda.LongTensor([LABELS.index('_START_TAG_')]), tags))
            for i, feat in enumerate(feats):
                if tags[i + 1] == LABELS.index('_PAD_'):
                    break
                score = score + transitions.contiguous()[tags[i + 1], tags[i]] + feat[tags[i + 1]]
            score = score + self.transitions[LABELS.index('_END_TAG_'), tags[-1]]
            scores.append(score)
        scores = torch.cat(scores, 0)
        return scores

    def _viterbi_decode(self, feats, max_len=100):
        feats = autograd.Variable(feats.data, requires_grad=False).cpu()
        transitions = F.softmax(self.transitions.permute(1, 0)).permute(1, 0)
        transitions = transitions.cpu()
        transitions.data[:, LABELS.index('_END_TAG_')] = -100
        transitions.data[LABELS.index('I'), LABELS.index('O')] = -100
        backpointers = []

        # Initialize the viterbi variables in log space
        # init_vvars = torch.cuda.FloatTensor(1, self.tagset_size).fill_(-100.)
        init_vvars = torch.FloatTensor(1, self.tagset_size).fill_(-100.)
        init_vvars[0][LABELS.index('_START_TAG_')] = 0

        # forward_var at step i holds the viterbi variables for step i-1
        forward_var = autograd.Variable(init_vvars, requires_grad=False)
        for i, feat in enumerate(feats):
            if i == max_len:
                break
            bptrs_t = []  # holds the backpointers for this step
            viterbivars_t = []  # holds the viterbi variables for this step

            for next_tag in range(self.tagset_size):
                # next_tag_var[i] holds the viterbi variable for tag i at the
                # previous step, plus the score of transitioning
                # from tag i to next_tag.
                # We don't include the emission scores here because the max
                # does not depend on them (we add them in below)
                # print(forward_var.size(), self.transitions[next_tag].size())

                next_tag_var = forward_var + transitions.contiguous()[next_tag]

                best_tag_id = argmax(next_tag_var)
                bptrs_t.append(best_tag_id)
                viterbivars_t.append(next_tag_var[0][best_tag_id])
            # Now add in the emission scores, and assign forward_var to the set
            # of viterbi variables we just computed
            forward_var = (torch.cat(viterbivars_t) + feat).view(1, -1)
            backpointers.append(bptrs_t)

        # Transition to STOP_TAG
        terminal_var = forward_var + transitions[LABELS.index('_END_TAG_')]
        best_tag_id = argmax(terminal_var)
        path_score = terminal_var[0][best_tag_id]

        # Follow the back pointers to decode the best path.
        best_path = [best_tag_id]
        for bptrs_t in reversed(backpointers):
            best_tag_id = bptrs_t[best_tag_id]
            best_path.append(best_tag_id)
        # Pop off the start tag (we dont want to return that to the caller)
        start = best_path.pop()
        assert start == LABELS.index('_START_TAG_')  # Sanity check
        best_path.reverse()
        return path_score, best_path

    def neg_log_likelihood(self, input, labels):
        words, pos_tags = input
        feats_batch = self._get_lstm_features(words, pos_tags)
        # print("feats", feats_batch)
        forward_scores = self._forward_alg(feats_batch)
        gold_scores = self._score_sentence(feats_batch, labels)
        loss = torch.mean(forward_scores - gold_scores)
        return loss

    def forward(self, x):  # dont confuse this with _forward_alg above.
        words, pos_tags = x
        # Get the emission scores from the BiLSTM
        lstm_feats_batch = self._get_lstm_features(words, pos_tags)  # (batch, seq, categories)
        # print("lstm_feats example", lstm_feats_batch[0])

        # Find the best path, given the features.
        scores = []
        tag_seqs = []
        sentence_lens = self.mask.sum(dim=1).data.cpu().numpy()
        # print(sentence_lens)
        for lstm_feats, length in zip(lstm_feats_batch, sentence_lens):  # loop over batches
            score, tag_seq = self._viterbi_decode(lstm_feats, max_len=length)
            scores.append(score)
            tag_seqs.append(tag_seq)
        return scores, tag_seqs

def test_model():
    """Test the model"""
    # Make up some training data

    sentences = ["The wall street journal reported today that Apple Corporation made money.",
                 "Georgia Tech is a university in Georgia."]

    word_input = []
    postag_input = []
    all_words = set([])
    all_postags = set([])
    for sentence in sentences:
        tokens = nltk.word_tokenize(sentence)
        tagged_tokens = nltk.pos_tag(tokens)
        pos_tags = [s[1] for s in tagged_tokens]
        word_input.append(tokens)
        postag_input.append(pos_tags)

        all_words.update(tokens)
        all_postags.update(pos_tags)

    labels = ["B I I I O O O B I O O O".split(), "B I O O O O B O".split()]

    # glove_path = os.environ["TEA_PATH"] + 'embeddings/glove.840B.300d.txt'
    # word_vectors = build_vocab(all_words, glove_path, K=1000)

    word_indexes = {}
    for word in all_words:
        word_indexes[word] = len(word_indexes)
    pos_indexes = {}
    for pos in all_postags:
        pos_indexes[pos] = len(pos_indexes)

    model = BiLSTM_CRF(len(word_indexes), len(pos_indexes))
    if torch.cuda.is_available():
        model = model.cuda()
    optimizer = optim.SGD(model.parameters(), lr=0.01, weight_decay=1e-4)
    # for p in model.parameters():
    #     print("model parameters:", p.data.size())

    X0 = []
    X1 = []
    y = []
    for i in range(len(labels)):
        X0.append([word_indexes[w] for w in word_input[i]])
        X1.append([pos_indexes[t] for t in postag_input[i]])
        y.append([LABELS.index(s) for s in labels[i]])
    X0 = pad_sequences(X0, maxlen=None, dtype='int32', padding='post', truncating='post', value=0)
    X1 = pad_sequences(X1, maxlen=None, dtype='int32', padding='post', truncating='post', value=0)
    y = pad_sequences(y, maxlen=None, dtype='int32', padding='post', truncating='post', value=LABELS.index('_PAD_'))
    # print(X0.shape, X1.shape, y.shape)

    # Check predictions before training
    input_data = get_variable([X0.tolist(), X1.tolist()], dtype=int)
    # input_data = [autograd.Variable(torch.from_numpy(X0).type(torch.cuda.LongTensor)), autograd.Variable(torch.from_numpy(X1).type(torch.cuda.LongTensor))]
    print(input_data[0].size(), input_data[1].size())
    print("Initial predictions", model(input_data))
    print("Initial transition matrix", F.softmax(model.transitions.permute(1, 0)).permute(1, 0))

    # Make sure prepare_sequence from earlier in the LSTM section is loaded
    for epoch in range(100):  # again, normally you would NOT do 300 epochs, it is toy data
        
        # Step 1. Remember that Pytorch accumulates gradients.
        # We need to clear them out before each instance
        model.zero_grad()

        # Step 2. Get our inputs ready for the network, that is,
        # turn them into Variables of word indices.
        targets = torch.from_numpy(y).type(torch.cuda.LongTensor)

        # Step 3. Run our forward pass.
        neg_log_likelihood = model.neg_log_likelihood(input_data, targets)

        # Step 4. Compute the loss, gradients, and update the parameters by
        # calling optimizer.step()
        neg_log_likelihood.backward()
        # print("model.transitions.grad", model.transitions.grad)
        optimizer.step()
        # print(model(input_data)[1])

    print("Final predictions", model(input_data))
    print("Final transition matrix", F.softmax(model.transitions.permute(1, 0)).permute(1, 0))


def train():
    epochs = 100
    train_path = '/home/ymeng/projects/conll2012/conll-2012/v4/data/train/data/english/'
    # train_path = '/home/ymeng/projects/conll2012/conll-2012/v4/data/train/data/english/annotations/bc/'
    data_gen = DataGen(train_path)
    pickle.dump(data_gen.word_indexes, open('models/word_indexes.pkl', 'wb'))
    pickle.dump(data_gen.pos_tag_indexes, open('models/pos_tag_indexes.pkl', 'wb'))

    test_file = 'original_files/blog_01.txt'

    # test data
    X0_test, X1_test, tokens_test, postags_test = prepare_test_data(test_file, data_gen.word_indexes, data_gen.pos_tag_indexes)
    X0_test = pad_sequences(X0_test, maxlen=30, dtype='int32', padding='post', truncating='post', value=0)
    X1_test = pad_sequences(X1_test, maxlen=30, dtype='int32', padding='post', truncating='post', value=0)
    X0_test = autograd.Variable(torch.from_numpy(X0_test).type(torch.cuda.LongTensor), volatile=True)
    X1_test = autograd.Variable(torch.from_numpy(X1_test).type(torch.cuda.LongTensor), volatile=True)
    print(X0_test.size(), X1_test.size())

    model = BiLSTM_CRF(len(data_gen.word_indexes), len(data_gen.pos_tag_indexes), word_embeddings=data_gen.embedding_matrix)
    # model = BiLSTM_CRF(len(data_gen.word_indexes), len(data_gen.pos_tag_indexes), word_embeddings=None)
    model = model.cuda()
    # optimizer = optim.SGD(model.parameters(), lr=0.01, weight_decay=1e-4)
    optimizer = optim.SGD(model.parameters(), lr=0.01, weight_decay=1e-4)
    # print("Before training...")
    # # labels = predict(model, [X0_test, X1_test])
    # scores, labels = model([X0_test, X1_test])
    # print(labels[0])
    for epoch in range(epochs):
        model.train()
        batch_count = 0
        input_generator = data_gen.generate_input(batch=10)
        for data in input_generator:
            model.zero_grad()
            X, y = data
            X0 = autograd.Variable(torch.from_numpy(X[0]).type(torch.cuda.LongTensor))
            X1 = autograd.Variable(torch.from_numpy(X[1]).type(torch.cuda.LongTensor))
            y = torch.from_numpy(y).type(torch.cuda.LongTensor)

            neg_log_likelihood = model.neg_log_likelihood([X0, X1], y)
            neg_log_likelihood.backward()
            optimizer.step()

            batch_count += 1
            if batch_count % 100 == 0: break # end an epoch
        torch.save(model, 'model.pt')
        # predict
        print('-----------------------------')
        print("epoch %d training loss %f:" %(epoch, neg_log_likelihood.data[0]) )
        # model.eval()
        eval(model, test_data=[X0_test, X1_test, tokens_test, postags_test])
        # for i in range(5):
        #     print(scores[i], [_ for _ in zip(tokens_test[i], postags_test[i])])
        #     print([LABELS[l] for l in labels[i]])
        print(F.softmax(model.transitions.permute(1, 0)).permute(1, 0))


def prepare_test_data(path, word_indexes, postag_indexes, max_len=30):
    word_input = []
    postag_input = []
    words = []
    tags = []
    with open(path) as f:
        file_string = f.read()
        sentences = nltk.sent_tokenize(file_string)
        for sentence in sentences:
            sentence = sentence.strip()
            tokens = nltk.word_tokenize(sentence)
            tagged_tokens = nltk.pos_tag(tokens)
            pos_tags = [s[1] for s in tagged_tokens]
            while(len(tokens) > max_len):
                words.append(tokens[:max_len])
                tokens = tokens[max_len:]

                tags.append(pos_tags[:max_len])
                pos_tags = pos_tags[max_len:]

                word_input.append(
                    [word_indexes[word] if word in word_indexes else word_indexes['UKN'] for word in tokens[:max_len]])
                postag_input.append(
                    [postag_indexes[tag] if tag in postag_indexes else postag_indexes['UKN'] for tag in pos_tags[:max_len]])

            words.append(tokens)
            tags.append(pos_tags)
            word_input.append(
                [word_indexes[word] if word in word_indexes else word_indexes['UKN'] for word in tokens])
            postag_input.append(
                [postag_indexes[tag] if tag in postag_indexes else postag_indexes['UKN'] for tag in pos_tags])


    return word_input, postag_input, words, tags

def eval(model, test_data=None, test_file=None, word_indexes=None, pos_tag_indexes=None):
    # loop over batches because sometimes the sequence is too long to fit in memory
    # model.eval()

    if test_data is not None:
        X0, X1, tokens, postags = test_data
    else:
        if word_indexes is None:
            word_indexes = pickle.load(open('model/word_indexes.pkl'))
        if pos_tag_indexes is None:
            pos_tag_indexes = pickle.load(open('model/pos_tag_indexes.pkl'))
        X0, X1, tokens, postags = prepare_test_data(test_file, word_indexes, pos_tag_indexes)

    # scores, labels = model([X0, X1])
    # for i in range(6):
    #     print(scores[i], [_ for _ in zip(tokens[i], postags[i])])
    #     print([LABELS[l] for l in labels[i]])


    # batches = X0.data.size()[0]
    # all_scores = []
    # all_labels = []
    for i in range(6):
        score, labels = model([X0[i].unsqueeze(0), X1[i].unsqueeze(0)])
        print([item for item in zip(tokens[i], postags[i])])
        # print(labels)
        print([LABELS[l] for l in labels[0]])
        # all_scores.append(score)
        # all_labels.append(labels)
    # return all_scores, all_labels

def predict():
    model = torch.load('models/model.pt')
    model = model.cpu()
    model.eval()
    word_indexes = pickle.load(open('models/word_indexes.pkl', 'rb'))
    pos_tag_indexes = pickle.load(open('models/pos_tag_indexes.pkl', 'rb'))

    input_files = glob.glob('original_files/*.txt')
    for input_file in input_files:
        filename = os.path.split(input_file)[-1]
        basename = filename.split('.')[0]
        print("Processing %s" % basename)
        fout = open('processed/tmp/' + filename, 'w')
        X0, X1, tokens, postags = prepare_test_data(input_file, word_indexes, pos_tag_indexes, max_len=100)
        n_sentences = len(X0)
        for i in range(n_sentences):
            x0 = autograd.Variable(torch.LongTensor(X0[i]), volatile=True).unsqueeze(0)
            x1 = autograd.Variable(torch.LongTensor(X1[i]), volatile=True).unsqueeze(0)
            try:
                score, labels = model([x0, x1])
            except RuntimeError:
                print(len(tokens[i]), tokens[i])
                print(x0, x1)
                raise RuntimeError
            labels = labels[0]
            n_labels = len(labels)
            for j in range(n_labels):
                token = tokens[i][j]
                pos_tag = postags[i][j]
                label = labels[j]

                if label == LABELS.index('B'):
                    if labels[j+1] == LABELS.index('I'):
                        coref = '(0'
                    else:
                        coref = '(0)'
                elif label == LABELS.index('I') and (j == n_labels-1 or labels[j+1] != LABELS.index('I')):
                    coref = '0)'
                else:
                    coref = '-'

                fout.write('\t'.join([basename, token, pos_tag, coref]) + '\n')
        fout.close()


if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        option = sys.argv[1]
    else:
        option = 'predict'

    if option == 'test':
        test_model()
    elif option == 'train':
        train()
    else:
        predict()