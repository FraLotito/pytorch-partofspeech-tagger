import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.optim as optim
import torch.nn.functional as F

word_to_ix = {}
car_to_ix = {}

def get_index_of_max(input):
    index = 0
    for i in range(1, len(input)):
        if input[i] > input[index]:
            index = i 
    return index

def get_max_prob_result(input, ix_to_tag):
    return ix_to_tag[get_index_of_max(input)]


def prepare_car_sequence(word, to_ix):
	idxs = []
	for car in word:
		idxs.append(to_ix[car])
	return idxs

def prepare_sequence(seq, to_ix):
    res = []
    for w in seq:
	    res.append((to_ix[w], prepare_car_sequence(w, car_to_ix)))
    return res

def prepare_target(seq, to_ix):
	idxs = []
	for w in seq:
		idxs.append(to_ix[w])
	return autograd.Variable(torch.LongTensor(idxs))

training_data = [
    ("The dog ate the apple".split(), ["DET", "NN", "V", "DET", "NN"]),
    ("Everybody read that book".split(), ["NN", "V", "DET", "NN"])
]


for sent, tags in training_data:
    for word in sent:
        if word not in word_to_ix:
            word_to_ix[word] = len(word_to_ix)
        for car in word:
        	if car not in car_to_ix:
        		car_to_ix[car] = len(car_to_ix)


tag_to_ix = {"DET": 0, "NN": 1, "V": 2}
ix_to_tag = {0: "DET", 1: "NN", 2: "V"}

CAR_EMBEDDING_DIM = 3
WORD_EMBEDDING_DIM = 6
HIDDEN_DIM = 6

class LSTMTagger(nn.Module):

    def __init__(self, word_embedding_dim, car_embedding_dim, hidden_dim, vocab_size, alphabet_size, tagset_size):

        super(LSTMTagger, self).__init__()

        self.hidden_dim = hidden_dim
        self.car_embedding_dim = car_embedding_dim

        self.car_embeddings = nn.Embedding(alphabet_size, car_embedding_dim)
        self.lstm_car = nn.LSTM(car_embedding_dim, car_embedding_dim)

        self.word_embeddings = nn.Embedding(vocab_size, word_embedding_dim)
        self.lstm_word = nn.LSTM(word_embedding_dim+car_embedding_dim, hidden_dim)

        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

        self.hidden = self.init_hidden(hidden_dim)
        self.hidden_car = self.init_hidden(CAR_EMBEDDING_DIM)

    def init_hidden(self, dim):

        return (autograd.Variable(torch.zeros(1, 1, dim)),
                autograd.Variable(torch.zeros(1, 1, dim)))

    def forward(self, sentence):
        word_idxs = []
        lstm_car_result = []
        for word in sentence:
        	self.hidden_car = self.init_hidden(CAR_EMBEDDING_DIM)
        	word_idxs.append(word[0])     		
        	char_idx = autograd.Variable(torch.LongTensor(word[1]))
        	car_embeds = self.car_embeddings(char_idx)
        	lstm_car_out, self.hidden_car = self.lstm_car(car_embeds.view(len(word[1]), 1, CAR_EMBEDDING_DIM), self.hidden_car)
        	lstm_car_result.append(lstm_car_out[-1])

        lstm_car_result = torch.stack(lstm_car_result)
        
        word_embeds = self.word_embeddings(autograd.Variable(torch.LongTensor(word_idxs))).view(len(sentence), 1, WORD_EMBEDDING_DIM)

        lstm_in = torch.cat((word_embeds, lstm_car_result), 2)

        lstm_out, self.hidden = self.lstm_word(lstm_in, self.hidden)

        tag_space = self.hidden2tag(lstm_out.view(len(sentence), -1))
        tag_scores = F.log_softmax(tag_space)
        return tag_scores

model = LSTMTagger(WORD_EMBEDDING_DIM, CAR_EMBEDDING_DIM, HIDDEN_DIM, len(word_to_ix), len(car_to_ix), len(tag_to_ix))
loss_function = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)


for epoch in range(300):  
    for sentence, tags in training_data:
        model.zero_grad()

        model.hidden = model.init_hidden(HIDDEN_DIM)

        sentence_in = prepare_sequence(sentence, word_to_ix)
        
        targets = prepare_target(tags, tag_to_ix)

        tag_scores = model(sentence_in)

        loss = loss_function(tag_scores, targets)
        loss.backward()
        optimizer.step()


# ======================= TEST

test_sentence = training_data[0][0]
inputs = prepare_sequence(test_sentence, word_to_ix)
tag_scores = model(inputs)
for i in range(len(test_sentence)):
	print('{}: {}'.format(test_sentence[i],get_max_prob_result(tag_scores[i].data.numpy(), ix_to_tag)))
