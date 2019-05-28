import numpy as np


data = open("kafka.txt", 'r').read()
chars = list(set(data))

data_size, vocab_size = len(data), len(chars)

char_to_ix = {ch: i for i, ch in enumerate(chars)}
ix_to_char = {i: ch for i, ch in enumerate(chars)}

vector_for_char_a = np.zeros((vocab_size, 1))
vector_for_char_a[char_to_ix['a']] = 1

# hyper parameters
hidden_size = 100
seq_length = 25
learning_rate = 1e-1

# model parameters
Wxh = np.random.randn(hidden_size, vocab_size)*0.1
Whh = np.random.randn(hidden_size, hidden_size)*0.1
Why = np.random.randn(vocab_size, hidden_size)*0.1
bh = np.zeros((hidden_size, 1))
by = np.zeros((vocab_size, 1))


def loss_fun(inputs, targets, hprev):
    xs, hs, ys, ps = {}, {}, {}, {}
    hs[-1] = np.copy(hprev)
    loss = 0

    for i in range(len(inputs)):
        xs[i] = np.zeros((vocab_size, 1))
        xs[i][inputs[i]] = 1

        hs[i] = np.tanh(np.dot(Wxh, xs[i]) + np.dot(Whh, hs[i-1]) + bh)
        ys[i] = np.dot(Why, hs[i]) + by
        ps[i] = np.exp(ys[i])/np.sum(np.exp(ys[i]))

        loss += -np.log(ps[i][targets[i], 0])

    dWxh, dWhh, dWhy = np.zeros_like(Wxh), np.zeros_like(Whh), np.zeros_like(Why)
    dbh, dby = np.zeros_like(bh), np.zeros_like(by)
    dhnext = np.zeros_like(hs[0])

    for j in reversed(range(len(inputs))):
        dy = np.copy(ps[j])
        dy[targets[j]] -= 1
        dWhy += np.dot(dy, hs[j].transpose())
        dby += dy

        dh = np.dot(Why.T, dy)+dhnext
        dhraw = (1 - hs[j]*hs[j]) * dh

        dbh += dhraw

        dWxh += np.dot(dhraw, xs[j].T)
        dWhh += np.dot(dhraw, hs[j-1].T)

        dhnext = np.dot(Whh.T, dhraw)
    return loss, dWxh, dWhh, dWhy, dbh, dby, hs[len(inputs)-1]


def sample(h, seed, n):
    x = np.zeros((vocab_size, 1))
    x[seed] = 1

    ixs = []

    for i in range(n):
        h = np.tanh(np.dot(Wxh, x) + np.dot(Whh, h) + bh)
        y = np.dot(Why, h) + by
        p = np.exp(y) / np.sum(np.exp(y))
        ix = np.random.choice(range(vocab_size), p=p.ravel())
        x = np.zeros((vocab_size, 1))
        x[ixs] = 1
        ixs.append(ix)

    text = ''.join(ix_to_char[ix] for ix in ixs)
    print(text)


n, q = 0, 0
mWxh, mWhh, mWhy = np.zeros_like(Wxh), np.zeros_like(Whh), np.zeros_like(Why)
mbh, mby = np.zeros_like(bh), np.zeros_like(by)
smooth_loss = -np.log(1.0/vocab_size)*seq_length

while n < 100 * 1000:
    if q+seq_length+1 > data_size or n == 0:
        hprev = np.zeros((hidden_size, 1))
        p = 0
    inputs = [char_to_ix[ix] for ix in data[q:q+seq_length]]
    targets = [char_to_ix[iy] for iy in data[q+1:q+seq_length+1]]

    loss, dWxh, dWhh, dWhy, dbh, dby, hprev = loss_fun(inputs, targets, hprev)
    smooth_loss = smooth_loss * 0.999 + loss * 0.001

    if n % 1000 == 0:
        print("iteration number = " + str(n))
        print("loss = ", smooth_loss)

    for param, dparam, mem in zip([Wxh, Whh, Why, bh, by],
                                  [dWxh, dWhh, dWhy, dbh, dby],
                                  [mWxh, mWhh, mWhy, mbh, mby]):
        mem += dparam * dparam
        param += -learning_rate * dparam / np.sqrt(mem + 1e-8)

    q += seq_length
    n += 1

print("Text Generated-------")
sample(hprev, inputs[0], 200)









