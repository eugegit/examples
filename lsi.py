# "Eugene Morozov"<Eugene ~at~ HiEugene.com>

import scipy.linalg
import matplotlib.pyplot as plt
import pickle
from util import *

v = 1
epoch_sec = 1 # int(time.time())
start_time = preamble(epoch_sec)

# picklefile = open("/data1/wiki/terms-stats-topology.bin", "rb")
data = pickle.load(picklefile) # id -> (title, url, term -> count)
picklefile.close()

# trim dataset
data2 = {}
keys = list(data.keys()) # [0:100]
for id in keys:
  tc = { k:v for(k,v) in data[id][2].items() if len(k)>3 }
  data2[id] = (data[id][0], data[id][1], tc)
data = data2

# build term-by-document matrix (where rows are words and columns are documents; values are weighted occurrences)
terms = set()
docs = set()
global_term_count = {}
for doc, x in data.items():
  docs.update([doc])
  title, url, term_count = x
  terms.update(term_count.keys())
  for term, count in term_count.items():
    if term in global_term_count:
      global_term_count[term] += count
    else:
      global_term_count[term] = count
row_idx = sorted(list(terms))
col_idx = sorted(list(docs))
M = len(row_idx)
N = len(col_idx)
print(f"# of terms = {M:,d}; # of documents = {N:,d}")

A = np.zeros([M, N])
for i, term in enumerate(row_idx):
  if i%2000 == 0: print(f"i = {i:,d} of {M:,d}")
  df = 0
  for j, doc in enumerate(col_idx):
    if term in data[doc][2]:
      df += 1
  for j, doc in enumerate(col_idx):
    if term in data[doc][2]:
      tf = data[doc][2][term]
    else:
      tf = 0
    tf_idf = 0 if tf == 0 else (1+np.log(tf))*np.log(N/df)
    A[i,j] = tf_idf

T, s, D = scipy.linalg.svd(A) # D is already transposed

K = 20
T2 = T[:,0:K]
S2 = np.diag(s[0:K])
D2 = D[:,0:K]

# T2 = np.copy(T)
# D2 = np.copy(D)
# fn2 = scipy.linalg.norm(A)
# T2[:, N:] = 0
# D2[:, N:] = 0
# fn1 = np.zeros(N)
# for K in range(N-1, 2-2, -1):
#   S[K, K] = 0
#   T2[:, K] = 0
#   D2[:, K] = 0
#   Ahat = T2.dot(S).dot(D2)
#   fn1[K] = scipy.linalg.norm(A - Ahat)
#   print(f"K = {K+1}; Frobenius norm of diff = {fn1[K]:.02f} ({int(fn1[K]*100/fn2):d}%)")
#
# if v:
#   fig = plt.figure()
#   ax = fig.gca()
#   plt.ion()
#   plt.tight_layout()
#   plt.grid(True)
#   plt.show()
#   ax.plot(fn1, marker="*")
#   ax.set_xlabel("K")
#   ax.set_title("Frobenius norm of diff")

def qw(s):
  return tuple(s.split())

def plot_word(ax, coor, term):
  ax.scatter(coor[0], coor[1], marker="o")
  ax.annotate(term, coor)

if True:
  # document similarity = (DS)(DS)^T
  B = S2.dot(D2.T).T
  B = B.dot(B.T)
  if v:
    fig = plt.figure(); ax = fig.gca(); plt.ion()
    plt.show(); # plt.tight_layout()
    im = ax.imshow(B)
    ax.set_xticks(np.arange(N))
    ax.set_yticks(np.arange(N))
    labels = [data[doc][0] for doc in col_idx]
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    ax.set_title(f"document similarity; K={K}")
    ax.figure.colorbar(im, ax=ax)
    idx1 = col_idx.index("382447") # "76340"
    idx2 = col_idx.index("253492")
    plt.axis([idx1-20, idx1+20, idx2-20, idx2+20])

  # term similarity = (TS)(TS)^T
  C = S2.dot(T2.T).T
  C = C.dot(C.T)
  if v:
    fig = plt.figure(); ax = fig.gca(); plt.ion()
    plt.show(); # plt.tight_layout()
    im = ax.imshow(C)
    ax.set_xticks(np.arange(M))
    ax.set_yticks(np.arange(M))
    labels = row_idx
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    ax.set_title(f"term similarity; K={K}")
    ax.figure.colorbar(im, ax=ax)
    idx1 = row_idx.index("dimensionality")
    idx2 = row_idx.index("reduction")
    plt.axis([idx1-20, idx1+20, idx2-20, idx2+20])

q_str = """dimensionality reduction eigenvalue eigenvalues"""
q = qw(q_str)
d = np.zeros([M, 1])
for term in q:
  term = term.lower()
  if term in row_idx:
    i = row_idx.index(term)
    d[i] = 1
  else:
    print(f"'{term}' is not corpus")
S21 = scipy.linalg.inv(S2)
dhat = S21.dot(T2.T).dot(d)
D2norm = np.diag(D2.dot(D2.T))
res = dhat.T.dot(D2.T)
res_norm = res.flatten() / D2norm / dhat.T.dot(dhat)
if v:
  fig = plt.figure(); ax = fig.gca(); plt.ion()
  plt.show(); plt.grid(True); plt.tight_layout()
  [plot_word(ax, [j, res_norm[0,j]], data[doc][0]) for j, doc in enumerate(col_idx)]
  ax.set_title(f"query: \"{q_str}\"; K={K}")

postscript(start_time)
