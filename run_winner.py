import pickle
d = pickle.load(open("curious_winner.pkl", "rb"))
nodes, conns = d["genome"]
print(nodes.shape, nodes.dtype, conns.shape, conns.dtype)
print(nodes[:3]); print(nodes[-3:]); print(conns[:3])