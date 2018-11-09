import snap
import numpy as np
import matplotlib.pyplot as plt

class WikiVoteGraph(object):
    def __init__(self):
        self.G = snap.LoadEdgeList(snap.PNGraph, 'data/snap-source-web-2016-09-links-clean-1.paj', 0, 1)
        self.out_deg_v = snap.TIntPrV()
        snap.GetNodeOutDegV(self.G, self.out_deg_v)

        self.deg_freq = self.get_deg_freq_map(self.out_deg_v)

    def get_deg_freq_map(self, out_deg_v):
        deg_freq = {}
        for pair in out_deg_v:
            if pair.GetVal1() == 0 or pair.GetVal2() == 0:
                continue
            if pair.GetVal2() in deg_freq:
                deg_freq[pair.GetVal2()] += 1
            else:
                deg_freq[pair.GetVal2()] = 1
        return deg_freq

    def get_x_and_y(self):
        degs = sorted(self.deg_freq.keys())
        counts = [self.deg_freq[d] for d in degs]
        return degs, counts

wv_graph = WikiVoteGraph()

print wv_graph.G.GetNodes()
print wv_graph.G.GetEdges()

x, y = wv_graph.get_x_and_y()

plt.clf()
plt.cla()

### 2.1 ###
plt.loglog(x, y, ".")

### 2.2 ###
a, b = np.polyfit(np.log10(x), np.log10(y), 1)
print "2.2 coefficients a = %f, b = %f" % (a, b)
lin_reg = 10**(b) * (x ** a)
plt.plot(x, lin_reg)

plt.legend()
plt.title('Article Citation Network: Log-Log Out-Degree Distribution')
plt.xlabel('out-degree (log-10)')
plt.ylabel('count (log-10)')
plt.draw()
plt.show()

