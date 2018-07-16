import numpy as np

class NBC:
    def dnorm(self,x,u,sd):
        #calculates the log-normal
        bottom = np.log(np.sqrt(2*np.pi*sd*sd))
        top = -(x-u)*(x-u) / (2*sd*sd)
        return top - bottom

    def __init__(self, feature_types, num_classes, epsilon = 0.0025):
        self.ftypes = feature_types
        self.D = len(feature_types)
        self.num_classes = num_classes
        self.eps = epsilon

    def fit(self, xdata, ydata):
        self.N = len(ydata)
        self.classes, self.ycounts = np.unique(ydata, return_counts=True)
        self.dists = [] 
        nr = np.arange(self.N)
        for c in self.classes:
            cond = ydata[nr] == c
            #training data belonging to class c
            samples_belonging = xdata[cond]
            curdist = []
            for i in range(len(self.ftypes)):
                if self.ftypes[i] == 'b':
                    uq, counts = np.unique(samples_belonging[:,i], return_counts=True)
                    numones = 0
                    if len(uq) == 0: #there are no samples belonging to class c
                        numones = 0
                    elif len(uq) == 1 and uq[0] == 1: #there are only items of value 1
                        numones = counts[0]
                    elif len(uq) == 2: #there are items of both values
                        numones = counts[1]
                    #if only zeroes, no need to do anything

                    #laplace-smoothed p-parameter
                    curdist.append((1 + numones) / (len(samples_belonging)+2)) 
                
                else: #real-valued parameter
                    u = np.mean(samples_belonging[:,i])
                    sd = max(np.std(samples_belonging[:,i]), self.eps)
                    
                    #smoothed normal distribution
                    curdist.append((u, sd))
            self.dists.append(curdist)

    def predict(self, Xt):
     ans = []
     for X in Xt:
        max_likelihood = -float("inf")
        mle_class = None
        for ci in range(len(self.classes)):
            cur_likelihood = np.log(self.ycounts[ci] / self.N)
            curdist = self.dists[ci]
            for fi in range(len(self.ftypes)):
                if self.ftypes[fi] == 'b':
                    p = curdist[fi]
                    if X[fi] == 0:
                        p = 1 - p
                    cur_likelihood += np.log(p)
                
                else:
                    mean, sd = curdist[fi]
                    p = self.dnorm(X[fi],mean,sd)      
                    cur_likelihood += p

            if cur_likelihood > max_likelihood:
                max_likelihood = cur_likelihood
                mle_class = self.classes[ci]
        ans.append(mle_class)
     return np.array(ans)
