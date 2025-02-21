import cvxpy as cp
import numpy as np
from sklearn.datasets import load_svmlight_file # type: ignore
import time
import matplotlib.pyplot as plt

plt.rcParams.update({
    "text.usetex": True,  
    "font.family": "serif",  
    "font.size": 12 
})

def load_data(path):
    """
    Loading SVM file at path
    and computing the matrix X defined in the exercise statement.
    """
    data, labels = load_svmlight_file(path)
    data = data.toarray().T
    data = np.vstack([data, np.ones(data.shape[1])])
    Xy = data*labels
    return data, labels

def D(Xy, lb):
    """Function D defined in the exercise statement."""
    return np.sum(lb) - 0.5 * np.linalg.norm(Xy @ lb)**2

def GradD(Xy, lb):
    """Gradient of the function D with respect to lambda."""
    return np.ones_like(lb) -  Xy.T @ (Xy @ lb)

def PGA(Xy, N):
    """Projected gradient ascent algorithm (PGA)."""
    lb = np.zeros(Xy.shape[1])
    h = 1.0 / np.linalg.norm(Xy.T @ Xy, ord=2)  
    gaps = [pdGap(Xy, lb)]
    miss = [misclassified(Xy, Xy @ lb)]
    for _ in range(N):
        lb_new = np.clip(lb + h * GradD(Xy, lb), 0, alpha)
        while D(Xy, lb_new) <= (D(Xy, lb) + (0.5/h)*(np.linalg.norm(lb_new - lb)**2)):
            h = h/2
            lb_new = np.clip(lb + h * GradD(Xy, lb), 0, alpha)
        lb = lb_new
        gaps.append(pdGap(Xy, lb))
        miss.append(misclassified(Xy, Xy @ lb))
    return lb, gaps, miss

def RCA(Xy, N):
    """Randomized coordinate ascent algorithm (RCA)."""
    lb = np.zeros(Xy.shape[1])
    w = np.zeros(Xy.shape[0])
    gaps = [pdGap(Xy, lb)]
    miss = [misclassified(Xy, w)]
    for _ in range(N):
        i = np.random.randint(0, Xy.shape[1])
        old_lambda = lb[i]
        lb[i] = np.clip(old_lambda + (1/np.linalg.norm(Xy[:, i])**2)*(1 - Xy[:, i].T @ w),
                     0., alpha)
        w = w + Xy[:, i]*(lb[i] - old_lambda)
        gaps.append(pdGap(Xy, lb))
        miss.append(misclassified(Xy, w))
    return lb, gaps, miss

def L1_SMV(Xy):
    """l1 version of the SVM problem."""
    n = Xy.shape[0]
    m = Xy.shape[1]
    w = cp.Variable(n)
    u = cp.Variable(n)
    v = cp.Variable(m)
    
    prob = cp.Problem(
        cp.Minimize(cp.sum(u) + alpha * cp.sum(v)),
        [
            u >= w,
            u >= -w,
            v >= 0,
            v >= 1 - Xy.T @ w
        ]
    )
    
    prob.solve()
    return w.value

def pdGap(Xy, lb):
    """Computing the primal-dual gap of the solutions."""
    w = Xy @ lb
    p = .5 * np.linalg.norm(w)**2 + alpha * np.sum(np.maximum(0., 1 - w.T @ Xy))
    d = D(Xy, lb)
    return np.abs(p - d)

def misclassified(Xy, w):
    """Computing the number of misclassified samples."""
    predictions = Xy.T @ w
    return np.sum(predictions <= 0)

alpha = 1
max_iter = 200

def evaluate(path):
    Xy, _ = load_data(path)
    
    if path == "data/sonar_scale":
        ssize = "tiny_"
    if path == "data/mushrooms":
        ssize = "small_"

    t1_start = time.time()
    lb1, gaps1, miss1 = PGA(Xy, max_iter)
    t1_end = time.time()

    t2_start = time.time()
    lb2, gaps2, miss2 = RCA(Xy, max_iter)
    t2_end = time.time()

    w1 = Xy @ lb1
    w2 = Xy @ lb2

    t3_start = time.time()
    w3 = L1_SMV(Xy)
    t3_end = time.time()

    print(f"Value of alpha: {alpha}")
    print(f"Number of iterations: {max_iter}")
    print(f"Number of features: {Xy.shape[0] - 1}")
    print(f"Number of samples: {Xy.shape[1]}")

    plt.yscale("log")
    plt.plot(np.arange(max_iter + 1), gaps1, linestyle="-", color="blue", label="PGA", alpha=0.7)
    plt.plot(np.arange(max_iter + 1), gaps2, linestyle="-", color="red", label="RCA", alpha=0.7)
    plt.xlabel("Number of iterations")
    plt.ylabel("Primal-dual gap (log-scale)")
    plt.title("Primal-dual gap evolution")
    plt.legend()
    plt.savefig(f"plots/" + ssize + "pd_plot.pdf", bbox_inches="tight")

    plt.clf()

    plt.plot(np.arange(max_iter + 1), miss1, linestyle="-", color="blue", label="PGA", alpha=0.7)
    plt.plot(np.arange(max_iter + 1), miss2, linestyle="-", color="red", label="RCA", alpha=0.7)
    plt.xlabel("Number of iterations")
    plt.ylabel("Misclassified samples")
    plt.title("Number of missclassified samples per iteration")
    plt.legend()
    plt.savefig("plots/" + ssize + "miss_plot.pdf", bbox_inches="tight")

    plt.clf()

    print("EXECUTION TIME")
    print(f"PGA: {t1_end - t1_start}")
    print(f"RCA: {t2_end - t2_start}")
    print(f"l1-SVM: {t3_end - t3_start}")

    print("FINAL PRIMAL-DUAL GAP")
    print(f"PGA: {gaps1[-1]}")
    print(f"RCA: {gaps2[-1]}")

    print("MISCLASSIFIED SAMPLES")
    print(f"PGA: {misclassified(Xy, w1)}")
    print(f"RCA: {misclassified(Xy, w2)}")
    print(f"l1-SVM: {misclassified(Xy, w3)}")

evaluate("data/sonar_scale")
evaluate("data/mushrooms")
