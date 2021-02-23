# %%
import numpy as np
import matplotlib.pyplot as plt

def reg1dim(x, y):
    n = len(x)
    a = ((np.dot(x, y)- y.sum() * x.sum()/n)/
        ((x ** 2).sum() - x.sum()**2 / n))
    b = (y.sum() - a * x.sum())/n
    return a, b


def calc_error(x_list, y_list):
    dim = len(x_list)
    err_list = np.zeros((dim, dim))
    for i in range(dim-1):
        for j in range(i+1, dim):
            x = x_list[i:j+1]
            y = y_list[i:j+1]

            a, b = reg1dim(x, y)
            y1 = a * x + b
            err_list[i][j]=((y1 - y)**2).sum()
    
    return(err_list)


def make_points():
    seg1 = [0, 5]
    seg2 = [6, 12]
    seg3 = [13, 20]
    seg_list=[seg1,seg2,seg3]
    a_list = [1, 8, 2]
    b = 5.

    x_list = []
    y_list = []
    by = b

    np.random.seed(40)

    for s, a in zip(seg_list, a_list):
        for x in range(s[0], s[1]+1):
            x_list.append(np.random.normal(0, 0.2) + float(x))
            y = by + a
            by = y
            y_list.append(np.random.normal(0, 0.5) + float(y))

    return np.array(x_list), np.array(y_list)

def calc_opt(e, c):
    M = np.array([0]*len(e[0]))
    M[0] = 0
    for j in range(1,len(M)):
        opt = np.inf
        for i in range(0,j):
            o = e[i][j] + c + M[i-1]
            if opt > o:
                opt = o
        
        M[j] = opt
    
    return M

def calc_segment(M, e, c):
    seg_list = []
    j = len(M)-1
    while j > 0:
        ii = np.inf
        min_i = j
        for i in list(range(j))[::-1]:
            new = e[i][j] + c + M[i-1]
            if ii > new:
                ii = new
                min_i = i
        
        if min_i == 1:
            min_i = 0
        seg_list.append([min_i,j])
        j = min_i - 1
    
    return seg_list
            

def main():
    c = 2
    x,y = make_points()
    e=calc_error(x,y)
    M = calc_opt(e, c)
    seg_list = calc_segment(M, e, c)

    # plotしたら完成!!
    l_list = []
    plt.scatter(x, y)
    for s in seg_list:
        xx = x[s[0]:s[1] + 1]
        yy = y[s[0]:s[1] + 1]
        a, b = reg1dim(xx, yy)

        yy = a * xx + b

        plt.plot(xx, yy)


def test2():
    x,y = make_points()
    a,b = reg1dim(x,y)

def test():
    x,y = make_points()
    a,b = reg1dim(x,y)
    print(calc_error(x,y))
    plt.scatter(x,y)
    plt.scatter(x,a*x+b)
    plt.show()


main()


# %%
for i in list(range(0,10))[::-1]:
    print(i)