def generate_matrix(k_fk, boxres, boxsize, alpha):
    """
    生成高斯随机场并根据该场的值决定每个体素的状态。
    :param k_fk: 一个二维数组，包含波数和对应的光谱函数值。
    :param boxres: 体素网格的分辨率。
    :param boxsize: 体素网格的大小。
    :param alpha: 决定体素开放或关闭的阈值。
    :return: 一个三维数组，表示高斯随机场的体素网格。
    """

    def interpolate(xvals, yvals, interx):
        j = 0
        while interx > xvals[j] and j < len(xvals) - 1:
            j += 1
        if j == 0:
            return yvals[0]
        if j == len(xvals):
            return yvals[-1]
        m = (yvals[j] - yvals[j - 1]) / (xvals[j] - xvals[j - 1])
        c = yvals[j] - m * xvals[j]
        result = m * interx + c
        return result



    mink = np.min(k_fk['K'])
    minfk = np.min(k_fk['fK'])
    maxk = np.max(k_fk['K'])
    maxfk = np.max(k_fk['fK'])
    Kn = np.zeros((10000, 3))
    phin = np.random.uniform(0, 2 * math.pi, 10000)
    for j in range(10000):
        while True:
            randk = np.random.uniform(mink, maxk)
            randfk = np.random.uniform(minfk, maxfk)
            calcfk = interpolate(k_fk['K'], k_fk['fK'], randk)
            if randfk <= calcfk:
                kvec = np.random.uniform(-1, 1, 3)
                kvecnorm = np.linalg.norm(kvec)
                # 乘以 randk 可以被理解为调整向量的长度，使其与 randk 成比例。
                # 结果存储在 Kn[j] 中，这里 Kn 是一个数组，用于存储这种处理过的向量。
                Kn[j] = kvec * randk / kvecnorm
                break
    print(Kn)
    gaussrandfield = np.empty((boxres, boxres, boxres), dtype=object)
    rmat = np.array([[(k * 1.0 * (boxsize / boxres)) for k in range(boxres)] for _ in range(3)]).T
    print(rmat)

    for l in range(boxres):
        for m in range(boxres):
            for n in range(boxres):
                sumtemp = 0.0
                for N in range(100):
                    sumtemp += np.cos(Kn[N][0] * rmat[l][0] + Kn[N][1] * rmat[m][1] + Kn[N][2] * rmat[n][2] + phin[N])
                t = math.sqrt(2.0) * sumtemp / 100.0
                issolid = t < alpha
                pos = globalvars.MatrixPosition(l, m, n)
                gaussrandfield[l, m, n] = globalvars.MorphVoxel(issolid)
                gaussrandfield[l, m, n].set_position(pos)

    print(gaussrandfield)
    return gaussrandfield