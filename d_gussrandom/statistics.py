import numpy as np


def calc_stats(gaussrandfield, apar, porosity, boxres, poreList, solidList):
    def calculateSVRatio():
        # 这里简化了积分计算，您需要根据实际情况调整
        d = 1  # 假设的积分结果
        return np.pi * porosity * (1.0 - porosity) * apar / d

    def checkIfBicontinuous():
        porecheck = [set() for _ in range(6)]
        solidcheck = [set() for _ in range(6)]
        for a in range(boxres):
            for b in range(boxres):
                for dim, (front, back) in enumerate(zip([0, a, b], [boxres - 1, boxres - 1, boxres - 1])):
                    if gaussrandfield[front][a][b].getSolid():
                        solidcheck[dim].add(gaussrandfield[front][a][b].getGroup())
                    else:
                        porecheck[dim].add(gaussrandfield[front][a][b].getGroup())
                    if gaussrandfield[back][a][b].getSolid():
                        solidcheck[dim+3].add(gaussrandfield[back][a][b].getGroup())
                    else:
                        porecheck[dim+3].add(gaussrandfield[back][a][b].getGroup())

        solidcont = any(solidcheck[i].intersection(solidcheck[i+3]) for i in range(3))
        porecont = any(porecheck[i].intersection(porecheck[i+3]) for i in range(3))
        return solidcont and porecont

    def calculateConnectivity():
        maxporesize = max(poreList, default=0)
        maxsolidsize = max(solidList, default=0)
        sum_excluding_max = sum(size for i, size in enumerate(poreList + solidList) if size not in [maxporesize, maxsolidsize])
        total = boxres ** 3
        return (1.0 - sum_excluding_max / total) * 100

    svRatio = calculateSVRatio()
    isBicontinuous = checkIfBicontinuous()
    connectivity = calculateConnectivity()

    return svRatio, isBicontinuous, connectivity


