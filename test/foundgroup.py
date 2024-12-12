

def assign_pore_groups(boxres, gaussrandfield):
    try:
        # 初始化变量：
        # group：用于追踪当前正在处理的组号。
        # solid_list和pore_list：分别存储每个组中固体和孔的单元格数量。
        # this_cycle_num_cells：当前组中单元格的数量。
        # moves：一个栈，用于存储待检查的单元格位置。
        group = 0
        solid_list = []
        pore_list = []
        this_cycle_num_cells = 0
        moves = deque()  # 使用deque作为栈
        # 遍历三维空间：通过三个嵌套的for循环遍历整个三维空间（由boxres定义边界）。
        # 对于每个单元格，检查它是否已经被分配到一个组中（getGroup() == -1表示未分配）。
        for starti in range(boxres):
            for startj in range(boxres):
                for startk in range(boxres):
                    # 处理未分配的单元格：对于每个未分配的单元格：
                    # 确定它是固体还是孔（通过getSolid()方法)。
                    # 为当前单元格分配一个新的组号（group）并增加num_cells。
                    # 将当前单元格的位置加入到moves栈中，以开始深度优先搜索。
                    if gaussrandfield[starti][startj][startk].getGroup() == -1:
                        issolid = gaussrandfield[starti][startj][startk].getSolid()
                        group += 1
                        gaussrandfield[starti][startj][startk].setGroup(group)
                        this_cycle_num_cells += 1

                        i, j, k = starti, startj, startk
                        moves.append(gaussrandfield[i][j][k].getPosition())
                        # 深度优先搜索（DFS）：循环执行，直到moves栈为空。
                        # 检查当前单元格是否已经分配到当前组或另一个组中。如果已分配到另一组，则结束当前组的搜索。
                        # 尝试找到一个符合条件的相邻单元格（即与当前单元格相连、状态相同且未分配组号的单元格）。
                        # 这通过检查六个可能的方向来实现。
                        # 如果找到这样的单元格，更新i, j, k为新单元格的位置，并将其位置加入到moves栈中，继续搜索。
                        # 如果在所有方向上都没有找到符合条件的单元格，尝试从moves栈中回退到之前的单元格，并继续搜索。
                        while moves:
                            # 这是一个条件判断语句。如果当前位置[i, j, k]的单元格的getGroup()方法返回的值
                            # 既不是-1（未分组）也不是当前组号group，则退出循环。这里使用not in来检查一个值是否不在列表中。
                            if gaussrandfield[i][j][k].getGroup() not in [-1, group]:
                                break
                            # 如果当前单元格还没有被分配到任何组，增加当前组的单元格计数。
                            if gaussrandfield[i][j][k].getGroup() == -1:
                                this_cycle_num_cells += 1
                            gaussrandfield[i][j][k].setGroup(group)

                            quitloop = False
                            # 遍历六个可能的方向（上、下、左、右、前、后），尝试找到一个符合条件的相邻单元格。
                            for di, dj, dk in [(1, 0, 0), (-1, 0, 0), (0, 1, 0),
                                               (0, -1, 0), (0, 0, 1), (0, 0, -1)]:
                                ni, nj, nk = i + di, j + dj, k + dk
                                # 检查相邻单元格是否在空间边界内，是否与当前单元格状态相同（固体或孔），并且是否未被分配到任何组。
                                # 检查索引是否在有效范围内
                                if 0 <= ni < boxres and 0 <= nj < boxres and 0 <= nk < boxres:
                                    # 获取当前位置的单元格
                                    subcell = gaussrandfield[ni][nj][nk]
                                    # 如果有满足条件的则提前结束循环
                                    if subcell.getSolid() == issolid and subcell.getGroup() == -1:
                                        i, j, k = ni, nj, nk
                                        quitloop = True
                                        break
                            # 如果没有任何一个相邻单元格满足条件，循环结束后
                            # quitloop保持为False。
                            # 检查栈moves是否非空，如果不为空，则从栈中弹出一个位置并更新i, j, k。
                            # 实际上这是在执行回退的过程，知道找到符合条件的新单元格
                            if not quitloop and moves:  # 防止在空栈上pop
                                mp = moves.pop()
                                i, j, k = mp.getX(), mp.getY(), mp.getZ()
                            else:
                                moves.append(gaussrandfield[i][j][k].getPosition())
                        # 分组完成：当moves栈为空时，表示当前组的所有相连单元格都已经被找到和标记。根据当前单元格是固体还是孔，
                        # 将num_cells加入到solid_list或pore_list中，并将另一个列表中对应位置加入0。
                        if issolid:
                            solid_list.append(this_cycle_num_cells)
                            pore_list.append(0)
                        else:
                            pore_list.append(this_cycle_num_cells)
                            solid_list.append(0)
                        this_cycle_num_cells = 0
        return solid_list, pore_list, gaussrandfield
    except MemoryError:
        print("Resolution parameter is too large!!")
        return [], []  # 返回空列表表示失败
