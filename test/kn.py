# def generate_kn(mink, maxk, minfk, maxfk, k_fk, num_samples=10000):
#     """
#     生成满足特定条件的Kn向量集合。
#
#     参数:
#     - mink, maxk: K的最小和最大值。
#     - minfk, maxfk: fK的最小和最大值。
#     - k_fk: 包含K和fK值的字典。
#     - num_samples: 生成随机数的数量。
#
#     返回:
#     - Kn: 满足条件的Kn向量集合。
#     - valid_count: 满足条件的Kn数量。
#     """
#     # 创建插值函数
#     interp_func = interp1d(k_fk['K'], k_fk['fK'], kind='linear')
#
#     # 批量生成随机数
#     randks = np.random.uniform(mink, maxk, num_samples)
#     randfks = np.random.uniform(minfk, maxfk, num_samples)
#     calcfks = interp_func(randks)
#
#     # 筛选满足条件的随机数
#     valid_indices = randfks <= calcfks
#     valid_randks = randks[valid_indices]
#
#     # 生成随机向量并标准化
#     kvecs = np.random.uniform(-1, 1, (len(valid_randks), 3))
#     kvecsnorms = np.linalg.norm(kvecs, axis=1, keepdims=True)
#     normalized_kvecs = kvecs / kvecsnorms
#
#     # 调整向量长度
#     Kn = normalized_kvecs * valid_randks[:, np.newaxis]
#
#     # 统计满足条件的Kn数量
#     valid_count = len(valid_randks)
#
#     return Kn, valid_count
#
# def parallel_generate_kn(mink, maxk, minfk, maxfk, k_fk, num_samples, num_workers=10):
#     """
#     并行生成满足特定条件的Kn向量集合。
#
#     参数:
#     - num_samples: 总共生成随机数的数量。
#     - num_workers: 并行执行的工作进程数量。
#     """
#     samples_per_worker = num_samples // num_workers
#     futures = []
#     Kn_list = []
#     valid_count_total = 0
#
#     with ProcessPoolExecutor(max_workers=num_workers) as executor:
#         for _ in range(num_workers):
#             future = executor.submit(generate_kn, mink, maxk, minfk, maxfk, k_fk, samples_per_worker)
#             futures.append(future)
#
#         for future in futures:
#             Kn, valid_count = future.result()
#             Kn_list.append(Kn)
#             valid_count_total += valid_count
#
#     # 合并所有工作进程生成的Kn
#     Kn_total = np.vstack(Kn_list)
#     print(Kn_total, valid_count_total)
#     return Kn_total, valid_count_total


# Kn = parallel_generate_kn(mink, maxk, minfk, maxfk, k_fk, num_samples, num_workers=4)