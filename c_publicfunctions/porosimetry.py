import numpy as np
import logging
from typing import Literal
from tqdm import tqdm
import inspect as insp
import dask
from scipy.ndimage import label as spim_label, distance_transform_edt
from scipy.signal import fftconvolve
from skimage.morphology import disk, ball, cube, square

# 设置日志记录器
logger = logging.getLogger(__name__)

# 全局设置类
class Settings:
    def __init__(self):
        self.ncores = -1  # 默认用所有核心
        self.tqdm = {'disable': False}  # tqdm进度条设置

settings = Settings()


def get_border(shape, mode='faces'):
    """
    获取指定模式下的边界掩码。
    例如 mode='faces' 则返回正方体/立方体6个面的布尔掩码。
    """
    ndim = len(shape)
    # 获取输入shape的维度数量，比如二维图像ndim=2，三维体数据ndim=3
    output = np.zeros(shape, dtype=bool)
    # 创建一个与输入shape相同形状的布尔数组，初始值全为False，用作边界掩码
    if mode == 'faces':
        # 如果mode是'faces'，则标记所有维度的边界面
        for ax in range(ndim):
            # 遍历每一个维度轴
            slc = [slice(None)] * ndim
            # 构造切片列表，初始时表示选择该维度上的所有元素，比如二维中为 [slice(None), slice(None)]
            slc[ax] = 0
            # 将当前维度的切片替换为索引0，代表该维度的最前面一层边界面
            output[tuple(slc)] = True
            # 将output中该边界面对应的位置赋值为True，表示该位置属于边界
            slc[ax] = -1
            # 将当前维度切片替换成索引-1，代表该维度的最后一层边界面
            output[tuple(slc)] = True
            # 同样将该维度最后一层边界位置设为True
    return output
    # 返回标记了边界的布尔数组


def extract_subsection(a, shape):
    """
    从更大的数组a中提取与shape大小一致的子数组，中心裁剪。
    """
    slices = []
    for i, dim in enumerate(shape):
        if a.shape[i] > dim:
            diff = a.shape[i] - dim
            slices.append(slice(diff // 2, -diff // 2 if diff > 1 else None))
        else:
            slices.append(slice(0, dim))
    return a[tuple(slices)]

def subdivide(im, divs, overlap):
    """
    对输入im划分分块，divs为每维的块数，overlap为每块的重叠像素数。
    返回每个分块的切片信息。
    """
    im_shape = np.array(im.shape)
    overlap = np.array(overlap, ndmin=1)
    if overlap.size == 1:
        overlap = np.ones_like(im_shape) * overlap
    divs = np.array(divs, ndmin=1)
    if divs.size == 1:
        divs = np.ones_like(im_shape) * divs
    chunk_sizes = np.ceil(im_shape / divs).astype(int)
    slices = []
    for i in range(int(np.prod(divs))):
        div_loc = np.unravel_index(i, divs)
        div_loc = np.array(div_loc)
        # 计算每块的起止坐标，保留重叠区
        starts = div_loc * chunk_sizes - overlap * (div_loc > 0)
        starts = np.maximum(starts, 0)
        ends = np.minimum((div_loc + 1) * chunk_sizes + overlap * (div_loc < divs - 1), im_shape)
        slc = [slice(starts[dim], ends[dim]) for dim in range(im.ndim)]
        slices.append(slc)
    return slices

def recombine(ims, slices, overlap):
    """
    将分块处理后的结果ims，按slices指定的位置还原成完整图像。
    """
    shape = []
    # 计算输出的整体shape
    for ax in range(len(slices[0])):
        length = 0
        for slc in slices:
            length = max(length, slc[ax].stop)
        shape.append(length)
    im_out = np.zeros(shape, dtype=ims[0].dtype)
    # 每块放置回对应区域，后写入优先生效
    for im_i, slc in zip(ims, slices):
        im_out_slc = [slice(slc[ax].start, slc[ax].stop) for ax in range(len(slc))]
        im_out[tuple(im_out_slc)] = im_i
    return im_out


def trim_disconnected_blobs(im, inlets, strel=None):
    """
    仅保留与inlets连通的前景部分（一般用于去除未贯通孤立区域）。
    支持2D/3D。
    """
    # 下面这部分是处理入口区域inlets的格式，转换为布尔掩码
    if isinstance(inlets, tuple):
        # 如果inlets是索引元组（例如坐标数组），则创建一个全False的数组inlets
        temp = np.copy(inlets)
        inlets = np.zeros_like(im, dtype=bool)
        inlets[temp] = True
        # 将对应坐标设为True，表示入口区域
    elif (inlets.shape == im.shape) and (inlets.max() == 1):
        # 如果inlets已经是和im同形状的二值（0/1）数组，直接转换为bool型
        inlets = inlets.astype(bool)
    else:
        # 传入的inlets格式不符合要求，则抛出异常
        raise Exception("inlets not valid, refer to docstring for info")
    # 选择用于连通性判断的结构元素（邻域）
    if strel is None:
        # 如果未指定结构元素，3D用立方体3x3x3，2D用正方形3x3
        strel = cube(3) if im.ndim == 3 else square(3)
    # 对入口区域inlets和前景区域im > 0进行连通域标记
    # 用“入口区域” 与 “前景” 的组合，连通区域打标签
    labels = spim_label(inlets + (im > 0), structure=strel)[0]
    # 找出所有和入口区域相连通的标签值
    keep = np.unique(labels[inlets])
    # 去掉标签0（背景）
    keep = keep[keep > 0]
    # 只保留与入口连通的那些标签对应的区域
    im2 = np.isin(labels, keep) * im
    # 这里用isin判断每个像素是否属于keep列表内的区域，是则保留，否则清除
    # 用 dt >= r **
    # 筛选每步能装下至少r-半径球的位置;
    # 所有连通路径上的最小半径。
    # 再用 trim_disconnected_blobs 保证这些位置是跟入口连通的（不被“窄口”或者“隔断”阻挡）。

    return im2


def chunked_func(func,
                 overlap=None,
                 divs=2,
                 cores=None,
                 im_arg=["input", "image", "im"],
                 strel_arg=["strel", "structure", "footprint"],
                 **kwargs):
    """
    对大图像做分块并行处理，每块自行调用func，最后拼合。
    func需支持im_arg作为主输入。
    """
    @dask.delayed
    def apply_func(func, **kwargs):
        return func(**kwargs)
    # 识别输入的图像关键字
    if isinstance(im_arg, str):
        im_arg = [im_arg]
    for item in im_arg:
        if item in kwargs:
            im = kwargs[item]
            im_arg = item
            break
    im = kwargs[im_arg]
    divs = np.ones((im.ndim,), dtype=int) * np.array(divs)
    if cores is None:
        cores = settings.ncores
    # 自动推断重叠区域大小
    if overlap is not None:
        overlap = overlap * (divs > 1)
    else:
        if isinstance(strel_arg, str):
            strel_arg = [strel_arg]
        for item in strel_arg:
            if item in kwargs:
                strel = kwargs[item]
                break
        overlap = np.array(strel.shape) * (divs > 1)
    slices = subdivide(im=im, divs=divs, overlap=overlap)
    res = []
    for s in slices:
        # 分块后每块输入func，主输入更换为分块
        kwargs[im_arg] = dask.delayed(np.ascontiguousarray(im[tuple(s)]))
        res.append(apply_func(func=func, **kwargs))
    ims = dask.compute(res, num_workers=cores)[0]
    im2 = recombine(ims=ims, slices=slices, overlap=overlap)
    return im2

def fftmorphology(im, strel, mode='opening'):
    """
    用快速傅里叶卷积FFT实现腐蚀/膨胀/开/闭等形态学操作，适和大结构元提速。
    支持2D/3D。
    """
    im = np.squeeze(im)
    def erode(im, strel):
        # 腐蚀操作
        return fftconvolve(im, strel, mode='same') > (strel.sum() - 0.1)
    def dilate(im, strel):
        # 膨胀操作
        return fftconvolve(im, strel, mode='same') > 0.1
    temp = np.pad(im, pad_width=1, mode='constant', constant_values=0)  # 保证边界
    if mode.startswith('ero'):
        temp = erode(temp, strel)
    if mode.startswith('dila'):
        temp = dilate(temp, strel)
    # 去除填充
    if im.ndim == 2:
        result = temp[1:-1, 1:-1]
    elif im.ndim == 3:
        result = temp[1:-1, 1:-1, 1:-1]
    # 递归实现开运算和闭运算
    if mode.startswith('open'):
        temp = fftmorphology(im=im, strel=strel, mode='erosion')
        result = fftmorphology(im=temp, strel=strel, mode='dilation')
    if mode.startswith('clos'):
        temp = fftmorphology(im=im, strel=strel, mode='dilation')
        result = fftmorphology(im=temp, strel=strel, mode='erosion')
    return result

def porosimetry(
        im,
        sizes: int = 25,
        inlets=None,
        access_limited: bool = True,
        mode: Literal['hybrid', 'dt', 'mio'] = 'hybrid',
        divs=1,
):
    """
    孔隙模拟算法，对输入二值图像im计算各处可被球侵润的最大半径。
    返回: 每点可被侵润最大球半径的数组。
    支持三种模式:
        - 'mio'   : 形态学开运算（慢但准确）
        - 'dt'    : 距离变换法（快速近似）
        - 'hybrid': 结合距离变换+膨胀（兼顾效率和边界效果）
    access_limited: 是否采用限定入口算法（仅入口可达区域才进行填充，常用于连通通道分析）。
    divs: 大于1时自动并行分块。
    """
    im = np.squeeze(im)
    # 去除输入图像im中所有维度为1的轴，将数组变为最紧凑的形状，例如(1, H, W)变成(H, W)
    dt = distance_transform_edt(im > 0)
    # 计算im > 0的二值图像的欧式距离变换（distance transform），即每个非零像素点到最近零像素点的距离，结果存储在dt中，表示距离场
    if inlets is None:
        inlets = get_border(im.shape, mode="faces")
        # 如果参数inlets为空，调用get_border函数获得图像边界的点集，mode="faces"指可能只考虑面（二维边界）上的位置
    if isinstance(sizes, int):
        # 如果sizes是整数类型，则根据距离场dt生成一组尺度阈值序列：“待检测的侵润球半径阈值集合”。其实就是指定分箱的数量
        # 这组阈值在dt的最大值到0之间，按对数均匀间隔生成，共sizes个
        sizes = np.logspace(start=np.log10(np.amax(dt)), stop=0, num=sizes)
    else:
        sizes = np.unique(sizes)[-1::-1]
        # 否则sizes被视为序列，将其去重并倒序排列（从大到小）

    # 选择结构元素类型，用于形态学操作
    if im.ndim == 2:
        strel = disk
        # 如果输入图像是二维的，结构元素选用disk（圆盘）
    else:
        strel = ball
        # 如果是更高维（如3D），结构元素选用ball（球体）

    parallel = False
    # 初始化并行标志设置为False

    if isinstance(divs, int):
        divs = [divs] * im.ndim
        # 如果divs是整数，扩展为长度为图像维度数量的列表，表示每个维度上的划分数量相同

    if max(divs) > 1:
        logger.info(f'Performing {insp.currentframe().f_code.co_name} in parallel')
        # 如果divs中最大值大于1，说明需要对图像划分多个区块，执行并行处理，
        # 参数divs用于控制在每个维度上将图像划分成多少块。如果用户只传入一个整数，比如2，
        # 表示所有维度都均匀划分为2块；代码会自动扩展为一个列表，如【2，2，2】（假设3维图像）。
        # 当至少有一个维度划分数超过1时，表示需要分块处理，这里程序会开启并行计算以提升处理效率。
        parallel = True  # 设置并行标志为True

    # "mio": morphology-based实现
    if mode == "mio":
        pw = int(np.floor(dt.max()))
        impad = np.pad(im, mode="symmetric", pad_width=pw)
        inlets_pad = np.pad(inlets, mode="symmetric", pad_width=pw)
        imresults = np.zeros(np.shape(impad))
        for r in tqdm(sizes, **settings.tqdm):
            # 先腐蚀，再膨胀做开操作
            if parallel:
                imtemp = chunked_func(func=fftmorphology, im=impad, strel=strel(int(r)), overlap=int(r)+1, mode='erosion', cores=settings.ncores, divs=divs)
            else:
                imtemp = fftmorphology(im=impad, strel=strel(int(r)), mode='erosion')
            if access_limited:
                imtemp = trim_disconnected_blobs(imtemp, inlets_pad, strel=strel(1))
            if parallel:
                imtemp = chunked_func(func=fftmorphology, im=imtemp, strel=strel(int(r)), overlap=int(r)+1, mode='dilation', cores=settings.ncores, divs=divs)
            else:
                imtemp = fftmorphology(im=imtemp, strel=strel(int(r)), mode='dilation')
            if np.any(imtemp):
                imresults[(imresults == 0) * imtemp] = r
        imresults = extract_subsection(imresults, shape=im.shape)
    # "dt": 距离变换+二次侵润
    elif mode == "dt":
        imresults = np.zeros(im.shape)
        for r in tqdm(sizes, **settings.tqdm):
            imtemp = dt >= r
            if access_limited:
                imtemp = trim_disconnected_blobs(imtemp, inlets, strel=strel(1))
            if np.any(imtemp):
                if parallel:
                    imtemp = chunked_func(func=distance_transform_edt, data=~imtemp, im_arg='data', overlap=int(r)+1, parallel=0, cores=settings.ncores, divs=divs) < r
                else:
                    imtemp = distance_transform_edt(~imtemp) < r
                imresults[(imresults == 0) * imtemp] = r
    # "hybrid": DT做初筛, 膨胀细化边界
    elif mode == "hybrid":
        imresults = np.zeros(im.shape)
        for r in tqdm(sizes, **settings.tqdm):
            imtemp = dt >= r
            if access_limited:
                imtemp = trim_disconnected_blobs(imtemp, inlets, strel=strel(1))
            if np.any(imtemp):
                if parallel:
                    imtemp = chunked_func(func=fftmorphology, mode='dilation',
                                          im=imtemp, strel=strel(int(r)),
                                          overlap=int(r)+1,
                                          cores=settings.ncores, divs=divs)
                else:
                    imtemp = fftmorphology(imtemp, strel(int(r)), mode="dilation")
                imresults[(imresults == 0) * imtemp] = r
    else:
        raise Exception("Unrecognized mode " + mode)
    return imresults
