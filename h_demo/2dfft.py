import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from skimage.metrics import structural_similarity as compare_ssim

def enhance_difference_image(difference):
    log_difference = np.log1p(difference)
    log_difference_normalized = cv2.normalize(log_difference, None, 0, 255, cv2.NORM_MINMAX)
    log_difference_normalized = np.uint8(log_difference_normalized)
    difference_colored = cv2.applyColorMap(log_difference_normalized, cv2.COLORMAP_JET)
    return difference_colored

def apply_gaussian_window(image):
    rows, cols = image.shape
    gauss_window_row = cv2.getGaussianKernel(rows, 0.8*rows)
    gauss_window_col = cv2.getGaussianKernel(cols, 0.8*cols)
    gauss_window_2d = gauss_window_row * gauss_window_col.T
    windowed_image = image * gauss_window_2d
    return windowed_image

def apply_kaiser_window(image, beta=9):
    rows, cols = image.shape
    kaiser_window_row = np.kaiser(rows, beta)
    kaiser_window_col = np.kaiser(cols, beta)
    kaiser_window_2d = np.outer(kaiser_window_row, kaiser_window_col)
    windowed_image = image * kaiser_window_2d
    return windowed_image

def detect_periodicity_and_save(input_folder):
    image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
    image_files.sort()
    previous_magnitude_spectrum = None

    # 初始化 DataFrame 来存储统计数据
    stats_df = pd.DataFrame(columns=["Image", "Mean", "Variance", "Max", "Min"])

    for image_file in image_files:
        image_path = os.path.join(input_folder, image_file)
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            print(f"无法读取图像文件: {image_path}")
            continue

        _, binary_image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        windowed_image = apply_kaiser_window(binary_image, beta=14)

        f_transform = np.fft.fft2(windowed_image)
        f_shift = np.fft.fftshift(f_transform)
        magnitude_spectrum = np.log(np.abs(f_shift) + 1)

        output_image_path = os.path.join(input_folder, f"magnitude_spectrum_{image_file}.png")
        plt.imsave(output_image_path, magnitude_spectrum, cmap='gray')
        print(f"Processed and saved magnitude spectrum: {output_image_path}")

        if previous_magnitude_spectrum is not None:
            if previous_magnitude_spectrum.shape != magnitude_spectrum.shape:
                print(f"频谱图大小不匹配，跳过SSIM计算: {image_file}")
            else:
                ssim_value, _ = compare_ssim(previous_magnitude_spectrum, magnitude_spectrum, full=True, data_range=magnitude_spectrum.max() - magnitude_spectrum.min())
                print(f"SSIM between {image_file} and previous image: {ssim_value}")

                difference = np.abs(magnitude_spectrum - previous_magnitude_spectrum)

                difference_colored = enhance_difference_image(difference)
                difference_image_path = os.path.join(input_folder, f"difference_{image_file}.png")
                cv2.imwrite(difference_image_path, difference_colored)
                print(f"Processed and saved difference image: {difference_image_path}")

                # 振幅差异的统计分析
                difference_mean = np.mean(difference)
                difference_variance = np.var(difference)
                difference_max = np.max(difference)
                difference_min = np.min(difference)

                print(f"Amplitude Difference Statistics for {image_file}:")
                print(f"Mean: {difference_mean}, Variance: {difference_variance}, Max: {difference_max}, Min: {difference_min}")

                # 将统计数据添加到 DataFrame 中
                stats_df = pd.concat([stats_df, pd.DataFrame({
                    "Image": [image_file],
                    "Mean": [difference_mean],
                    "Variance": [difference_variance],
                    "Max": [difference_max],
                    "Min": [difference_min]
                })], ignore_index=True)

                # 振幅差异的直方图分布
                plt.figure()
                plt.hist(difference.ravel(), bins=50, color='blue', alpha=0.7)
                plt.title(f"Amplitude Difference Histogram - {image_file}")
                plt.xlabel("Amplitude Difference")
                plt.ylabel("Frequency")
                histogram_image_path = os.path.join(input_folder, f"histogram_{image_file}.png")
                plt.savefig(histogram_image_path)
                plt.close()
                print(f"Saved amplitude difference histogram: {histogram_image_path}")

        previous_magnitude_spectrum = magnitude_spectrum

    # 将统计数据保存到 Excel 文件
    excel_output_path = os.path.join(input_folder, "amplitude_difference_stats.xlsx")
    stats_df.to_excel(excel_output_path, index=False)
    print(f"Saved amplitude difference statistics to Excel: {excel_output_path}")

# 使用示例
if __name__ == "__main__":
    input_folder = r"D:\Desktop\images"
    detect_periodicity_and_save(input_folder)

