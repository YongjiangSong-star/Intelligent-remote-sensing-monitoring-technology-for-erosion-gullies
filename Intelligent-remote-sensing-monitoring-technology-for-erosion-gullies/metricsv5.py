import csv
import cv2
from osgeo import gdal
import numpy as np
from osgeo import gdal, ogr, osr
from scipy.interpolate import griddata

from skimage.morphology import skeletonize, medial_axis
import rasterio
from rasterio.transform import Affine
from scipy.ndimage import gaussian_filter
import math
from shapely.geometry import Polygon
from shapely.ops import transform
import os
from osgeo import gdal
import matplotlib.pyplot as plt
from scipy import ndimage
from skimage.util import img_as_ubyte

gdal.DontUseExceptions()  # 禁用GDAL异常处理


def extract_contour_points(image_path):
    """从二值图像中提取轮廓点坐标（列x, 行y）"""
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError("无法加载图像，请检查文件路径是否正确。")
    # 二值化处理
    ret, thresh = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
    # 查找轮廓，使用CHAIN_APPROX_NONE获取所有点
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    points = []
    for contour in contours:
        # 将轮廓点转换为 (N, 2) 的数组
        contour_points = contour.reshape(-1, 2)

        # 确保轮廓闭合
        if len(contour_points) > 0:
            first_point = contour_points[0]
            last_point = contour_points[-1]
            if (first_point[0] != last_point[0]) or (first_point[1] != last_point[1]):
                contour_points = np.vstack([contour_points, [first_point[0], first_point[1]]])

        for point in contour_points:
            x_img, y_img = point[0], point[1]  # x为列号，y为行号
            points.append((x_img, y_img))
    return points


def convert_to_geocoords(dem_path, points):
    try:
        dem_dataset = gdal.Open(dem_path, gdal.GA_ReadOnly)
        if dem_dataset is None:
            raise ValueError("无法加载DEM文件")

        geotransform = dem_dataset.GetGeoTransform()
        band = dem_dataset.GetRasterBand(1)
        dem_array = band.ReadAsArray()

        output_data = []
        for x_img, y_img in points:
            if (0 <= x_img < dem_array.shape[1]) and (0 <= y_img < dem_array.shape[0]):
                x = geotransform[0] + x_img * geotransform[1] + y_img * geotransform[2]
                y = geotransform[3] + x_img * geotransform[4] + y_img * geotransform[5]
                z = dem_array[y_img, x_img]
                output_data.append([x, y, z])

        # 显式释放资源
        band = None
        dem_dataset = None

        return output_data
    except Exception as e:
        # 确保异常时也释放资源
        if 'dem_dataset' in locals():
            dem_dataset = None
        raise


def save_to_csv(data, csv_path):
    """保存点数据到CSV文件"""
    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['X', 'Y', 'Z'])
        writer.writerows(data)
    print(f"成功保存{len(data)}个点到{csv_path}")


# ------------------------------
# 1. 读取侵蚀沟边界点
# ------------------------------
def read_boundary_points(csv_path):
    """读取CSV文件中的边界点坐标 (X, Y)"""
    points = []
    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # 跳过标题行
        for row in reader:
            x, y = float(row[0]), float(row[1])
            points.append((x, y))
    return np.array(points)


# ------------------------------
# 2. 生成侵蚀沟掩膜
# ------------------------------
def create_erosion_mask(dsm_path, boundary_points):
    """基于边界点生成侵蚀沟区域的二值掩膜"""
    # 读取原始DSM信息
    dsm = gdal.Open(dsm_path)
    geotransform = dsm.GetGeoTransform()
    cols = dsm.RasterXSize
    rows = dsm.RasterYSize

    # 创建内存中的矢量多边形
    driver = ogr.GetDriverByName('Memory')
    ds = driver.CreateDataSource('')
    srs = osr.SpatialReference()
    srs.ImportFromWkt(dsm.GetProjection())
    layer = ds.CreateLayer('erosion', srs, ogr.wkbPolygon)

    # 将边界点转换为多边形 - 确保闭合
    ring = ogr.Geometry(ogr.wkbLinearRing)

    # 添加所有边界点
    for x, y in boundary_points:
        ring.AddPoint(x, y)

    # 检查并闭合环
    if len(boundary_points) > 0:
        first_point = boundary_points[0]
        last_point = boundary_points[-1]
        if (first_point[0] != last_point[0]) or (first_point[1] != last_point[1]):
            ring.AddPoint(first_point[0], first_point[1])

    poly = ogr.Geometry(ogr.wkbPolygon)
    poly.AddGeometry(ring)

    # 将多边形写入图层
    feature = ogr.Feature(layer.GetLayerDefn())
    feature.SetGeometry(poly)
    layer.CreateFeature(feature)

    # 生成掩膜栅格
    mask = gdal.GetDriverByName('MEM').Create('', cols, rows, 1, gdal.GDT_Byte)
    mask.SetGeoTransform(geotransform)
    mask.SetProjection(dsm.GetProjection())
    gdal.RasterizeLayer(mask, [1], layer, burn_values=[1])
    mask_array = mask.ReadAsArray()
    return mask_array.astype(bool)


# ------------------------------
# 3. 修复DSM高程（插值填充侵蚀沟）
# ------------------------------
def interpolate_dsm(dsm_path, mask):
    """使用周围地形插值修复掩膜区域的高程"""
    # 读取原始DSM数据
    dsm = gdal.Open(dsm_path)
    geotransform = dsm.GetGeoTransform()  # 获取地理变换参数
    band = dsm.GetRasterBand(1)
    dsm_array = band.ReadAsArray().astype(float)

    # 标记无效值（如NoData）
    nodata = band.GetNoDataValue()
    if nodata is not None:
        dsm_array[dsm_array == nodata] = np.nan

    # 提取掩膜外的点坐标和高程
    rows, cols = np.where(~mask)
    x_coords = geotransform[0] + cols * geotransform[1] + rows * geotransform[2]
    y_coords = geotransform[3] + cols * geotransform[4] + rows * geotransform[5]
    valid_points = np.column_stack([x_coords, y_coords])
    valid_values = dsm_array[~mask]

    # 移除潜在的异常值（例如低于某个阈值的值）
    threshold = -100  # 假设低于-100的值是异常值
    valid_indices = valid_values > threshold
    valid_points = valid_points[valid_indices]
    valid_values = valid_values[valid_indices]

    # 提取掩膜内的点坐标（需要插值的位置）
    rows_mask, cols_mask = np.where(mask)
    x_target = geotransform[0] + cols_mask * geotransform[1] + rows_mask * geotransform[2]
    y_target = geotransform[3] + cols_mask * geotransform[4] + rows_mask * geotransform[5]
    target_points = np.column_stack([x_target, y_target])

    # 使用反距离加权插值（IDW）进行初步插值
    filled_idw = griddata(
        valid_points, valid_values, target_points,
        method='linear', fill_value=np.nanmean(valid_values)
    )

    # 使用最近邻插值填充剩余NaN
    filled_nearest = griddata(valid_points, valid_values, target_points, method='nearest')

    # 结合两种插值结果
    filled_values = np.where(np.isnan(filled_idw), filled_nearest, filled_idw)

    # 如果仍有NaN，使用全局均值
    if np.isnan(filled_values).any():
        global_mean = np.nanmean(valid_values)
        filled_values = np.nan_to_num(filled_values, nan=global_mean)

    # 更新DSM数组并处理NaN
    dsm_reconstructed = dsm_array.copy()
    dsm_reconstructed[mask] = filled_values

    # 处理非-9999的负数
    negative_mask = (dsm_reconstructed < 0)
    if np.any(negative_mask):
        dsm_reconstructed[negative_mask] = dsm_array[negative_mask]

    negative_mask2 = (dsm_array< 0)
    if np.any(negative_mask2):
        dsm_reconstructed[negative_mask2] = dsm_array[negative_mask2]

    # # 检查并处理复数
    # complex_mask = np.iscomplex(dsm_reconstructed) & (dsm_reconstructed.imag != 0)
    # if np.any(complex_mask):
    #     dsm_reconstructed[complex_mask] = dsm_array[complex_mask]

    # 确保无残留NaN
    dsm_reconstructed = np.nan_to_num(dsm_reconstructed, nan=nodata if nodata is not None else -9999)

    return dsm_reconstructed


# ------------------------------
# 4. 保存修复后的DSM为GeoTIFF
# ------------------------------
def save_reconstructed_dsm(dsm_path, output_path, reconstructed_array):
    """将修复后的DSM保存为GeoTIFF"""
    # 读取原始DSM的元数据
    dsm = gdal.Open(dsm_path)
    driver = dsm.GetDriver()
    cols = dsm.RasterXSize
    rows = dsm.RasterYSize

    # 创建输出文件
    out_ds = driver.Create(output_path, cols, rows, 1, gdal.GDT_Float32)
    out_ds.SetGeoTransform(dsm.GetGeoTransform())
    out_ds.SetProjection(dsm.GetProjection())

    # 写入数据并设置NoData值
    out_band = out_ds.GetRasterBand(1)
    out_band.WriteArray(reconstructed_array)
    #out_band.SetNoDataValue(dsm.GetRasterBand(1).GetNoDataValue())
    out_band.FlushCache()
    out_ds = None


    # 体积计算
    # 启用异常处理
    gdal.UseExceptions()
def calculate_volume_arcgis_style(before_dsm_path, after_dsm_path, plane_height=None):

    # 读取基准面DSM
    before_dsm = gdal.Open(before_dsm_path)
    before_band = before_dsm.GetRasterBand(1)
    before_array = before_band.ReadAsArray().astype(np.float32)
    nodata = before_band.GetNoDataValue()

    # 获取地理参数
    geotransform = before_dsm.GetGeoTransform()
    pixel_width = abs(geotransform[1])
    pixel_height = abs(geotransform[5])
    pixel_area = pixel_width * pixel_height

    # 处理NoData值
    before_array = np.where(before_array == nodata, np.nan, before_array)

    # 计算高度差
    if plane_height is not None:
        # 平面模式（类似ArcGIS中的基准平面）
        diff = before_array - plane_height
    else:
        # 两个表面之间模式
        after_dsm = gdal.Open(after_dsm_path)
        after_band = after_dsm.GetRasterBand(1)
        after_array = after_band.ReadAsArray().astype(np.float32)
        after_nodata = after_band.GetNoDataValue()
        after_array = np.where(after_array == after_nodata, np.nan, after_array)
        diff = after_array - before_array  # 注意方向与ArcGIS一致

    # ArcGIS体积计算核心逻辑
    volume = np.nansum(diff) * pixel_area

    # 修改：将体积改为绝对值
    return abs(volume)

def calculate_erosion_metrics(mask_path, dsm_path, volume=1000):
    # 读取二值掩膜图
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    _, binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    # 从DSM图像获取单位像素面积
    with rasterio.open(dsm_path) as src:
        transform = src.transform
        x_res = src.res[0]
        y_res = abs(src.res[1])
        pixel_area = x_res * y_res

    # 计算实际面积
    area_pixel = cv2.countNonZero(binary)
    area_real = area_pixel * pixel_area

    # 修改：将平均深度改为绝对值
    depth_average = abs(volume) / area_real if area_real > 0 else 0

    # ----【修改点①】骨架法计算所有沟壑的总长度----
    binary_image_smoothed = cv2.GaussianBlur(binary, (57, 57), 0)
    binary_for_skeleton = binary_image_smoothed // 255  # 转为 0/1 二值图像
    #skeleton = extract_main_gully_skeleton(binary_for_skeleton , min_branch_length=10)
    skeleton = skeletonize(binary_for_skeleton)  # 全局骨架化
    # plt.figure
    # plt.imshow(binary_image_smoothed)
    # plt.show()
    # plt.figure
    # plt.imshow(skeleton)
    # plt.show()


    skeleton_length_total = np.sum(skeleton) * np.sqrt(pixel_area)  # 单位长度 × 像素数

    average_width = area_real / skeleton_length_total if skeleton_length_total > 0 else 0

    # ----【修改点②】提取所有轮廓并计算总周长----
    ds = gdal.Open(dsm_path)
    geotransform = ds.GetGeoTransform()
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    if mask.shape[:2] != (ds.RasterYSize, ds.RasterXSize):
        mask = cv2.resize(mask, (ds.RasterXSize, ds.RasterYSize))
        print(f"[掩模处理] 已调整尺寸至DSM大小: {mask.shape}")

    # 二值化 + 模糊去锯齿
    _, binary_mask = cv2.threshold(mask, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    smoothed_mask = cv2.GaussianBlur(binary_mask.astype(np.float32), (5, 5), 0)
    smoothed_mask = (smoothed_mask > 0.5).astype(np.uint8)

    # 提取所有轮廓
    contours, _ = cv2.findContours(smoothed_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    total_perimeter = 0.0
    all_polygons = []

    for cnt in contours:
        # 转换为地理坐标
        cnt = cnt.squeeze()
        x_coords = geotransform[0] + cnt[:, 0] * geotransform[1]
        y_coords = geotransform[3] + cnt[:, 1] * geotransform[5]

        coords = list(zip(x_coords, y_coords))
        if len(coords) >= 4:  # 至少构成一个面
            polygon = Polygon(coords)
            simplified = polygon.simplify(2.0, preserve_topology=True)
            total_perimeter += simplified.length
            all_polygons.append(simplified)

    return {
        "面积 (m²)": area_real,
        "周长 (m)": total_perimeter,
        "平均宽度 (m)": average_width,
        "骨架长度 (m)": skeleton_length_total,
        "平均深度 (m)": depth_average
    }


def calculate_gully_slope(mask_path, dsm_path, filter_sigma=1.0, max_valid_slope=60):
    try:
        # 读取DSM数据
        with rasterio.open(dsm_path) as dsm_src:
            dsm = dsm_src.read(1)
            transform = dsm_src.transform
            crs = dsm_src.crs
            nodata = dsm_src.nodata

        # 读取掩膜数据
        if mask_path.lower().endswith(('.tif', '.tiff')):
            with rasterio.open(mask_path) as mask_src:
                mask = mask_src.read(1)
                if mask_src.crs != crs or mask_src.transform != transform:
                    print("警告：掩膜与DSM坐标系或地理变换不一致！")
        else:
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            _, mask = cv2.threshold(mask, 127, 1, cv2.THRESH_BINARY)

        # 提取侵蚀沟区域高程数据
        eroded_dsm = np.where(mask, dsm, np.nan)
        if nodata is not None:
            eroded_dsm = np.where(dsm == nodata, np.nan, eroded_dsm)

        # 高斯滤波降噪
        if filter_sigma > 0:
            eroded_dsm = gaussian_filter(eroded_dsm, sigma=filter_sigma, mode='nearest')

        # 计算分辨率
        x_res = transform.a
        y_res = abs(transform.e)

        # 计算梯度
        grad_y, grad_x = np.gradient(eroded_dsm)
        grad_x /= x_res
        grad_y /= y_res

        # 计算坡度角度（度）
        slope_ratio = np.sqrt(grad_x ** 2 + grad_y ** 2)
        slope_degree = np.degrees(np.arctan(slope_ratio))

        # 提取有效区域坡度
        valid_mask = ~np.isnan(slope_degree) & (slope_degree <= max_valid_slope)
        valid_slopes = slope_degree[valid_mask]

        # 确保返回标准的Python浮点数
        if valid_slopes.size > 0:
            slope_value = float(np.nanmean(valid_slopes))
        else:
            slope_value = 0.0

        return {'坡度': slope_value}

    except Exception as e:
        print(f"坡度计算失败：{str(e)}")
        return {'坡度': 0.0}  # 返回默认值而不是None


# 添加

def calculate_metrics(mask_path, dsm_path, output_dir):
    """主计算函数，返回所有参数结果（保留两位小数）"""
    # 1. 提取轮廓点
    points = extract_contour_points(mask_path)

    # 2. 转换为地理坐标并获取高程
    geocoords_data = convert_to_geocoords(dsm_path, points)

    # 3. 保存边界点CSV
    boundary_csv = os.path.join(output_dir, "boundary_points.csv")
    save_to_csv(geocoords_data, boundary_csv)

    # 4. 读取边界点
    points = read_boundary_points(boundary_csv)

    # 5. 生成掩膜
    mask = create_erosion_mask(dsm_path, points)

    # 6. 插值修复DSM
    dsm_reconstructed = interpolate_dsm(dsm_path, mask)

    # 7. 保存修复后的DSM
    dsm_recovery = os.path.join(output_dir, "reconstructed_dsm.tif")
    save_reconstructed_dsm(dsm_path, dsm_recovery, dsm_reconstructed)

    # 8. 计算体积
    volume_result = calculate_volume_arcgis_style(dsm_path, dsm_recovery)

    # 9. 计算形态参数
    metrics1 = calculate_erosion_metrics(mask_path, dsm_path, volume=volume_result)

    # 10. 计算坡度
    metrics2 = calculate_gully_slope(mask_path, dsm_path)

    # 确保坡度值正确
    slope_value = metrics2.get('坡度', 0.0)

    # 确保坡度是标量数值
    if isinstance(slope_value, (list, np.ndarray)):
        slope_value = float(slope_value[0]) if slope_value.size > 0 else 0.0
    elif isinstance(slope_value, np.generic):
        slope_value = float(slope_value)

    # 合并所有结果并保留两位小数
    all_metrics = {
        "体积 (立方米)": round(volume_result, 2),
        "面积 (平方米)": round(metrics1["面积 (m²)"], 2),
        "周长 (m)": round(metrics1["周长 (m)"], 2),
        "平均宽度 (m)": round(metrics1["平均宽度 (m)"], 2),
        "骨架长度 (m)": round(metrics1["骨架长度 (m)"], 2),
        "平均深度 (m)": round(metrics1["平均深度 (m)"], 2),
        "坡度(°)": round(slope_value, 2)  # 确保坡度被正确处理
    }

    return all_metrics
