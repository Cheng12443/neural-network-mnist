#!/usr/bin/env python3
"""
数据集下载脚本
用于下载MNIST数据集到本地MNIST目录
"""

import os
import gzip
import urllib.request
import shutil
from pathlib import Path

def download_file(url, filename):
    """下载文件并显示进度"""
    print(f"正在下载 {filename}...")
    urllib.request.urlretrieve(url, filename)
    print(f"下载完成: {filename}")

def extract_gz(gz_file, output_file):
    """解压gz文件"""
    print(f"正在解压 {gz_file}...")
    with gzip.open(gz_file, 'rb') as f_in:
        with open(output_file, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
    os.remove(gz_file)
    print(f"解压完成: {output_file}")

def main():
    """主函数：下载并处理MNIST数据集"""
    # 创建MNIST目录
    mnist_dir = Path("MNIST")
    mnist_dir.mkdir(exist_ok=True)
    
    # MNIST数据集URL
    base_url = "http://yann.lecun.com/exdb/mnist/"
    files = {
        "train-images-idx3-ubyte.gz": "train-images-idx3-ubyte",
        "train-labels-idx1-ubyte.gz": "train-labels-idx1-ubyte",
        "t10k-images-idx3-ubyte.gz": "t10k-images-idx3-ubyte",
        "t10k-labels-idx1-ubyte.gz": "t10k-labels-idx1-ubyte"
    }
    
    print("开始下载MNIST数据集...")
    
    for gz_file, output_file in files.items():
        gz_path = mnist_dir / gz_file
        output_path = mnist_dir / output_file
        
        if output_path.exists():
            print(f"文件已存在，跳过: {output_file}")
            continue
            
        # 下载
        url = base_url + gz_file
        download_file(url, str(gz_path))
        
        # 解压
        extract_gz(str(gz_path), str(output_path))
    
    print("MNIST数据集下载完成！")
    print("\n文件列表:")
    for file in mnist_dir.iterdir():
        if file.is_file():
            size_mb = file.stat().st_size / (1024 * 1024)
            print(f"  {file.name}: {size_mb:.1f} MB")

if __name__ == "__main__":
    main()