import numpy as np
from matplotlib.image import imread
from PIL import Image

def inner_product_rows(x):
    m = x.shape[0]
    y = np.zeros(m)
    for i in range(m):
        y[i] = x[i].dot(x[i])

    return y

def main(K, small_img_path, small_save_path, large_img_path, large_save_path):
    img = imread(small_img_path).astype(np.float64)
    h, w = img.shape[0], img.shape[1]
    c = np.zeros((h, w), dtype=np.int32)
    rand_row = np.random.choice(h, K, replace=False)
    rand_col = np.random.choice(w, K, replace=False)
    centroids = np.array([img[i, j] for i, j in zip(rand_row, rand_col)])
    cnts = None 
    it = 0
    converge = False
    while it < 30 or not converge:
        converge = True
        cnts = np.zeros(K, dtype=np.int32)
        centroids_ = np.zeros_like(centroids)
        for i in range(h):
            for j in range(w):
                ci = np.argmin(inner_product_rows(img[i, j]-centroids))
                if np.linalg.norm(ci - c[i, j]) > 1e-5 : converge = False
                cnts[ci] += 1 
                c[i, j] = ci
                centroids_[ci] += img[i, j]
        print(f'Iteration : {it}')
        print("cnts : ", cnts)
        centroids = centroids_ / cnts[:, np.newaxis]
        it += 1
    
    centroids_uint8 = centroids.astype(np.uint8)
    compress_img = np.zeros_like(img, dtype=np.uint8)
    for i in range(h):
        for j in range(w):
            compress_img[i, j] = centroids_uint8[c[i, j]]

    Image.fromarray(compress_img).save(small_save_path)

    img_large = imread(large_img_path).astype(np.float64)
    compress_img_large = np.zeros_like(img_large, dtype=np.uint8)
    h, w = img_large.shape[:2]
    for i in range(h):
        for j in range(w):
            ci = np.argmin(inner_product_rows(img_large[i, j]-centroids))
            compress_img_large[i, j] = centroids_uint8[ci]

    Image.fromarray(compress_img_large).save(large_save_path)
    

if __name__ == '__main__':
    k = 16
    pattern_img = '../data/peppers-{}.tiff'
    pattern_save = 'output/p05_peppers-{}.tiff'
    main(k, pattern_img.format('small'), pattern_save.format('small'), pattern_img.format('large'), pattern_save.format('large'))