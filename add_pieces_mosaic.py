from parameters import *
import numpy as np
import pdb
import timeit

def get_mask_hexagon(h, w):
    start = int(w/3)
    end = int(2*w/3)
    mid = int(h/2)
    mask = np.zeros((h, w, 1))

    for i in range(mid):
        for j in range(max(start, 0), min(end, w)):
            mask[i, j] = 1
            mask[h-i-1, j] = 1
        start -= 1
        end += 1
    return mask

def get_mean_color_small_images(params:Parameters, c):
    N, H, W, C = params.small_images.shape
    mean_colors = np.zeros((N, c))
    for i in range(N):
        current_image = params.small_images[i].copy()
        for ch in range(c):
            mean_colors[i, ch] = current_image[:, :, ch].mean()
    return mean_colors

def get_sorted_distances(mean_color_patch, mean_color_pieces):
    dist = np.sum((mean_color_pieces - mean_color_patch) ** 2, axis = 1)
    return np.argsort(dist)

def add_pieces_grid(params: Parameters):
    start_time = timeit.default_timer()
    img_mosaic = np.zeros(params.image_resized.shape, np.uint8)
    N, H, W, C = params.small_images.shape
    h, w, c = params.image_resized.shape
    num_pieces = params.num_pieces_vertical * params.num_pieces_horizontal

    if params.criterion == 'aleator':
        for i in range(params.num_pieces_vertical):
            for j in range(params.num_pieces_horizontal):
                index = np.random.randint(low=0, high=N, size=1)
                img_mosaic[i*H:(i+1)*H, j*W:(j+1)*W, :] = params.small_images[index]

        #print('Building mosaic %.2f%%' % (100 * (i * params.num_pieces_horizontal + j + 1) / num_pieces))

    elif params.criterion == 'distantaCuloareMedie':
        mean_color_pieces = get_mean_color_small_images(params, 3)

        for i in range(params.num_pieces_vertical):
            for j in range(params.num_pieces_horizontal):
                patch = params.image_resized[i*H:(i+1)*H, j*W:(j+1)*W, :]
                mean_color_patch = np.mean(patch, axis=(0, 1))
                dist = get_sorted_distances(mean_color_patch, mean_color_pieces)
                index = dist[0]
                img_mosaic[i*H:(i+1)*H, j*W:(j+1)*W, :] = params.small_images[index]
        #print('Building mosaic %.2f%%' % (100 * (i * params.num_pieces_horizontal + j) / num_pieces))
    else:
        print('Error! unknown option %s' % params.criterion)
        exit(-1)

    end_time = timeit.default_timer()
    print('Running time: %f s.' % (end_time - start_time))

    return img_mosaic


def add_pieces_random(params: Parameters):
    start_time = timeit.default_timer()
    end_time = timeit.default_timer()
    print('running time:', (end_time - start_time), 's')
    return None


"""def add_pieces_hexagon(params: Parameters):
    start_time = timeit.default_timer()
    N, H, W, C = params.small_images.shape
    h, w, c = params.image_resized.shape
    mask = get_mask_hexagon(H, W)
    bigger_image=np.zeros((h+2*H, w+2*W, c), np.uint8)
    bigger_image[H:-H, W:-W, :] = params.image_resized
    img_mosaic = np.zeros(params.image_resized.shape, np.uint8)
    mean_color_pieces = get_mean_color_small_images(params, c)

    for i in range(H//2, bigger_image.shape[0]-H, H):
        for j in range(0, bigger_image.shape[1]-W, W+W//3):
            patch = bigger_image[i:i+H, j:j+W, :]
            mean_patch = np.mean(patch, axis=(0,1))
            dist = get_sorted_distances(mean_patch, mean_color_pieces)
            index=dist[0]
            img_mosaic[i:i+H, j:j+W, :] = img_mosaic[i:i+H, j:j+W, :]*(1-mask) + params.small_images[index]*mask

    for i in range(0, bigger_image.shape[0]-H, H):
        for j in range(2*W//3, bigger_image.shape[1]-W, W+W//3):
            patch = bigger_image[i:i+H, j:j+W, :]
            mean_patch = np.mean(patch, axis=(0,1))
            dist = get_sorted_distances(mean_patch, mean_color_pieces)
            index=dist[0]
            img_mosaic[i:i+H, j:j+W, :] = img_mosaic[i:i+H, j:j+W, :]*(1-mask) + params.small_images[index]*mask

    end_time = timeit.default_timer()
    print('running time:', (end_time - start_time), 's')
    return img_mosaic
"""


def add_pieces_hexagon(params: Parameters):
    N, H, W, C = params.small_images.shape
    h, w, c = params.image_resized.shape
    mask = get_mask_hexagon(H, W)
    bigger_image = np.zeros((h + 2 * H, w + 2 * W, c), np.uint8)
    bigger_image[H:-H, W:-W, :] = params.image_resized
    img_mosaic = np.zeros(bigger_image.shape, np.uint8)
    mean_color_pieces = get_mean_color_small_images(params, c)
    for i in range(H // 2, bigger_image.shape[0] - H, H):
        for j in range(0, bigger_image.shape[1] - W, W + W // 3):
            patch = bigger_image[i:i + H, j:j + W, :]
            mean_patch = np.mean(patch, axis=(0, 1))
            dist = get_sorted_distances(mean_patch, mean_color_pieces)
            index = dist[0]
            img_mosaic[i:i + H, j:j + W, :] = img_mosaic[i:i + H, j:j + W, :] * (1 - mask) + mask * params.small_images[
                index]

    for i in range(0, bigger_image.shape[0] - H, H):
        for j in range(2 * W // 3, bigger_image.shape[1] - W, W + W // 3):
            patch = bigger_image[i:i + H, j:j + W, :]
            mean_patch = np.mean(patch, axis=(0, 1))
            dist = get_sorted_distances(mean_patch, mean_color_pieces)
            index = dist[0]
            img_mosaic[i:i + H, j:j + W, :] = img_mosaic[i:i + H, j:j + W, :] * (1 - mask) + mask * params.small_images[
                index]

    img_mosaic = img_mosaic[H:-H, W:-W, :]

    start_time = timeit.default_timer()
    end_time = timeit.default_timer()
    print('running time:', (end_time - start_time), 's')
    return img_mosaic
