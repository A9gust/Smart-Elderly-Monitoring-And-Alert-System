# extract patches based on keypoints
import torch
import math, cv2
import numpy as np

def norm(x):
    y = ((x - np.min(x)) / (np.max(x) - np.min(x))) * 255
    return y

def gen_kernel(size, sigma):
    kernel = np.fromfunction(lambda x, y: (1/(2*math.pi*sigma**2)) * math.e ** ((-1*((x-(size-1)/2)**2+(y-(size-1)/2)**2))/(2*sigma**2)), (size, size))
    kernel = kernel / np.sum(kernel)
    kernel = (kernel - np.min(kernel)) / (np.max(kernel) - np.min(kernel))
    return kernel

def extract_patches(img, kpts, kernel_size = 128, kernel_sigma=0.3, scale=1/4):
    img_shape = img.shape # 1080 x 1920 x 3 video resolution
    pad_img = np.zeros((img_shape[0]+kernel_size*2, img_shape[1]+kernel_size*2, 3))
    pad_img[kernel_size:-kernel_size, kernel_size:-kernel_size, :] = img
    kernel = gen_kernel(kernel_size,kernel_size*kernel_sigma)
    kernel = np.expand_dims(kernel,2).repeat(3,axis=2)
    kpts = np.delete(kpts, [[1],[-3],[-4]], axis=0) # 
    patches = np.zeros((15,math.ceil(scale*kernel_size),math.ceil(scale*kernel_size),3))
    for idx in range(15):
        tmp = norm(pad_img[int(kpts[idx,1]+0.5*kernel_size):int(kpts[idx,1]+1.5*kernel_size), int(kpts[idx,0]+0.5*kernel_size):int(kpts[idx,0]+1.5*kernel_size),:] * kernel)
        tmp = cv2.resize(tmp, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
        patches[idx,:,:,:] = tmp
    return patches

if __name__ == "__main__":
    img = cv2.imread('sample_2.jpg')
    kpts = np.array([
    [817, 340],
    [766, 437],
    [658, 447],
    [658, 589],
    [890, 492],
    [884, 429],
    [1007, 572],
    [1000, 476],
    [750, 706],
    [759, 869],
    [-1, -1],
    [907, 699],
    [932, 842],
    [916, 1015],
    [792, 311],
    [834, 319],
    [741, 320],
    [851, 344]
    ])
    res = extract_patches(img, kpts, kernel_size = 128, kernel_sigma=0.3, scale=1/4)
    arr = np.transpose(res, (0, 3, 1, 2))
    repeated = np.stack([arr] * 30, axis=0)
    temp = np.stack([repeated] * 2, axis=0)
    print(temp.shape)
    # Convert to torch tensor
    temp_tensor = torch.from_numpy(temp)

    # Save to file
    torch.save(temp_tensor, 'temp_tensor.pt')

    kpts_torch = np.stack([kpts] * 30, axis=0)
    kpts_res = np.stack([kpts_torch] * 2, axis=0)
    print(kpts_res.shape)
    kpts_tensor = torch.from_numpy(kpts_res)
    torch.save(kpts_tensor, 'kpts_tensor.pt')