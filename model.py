import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Net(nn.Module):
    def __init__(self,input_dim, args):

        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, args.nChannel, kernel_size=3, stride=1, padding=1, padding_mode='reflect')
        self.bn1 = nn.BatchNorm2d(args.nChannel)
        self.conv2 = nn.ModuleList()
        self.bn2 = nn.ModuleList()
        for i in range(args.nConv-1):
            self.conv2.append( nn.Conv2d(args.nChannel, args.nChannel, kernel_size=3, stride=1, padding=1, padding_mode='reflect') )
            self.bn2.append( nn.BatchNorm2d(args.nChannel) )
        self.upsample = nn.Upsample(scale_factor=2,mode = 'bicubic')
        self.conv3 = nn.Conv2d(args.nChannel, args.nChannel, kernel_size=1, stride=1, padding=0 )
        self.bn3 = nn.BatchNorm2d(args.nChannel)
        self.nConv = args.nConv

    def forward(self, x):
        # x = self.upsample(x)
        x = self.conv1(x)
        x = F.relu( x )
        x = self.bn1(x)
        for i in range(self.nConv-1):
            x = self.conv2[i](x)
            x = F.relu( x )
            x = self.bn2[i](x)
        x = self.conv3(x)
        x = self.bn3(x)
        return x

def detect_boundary(img):

    img = np.pad(img,((1,1),(1,1)),'constant',constant_values=100)
    m, n = img.shape
    return_img = np.zeros((m-2,n-2))

    for iii in range(1,m-1):
        for jjj in range(1,n-1):
            if img[iii,jjj] == 1:
                if img[iii - 1, jjj] == 0 or img[iii, jjj - 1] == 0 or img[iii + 1, jjj] == 0 or img[iii, jjj + 1] == 0:
                    return_img[iii-1, jjj-1] = 1

    return return_img

def remove_artifacts(Embedding_data,im_target,args):

    Embedding_data = Embedding_data.reshape(-1,args.n_components)
    m,n,_ = args.input_shape

    new_im_target = im_target.copy()

    im_targett = [int(i) for i in im_target]

    background = np.argmax(np.bincount(im_targett))

    to_img = np.where(im_target == background, 1, 0)
    to_img = to_img.reshape(m, n)

    return_img = np.zeros((m, n))

    for eponch in range(4):
        img_pad = np.pad(to_img, ((1, 1), (1, 1)), 'constant', constant_values=100)
        print(img_pad)
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if img_pad[i, j] == 0:
                    if img_pad[i + 1, j] != 0 or img_pad[i - 1, j] != 0 or img_pad[i, j + 1] != 0 or img_pad[
                        i, j - 1] != 0:
                        return_img[i - 1, j - 1] = 1

        to_img = to_img + return_img

    li = list(set(list(im_target)))
    subset_1 = []
    subset_2 = []
    for i in li:
        to_img = np.where(im_target == i, 1, 0)

        num_1 = np.sum(to_img.reshape(-1, 1)[:m * n // 2] * return_img.reshape(-1, 1)[:m * n // 2])
        num_2 = np.sum(to_img.reshape(-1, 1)[m * n // 2:] * return_img.reshape(-1, 1)[m * n // 2:])

        if num_1 / np.sum(to_img.reshape(-1, 1)[:m * n // 2]) > 0.5:
            subset_1.append(i)
            subset_2.append(1)
        if num_2 / np.sum(to_img.reshape(-1, 1)[m * n // 2:]) > 0.5:
            subset_1.append(i)
            subset_2.append(2)

    for kk in range(20):
        im_target = new_im_target

        im_target2 = im_target.copy()
        im_target3 = im_target.copy()
        uu = 0

        for i in list(set(list(im_target))):
            if uu < len(subset_1):
                if i in subset_1 and subset_2[uu] == 1:
                    im_target2[m * n // 2:] = 0
                    to_img = np.where(im_target2 == i, 1, 0)
                    to_img = to_img.reshape(args.input_shape[0], args.input_shape[1])

                    to_img = detect_boundary(to_img)

                    to_img = np.pad(to_img, ((1, 1), (1, 1)), 'constant', constant_values=100)
                    m, n = to_img.shape
                    for j in range(1, m - 1):
                        for jj in range(1, n - 1):
                            cluster = np.zeros((4, args.n_components)) + 1000

                            if to_img[j, jj] == 1:
                                if to_img[j - 1, jj] == 0:
                                    cluster[0] = np.mean(
                                        Embedding_data[np.where(im_target == im_target.reshape(args.input_shape[0], args.input_shape[1])[j - 1 - 1, jj - 1])[0]],
                                        axis=0)

                                if to_img[j, jj + 1] == 0:
                                    cluster[1] = np.mean(
                                        Embedding_data[np.where(im_target == im_target.reshape(args.input_shape[0], args.input_shape[1])[j - 1, jj + 1 - 1])[0]],
                                        axis=0)
                                if to_img[j + 1, jj] == 0:
                                    cluster[2] = np.mean(
                                        Embedding_data[np.where(im_target == im_target.reshape(args.input_shape[0], args.input_shape[1])[j - 1 + 1, jj - 1])[0]],
                                        axis=0)
                                if to_img[j, jj - 1] == 0:
                                    cluster[3] = np.mean(
                                        Embedding_data[np.where(im_target == im_target.reshape(args.input_shape[0], args.input_shape[1])[j - 1, jj - 1 - 1])[0]],
                                        axis=0)

                                u = np.sum((cluster - Embedding_data.reshape(args.input_shape[0],args.input_shape[1], args.n_components)[j - 1, jj - 1]) ** 2, axis=1)
                                u = np.argmin(u)

                                if u == 0:
                                    new_im_target.reshape(args.input_shape[0], args.input_shape[1])[j - 1, jj - 1] = im_target.reshape(args.input_shape[0], 107)[
                                        j - 1 - 1, jj - 1]
                                if u == 1:
                                    new_im_target.reshape(args.input_shape[0], args.input_shape[1])[j - 1, jj - 1] = im_target.reshape(args.input_shape[0], args.input_shape[1])[
                                        j - 1, jj - 1 + 1]
                                if u == 2:
                                    new_im_target.reshape(args.input_shape[0], args.input_shape[1])[j - 1, jj - 1] = im_target.reshape(args.input_shape[0], args.input_shape[1])[
                                        j - 1 + 1, jj - 1]
                                if u == 3:
                                    new_im_target.reshape(args.input_shape[0], args.input_shape[1])[j - 1, jj - 1] = im_target.reshape(args.input_shape[0], args.input_shape[1])[
                                        j - 1, jj - 1 - 1]
                    uu += 1
            if uu < len(subset_2):
                if i in subset_1 and subset_2[uu] == 2:
                    im_target3[:m * n // 2] = 0
                    to_img = np.where(im_target3 == i, 1, 0)
                    to_img = to_img.reshape(args.input_shape[0], args.input_shape[1])
                    to_img = detect_boundary(to_img)

                    to_img = np.pad(to_img, ((1, 1), (1, 1)), 'constant', constant_values=100)
                    m, n = to_img.shape
                    for j in range(1, m - 1):
                        for jj in range(1, n - 1):
                            cluster = np.zeros((4, args.n_components)) + 1000

                            if to_img[j, jj] == 1:
                                if to_img[j - 1, jj] == 0:
                                    cluster[0] = np.mean(
                                        Embedding_data[np.where(im_target == im_target.reshape(args.input_shape[0], args.input_shape[1])[j - 1 - 1, jj - 1])[0]],
                                        axis=0)

                                if to_img[j, jj + 1] == 0:
                                    cluster[1] = np.mean(
                                        Embedding_data[np.where(im_target == im_target.reshape(args.input_shape[0], args.input_shape[1])[j - 1, jj + 1 - 1])[0]],
                                        axis=0)
                                if to_img[j + 1, jj] == 0:
                                    cluster[2] = np.mean(
                                        Embedding_data[np.where(im_target == im_target.reshape(args.input_shape[0], args.input_shape[1])[j - 1 + 1, jj - 1])[0]],
                                        axis=0)
                                if to_img[j, jj - 1] == 0:
                                    cluster[3] = np.mean(
                                        Embedding_data[np.where(im_target == im_target.reshape(args.input_shape[0], args.input_shape[1])[j - 1, jj - 1 - 1])[0]],
                                        axis=0)

                                u = np.sum((cluster - Embedding_data.reshape(args.input_shape[0], args.input_shape[1], args.n_components)[j - 1, jj - 1]) ** 2, axis=1)
                                u = np.argmin(u)

                                if u == 0:
                                    new_im_target.reshape(args.input_shape[0], args.input_shape[1])[j - 1, jj - 1] = im_target.reshape(args.input_shape[0], args.input_shape[1])[
                                        j - 1 - 1, jj - 1]
                                if u == 1:
                                    new_im_target.reshape(args.input_shape[0], args.input_shape[1])[j - 1, jj - 1] = im_target.reshape(args.input_shape[0], args.input_shape[1])[
                                        j - 1, jj - 1 + 1]
                                if u == 2:
                                    new_im_target.reshape(args.input_shape[0], args.input_shape[1])[j - 1, jj - 1] = im_target.reshape(args.input_shape[0], args.input_shape[1])[
                                        j - 1 + 1, jj - 1]
                                if u == 3:
                                    new_im_target.reshape(args.input_shape[0], args.input_shape[1])[j - 1, jj - 1] = im_target.reshape(args.input_shape[0], args.input_shape[1])[
                                        j - 1, jj - 1 - 1]
                    uu += 1
    return new_im_target

def label2RGB(im_target,args):

    np.random.seed(3)
    label_colours = np.random.randint(255, size=(200, 3))
    label_colours[84] = [255, 255, 187]
    label_colours[25] = [7, 162, 177]
    label_colours[74] = [58, 198, 199]
    im_target_rgb = np.array([label_colours[c % args.nChannel] for c in im_target])
    m, n, _ = args.input_shape
    im_target_rgb = im_target_rgb.reshape(m, n, 3).astype(np.uint8)
    return im_target_rgb
