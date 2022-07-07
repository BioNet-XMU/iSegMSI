import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from loss import MaxEntropy
from model import Net,remove_artifacts,label2RGB
import torch.optim as optim
from torch.autograd import Variable
import cv2
import torch.nn.init
from sklearn.metrics import silhouette_score

def Dimension_Reduction(data, args):

    index_tissue = np.where(np.sum(data,axis=1) != 0)[0]
    m, n, _ = args.input_shape
    return_data = np.zeros((m*n,args.n_components))

    if args.DR_mode == 'umap':
        Embedding_data = umap.UMAP(n_components=args.n_components,metric='cosine',random_state=0).fit_transform(data[index_tissue])

    if args.DR_mode == 'pca':
        Embedding_data = PCA(n_components=args.n_components).fit_transform(data[index_tissue])

    if args.DR_mode == 'tsne':
        Embedding_data = TSNE(n_components=args.n_components,random_state=0).fit_transform(data[index_tissue])

    return_data[index_tissue] = Embedding_data

    return_data = MinMaxScaler().fit_transform(return_data)
    return_data = return_data.reshape(m, n, args.n_components)

    return return_data

def Feature_Clustering(Embedding_data,args):

    use_cuda = torch.cuda.is_available()
    torch.manual_seed(0)
    data = torch.from_numpy(np.array([Embedding_data.transpose((2, 0, 1)).astype('float32')]))

    if use_cuda:
        data = data.cuda()
    data = Variable(data)
    # load scribble


    if args.use_scribble:

        # scribble = np.loadtxt('data/fetus_mouse_scribble.txt')
        scribble = np.loadtxt(args.input_scribble)
        mask = scribble.reshape(-1)

        mask_inds = np.unique(mask)

        mask_inds = np.delete( mask_inds, np.argwhere(mask_inds==255) )
        inds_sim = torch.from_numpy( np.where( mask == 0 )[ 0 ] )
        inds_scr = torch.from_numpy( np.where( mask != 0 )[ 0 ] )
        target_scr = torch.from_numpy( mask.astype('int64') )

        # SC_score = silhouette_score(Embedding_data.reshape(-1, args.n_components)[inds_scr], mask[inds_scr])
        # print(SC_score)

        if use_cuda:
            inds_sim = inds_sim.cuda()
            inds_scr = inds_scr.cuda()
            target_scr = target_scr.cuda()
        target_scr = Variable( target_scr )
        # set minLabels

        args.minLabels = len(mask_inds)

        if args.n_components == 3:

            plt.subplot(1, 2, 1)
            plt.imshow(Embedding_data)
            plt.title('Embedding data')
            plt.subplot(1, 2, 2)
            plt.imshow(scribble)
            plt.title('Scribbles')
            # plt.show()

    else:

        if args.n_components == 3:
            plt.imshow(Embedding_data)
            plt.title('Embedding data')
            # plt.show()

    model = Net(data.size(1),args)

    if args.n_components == 3:
        model.load_state_dict(torch.load('data/weight.pth'))

    if use_cuda:
        model.cuda()
    model.train()

    # similarity loss definition
    loss_sim = torch.nn.CrossEntropyLoss()

    # scribble loss definition
    loss_scr = torch.nn.CrossEntropyLoss()

    # Entropy loss definition
    loss_ent = MaxEntropy()

    # continuity loss definition
    loss_hpy = torch.nn.L1Loss(size_average=True)
    loss_hpz = torch.nn.L1Loss(size_average=True)

    HPy_target = torch.zeros(args.input_shape[0] - 1, args.input_shape[1], args.nChannel)
    HPz_target = torch.zeros(args.input_shape[0], args.input_shape[1] - 1, args.nChannel)

    if use_cuda:
        HPy_target = HPy_target.cuda()
        HPz_target = HPz_target.cuda()

    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)

    for epoch in range(args.maxIter):
        # forwarding
        optimizer.zero_grad()
        output = model(data)[0]
        output = output.permute(1, 2, 0).contiguous().view(-1, args.nChannel)

        outputHP = output.reshape((args.input_shape[0], args.input_shape[1], args.nChannel))
        HPy = outputHP[1:, :, :] - outputHP[0:-1, :, :]
        HPz = outputHP[:, 1:, :] - outputHP[:, 0:-1, :]
        lhpy = loss_hpy(HPy, HPy_target)
        lhpz = loss_hpz(HPz, HPz_target)

        ignore, target = torch.max(output, 1)
        im_target = target.data.cpu().numpy()
        nLabels = len(np.unique(im_target))

        if args.visualize:
            im_target_rgb = label2RGB(im_target,args)
            cv2.imshow("output", im_target_rgb)
            cv2.waitKey(10)

        # loss
        if args.use_scribble == True:

            loss = args.stepsize_sim * loss_sim(output[inds_sim],target[inds_sim]) + \
                   args.stepsize_scr * loss_scr(output[inds_scr],target_scr[inds_scr]) + \
                   args.stepsize_con * (lhpy + lhpz) + loss_ent(output)

        else:

            loss = args.stepsize_sim * loss_sim(output, target) + \
                   args.stepsize_con * (lhpy + lhpz) + loss_ent(output)

        loss.backward()
        optimizer.step()

        print(epoch, '/', args.maxIter, ' | loss :', loss.item())

        if nLabels <= args.minLabels:
            print("nLabels", nLabels, "reached minLabels", args.minLabels, ".")
            break

    cv2.imwrite(args.output_file + '.png', im_target_rgb)

    if args.remove_artifacts:
        im_target = remove_artifacts(Embedding_data,im_target,args)
        im_target = remove_artifacts(Embedding_data,im_target,args)


    if args.visualize:
        im_target_rgb = label2RGB(im_target, args)
        cv2.imshow("output", im_target_rgb)
        cv2.waitKey(50)
        cv2.imwrite(args.output_file + '.png',im_target_rgb)

    return im_target
