import argparse

def get_arguments():

    parser = argparse.ArgumentParser(description='iSegMSI for MSI interactive segmentation')

    parser.add_argument('--nChannel', metavar='N', default=100, type=int,
                        help='number of channels')
    parser.add_argument('--maxIter', metavar='T', default=1000, type=int,
                        help='number of maximum iterations')
    parser.add_argument('--minLabels', metavar='minL', default=3, type=int,
                        help='minimum number of labels')
    parser.add_argument('--lr', metavar='LR', default=0.005, type=float,
                        help='learning rate')
    parser.add_argument('--nConv', metavar='M', default=2, type=int,
                        help='number of convolutional layers')
    parser.add_argument('--remove_artifacts', metavar='1 or 0', default=1, type=int,
                        help='remove the artifacts')
    parser.add_argument('--visualize', metavar='1 or 0', default=1, type=int,
                        help='visualization flag')
    parser.add_argument('--stepsize_sim', metavar='SIM', default=1, type=float,
                        help='step size for similarity loss', required=False)
    parser.add_argument('--stepsize_con', metavar='CON', default=1, type=float,
                        help='step size for continuity loss')
    parser.add_argument('--stepsize_scr', metavar='SCR', default=1, type=float,
                        help='step size for scribble loss')

    return parser