from config import get_arguments
import numpy as np
from Module import Dimension_Reduction, Feature_Clustering

parser = get_arguments()

parser.add_argument('--input_file',help='input file name', required = True)
parser.add_argument('--input_shape',help='input file shape',type = int, nargs = '+', required = True)
parser.add_argument('--DR_mode',help='Dimension reduction method', default = 'umap')
parser.add_argument('--n_components',help='Reduced dimension', type = int, default = 3)
parser.add_argument('--use_scribble',help='use scribbles', metavar='1 or 0', default=0, type=int)
parser.add_argument('--input_scribble', help = 'input scribble')
parser.add_argument('--output_file', help='output file name', required = True)


if __name__ == '__main__':

    args = parser.parse_args()

    input = np.loadtxt(args.input_file)

    Embedding_data = Dimension_Reduction(input, args)

    OAs = np.loadtxt('Parameter.csv',delimiter=',')

    for para in OAs:

        args.stepsize_sim, args.stepsize_con,args.stepsize_ent = para
        im_target = Feature_Clustering(Embedding_data, args)
        np.savetxt(args.output_file + '%.1f_%.1f_%.1f.txt'%(args.stepsize_sim,args.stepsize_con,args.stepsize_ent), im_target)