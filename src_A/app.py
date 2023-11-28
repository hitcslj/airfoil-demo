'''
Airfoil Visualization

Development Refresh Environment: 
python setup.py install && python ./src/app.py
'''
import os
import argparse
from pathlib import Path
from visTorch import visboard

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import re
import numpy as np

'''
Load airfoil dataset
return: list of airfoil surfaces
'''
def load_surfaces(datfile_path):
    surfaces = []
    for datfile in Path(datfile_path).glob('*.dat'):
        # print(f"{p.name}:\n")
        current_surface = []

        with open(datfile) as f:
            lines = [l.strip() for l in f.readlines()]

        for line in lines:
            try:
                xy = [float(str(i)) for i in re.split("\s", line) if len(str(i)) > 0]
            except Exception as e:
                print("Problem decoding line \"%s\"" % line)
                raise e
            if len(xy) > 1:
                current_surface.append([float(xy[0]), float(xy[1])])
        
        surfaces.append(current_surface)

    print ("Got %s surfaces" % len(surfaces))

    # verify all surfaces are of the same length
    len_verify = []
    for surface in surfaces:
        len_verify.append(len(surface))
    if len(set(len_verify)) > 1:
        raise Exception("Surfaces are not of the same length: %s" % set(len_verify))
    else:
        print ("All surfaces are of the same length (%s)" % len_verify[0])

    return surfaces


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Airfoil Visualization')
    parser.add_argument('-v','--visualize', action='store_true', default=True,
                        help='Visualize / Interact with the latent Space (default: %(default)s)')
    parser.add_argument('-c','--n_component', type=int, default=6,
                        help='num of principal components (default: %(default)s)')
    parser.add_argument('-o','--host', default='127.0.0.1',
                        help='IP address for hosting the visualization app (default: %(default)s)')
    parser.add_argument('-p','--port', default='8099',
                        help='hosting port (default: %(default)s)')
    parser.add_argument('-d','--data', default='picked_uiuc',
                        help='name of dataset (default: %(default)s)')
    
    args = parser.parse_args()

    # create relative paths
    dataset_dir = os.path.join(os.getcwd(), 'data/airfoil', args.data)
    
    # dataset
    print('====Loading dataset...====')
    data = load_surfaces(dataset_dir) 
    data = np.array(data)
    print('====Dataset loaded.====')

    pca_model = PCA(n_components=args.n_component, svd_solver='full')
    
    # Preprocess data
    print('====Preprocessing data...====')
    data_2d = np.array([surf.flatten() for surf in data])
    scaler = StandardScaler()
    data_2d_scaled = scaler.fit_transform(data_2d)
    pca_model.fit(data_2d_scaled)
    # data_pca = pca_model.transform(data_2d_scaled)
    print('====Preprocessing Done...====')


    if args.visualize:
        # initialize visualization app
        vis_board = visboard()
        vis_board.add_pca(pca_model, 
                          data_2d_scaled, 
                          latent_options={'n': pca_model.n_components_, 'min': -30, 'max': 30, 'step': 0.01},
                          pre_process=scaler)
        vis_board.run_server(args.host, args.port)
        server = vis_board.app.server
