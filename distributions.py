import numpy as np
from scipy import stats
import matplotlib.pyplot as plt


def jsd(p, q, base=np.e):
    '''
        Implementation of pairwise `jsd` based on
        https://en.wikipedia.org/wiki/Jensen%E2%80%93Shannon_divergence
    '''
    ## convert to np.array
    p, q = np.asarray(p), np.asarray(q)
    ## normalize p, q to probabilities
    p, q = p/p.sum(), q/q.sum()
    m = 1./2*(p + q)
    return stats.entropy(p,m, base=base)/2. +  stats.entropy(q, m, base=base)/2.


def plot_heatmap(divergence_matrix):
    """
    adapted from https://matplotlib.org/gallery/images_contours_and_fields/image_annotated_heatmap.html
    """
    fig, ax = plt.subplots()
    im = ax.imshow(divergence_matrix)
    num_dists = np.shape(divergence_matrix)[0]
    # We want to show all ticks...
    ax.set_xticks(np.arange(num_dists))
    ax.set_yticks(np.arange(num_dists))
    # ... and label them with the respective list entries
    labels = [str(i) for i in range(num_dists-1)] + ['total']
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    for i in range(num_dists):
        for j in range(num_dists):
            text = ax.text(j, i, '{:.2f}'.format(divergence_matrix[i, j]),
                           ha="center", va="center", color="w")

    ax.set_title("Divergence Matrix")
    fig.tight_layout()
    plt.show()

def divergence_matrix(vector_of_data, method='KL'):
    split_data = [vector_of_data[i:i + 100] for i in range(0, len(vector_of_data), 100)]

    KDES = []
    #estimator = FFTKDE(kernel='gaussian', bw= 'silverman')
    xmin = np.min(vector_of_data)
    xmax = np.max(vector_of_data)
    plt.figure()
    for i in range(0,len(split_data)):
        x_grid = np.linspace(xmin,xmax,1000)
        kernel = stats.gaussian_kde(split_data[i], bw_method = 'silverman' )
        pdf = kernel.evaluate(x_grid)
        plt.plot(x_grid,pdf,'-',label=i)
        KDES.append(pdf)

    kernel = stats.gaussian_kde(vector_of_data, bw_method = 'silverman' )
    pdf = kernel.evaluate(x_grid)
    plt.plot(x_grid,pdf,'-', label='total')
    KDES.append(pdf)
    plt.legend()
    KDES = np.vstack(KDES)
    # avoid divide by zero
    KDES[KDES < 1e-16 ] = 1e-15

    all_kl = []
    for i in range(0,len(KDES)):
        kls_ref = []
        for j in range(0,len(KDES)):
            if method == 'KL':
                kl = stats.entropy(KDES[j], qk=KDES[i])
            elif method == 'JS':
                kl = jsd(KDES[j], KDES[i], base=np.e)
            else:
                print('Method {} not implemented'.format(method))
                return
            kls_ref.append(kl)
        all_kl.append(kls_ref)

    divergence_matrix = np.array(all_kl)
    plot_heatmap(divergence_matrix)
    return divergence_matrix
