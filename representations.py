# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 09:02:12 2019

@author James Cass, Alex Brown, Imogen Taylor, Weicheng Zhu, Cheuk Ho Chan

This module contains functions for exploring three representations of the
protein data

1. 'C': only the centroids of each module
2. 'E': the endpoints of each module
3. 'CE': the centroids and endpoints of each module

It uses data from the file representationdata.txt, containing all centroids and
endpoints of the modules in the data.

(To see how the file representationdata.txt was generated see *.py)

Parameters for each distribution (lengths, angles and dihedrals) are then
calculated. Use get_parameters(rp,sim,rep) to obtain a vector of parameteres
for a particular protein simulation in one of the representations. Parameters are
referred to either by the position in this vector, of in string form e.g.
'Length 1' or 'Angle 3' or 'Dihedral 7'.
"""

from protein import get_CA_coords, get_centroids_from_modules, get_endpoints_from_modules
from protein import get_centroids_and_endpoints_from_modules, joint_locations, protein_string

import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.linalg import norm

from Bio.PDB.vectors import calc_angle, calc_dihedral, Vector
from alex import ellipse_cone, protein_synthesis


# load the matrix of coordinates of all endpoints and centroids
coords = np.loadtxt('representationdata.txt')

# numbers of lengths, angles and dihedrals in one joint in each representation

joint_param_split = {'C' : [2,1,0],     # 2 lengths, 1 angle, no dihedrals
                     'E' : [3,2,1],     # 3 lengths, 2 angles, 1 dihedral
                     'CE': [6,5,4]}     # 6 lengths, 5 angles, 4 dihedrals

lengths_per_module = {'C' : 1,
                      'E' : 1,
                      'CE': 2}

def joint_parameter_locations(joint_pos, rep='CE', protein_length=5):
    """
    eg.
    x = get_parameters(1, 1, 'CE')
    joint_parameter_locations(0) ->
    [0,1,2,3,4,5,10,11,12,13,14,19,20,21,22]
    are the positions in the vector x of the parameters for the first joint
    in a protein of 5 modules using the 'CE' representation
    """
    l,a,d = joint_param_split[rep]
    lpm = lengths_per_module[rep]

    # calculate total number of length
    total_lengths = l + lpm * (protein_length - 3)

    # add the length positions to the array
    joint_locs = [i + lpm*joint_pos for i in range(l)]

    # append the angle positions
    joint_locs.extend(
            [i + lpm*joint_pos for i in range(total_lengths, total_lengths+a)])

    # append the dihedral positions
    d_start = 2*total_lengths-1
    joint_locs.extend(
            [i + lpm*joint_pos for i in range(d_start,d_start+d)])

    return joint_locs


def param_split(rep='CE', protein_length=5):
    """
    The number of lengths, angles and dihedrals for a given protein
    length and representation.
    e.g.
    param_split('C',6) -> [5,4,3]
    i.e. in a protein of length 6, using the 'C' representation there
    are 5 lengths, 4 angles and 3 dihedrals
    """

    # number of parameters for one joint
    l,a,d = joint_param_split[rep]
    lpm = lengths_per_module[rep]

    # extend to number for whole protein
    extra = lengths_per_module[rep] * (protein_length - 3)

    l += extra; a += extra; d += extra
    return [l,a,d]



## Functions for interchanging between parameter positions and names


### e.g.
### parameter_name(0)
### 'Length 1'
### parameter_name(3, rep='C')
### 'Length 4'

def parameter_name(position, rep='CE', protein_length=5):
    """
    The string form of a parameter name given a position in a vector (e.g. as
    returned from get_parameters) for a given representation and protein
    length.

    e.g.
    parameter_name(0,'C') -> 'Length 1'
    parameter_name(10,'CE') -> 'Angle 1'
    """
    l,a,d = param_split(rep, protein_length)
    if position < l:
        return 'Length {}'.format(position + 1)
    elif position < (l+a):
        return 'Angle {}'.format(position - l + 1)
    elif position < (l+a+d):
        return 'Dihedral {}'.format(position - l - a + 1)
    else:
        raise Exception('No parameter at that position')

def parameter_position(name, rep='CE', protein_length=5):
    """
    The position in the parameters vector (e.g. as returned from get_parameters)
    of a parameter given by the string form of it's name, for a given
    representation and protein length.

    e.g.
    parameter_position('Dihedral 1', 'E', 7) -> 13
    should be understood as, the first dihedral angle is located at index 13 of
    the parameters vector in a protein of length 5 using the 'E' representation.
    """
    ptype, num = name.split(' ')
    num = int(num)
    l,a,d = param_split(rep, protein_length)
    if ptype == 'Length' and num <= l:
        return num - 1
    elif ptype == 'Angle' and num <= a:
        return num + l - 1
    elif ptype == 'Dihedral' and num <= d:
        return num + l + a - 1
    else:
        raise Exception('No parameter called {}'.format(name))

def joint_parameter_names(j, rep='CE', protein_length=5):
    """
    The names of all parameters specific to a joint (j=0,1,2) for a given
    representation and protein length.

    e.g.
    joint_parameter_names(2,'C') -> ['Length 3', 'Length 4', 'Angle 3']
    """
    names = []
    for i in joint_parameter_locations(j, rep, protein_length):
        names.append(parameter_name(i, rep, protein_length))
    return names


## Extract data from the matrix

def get_row(rp, sim):
    """
    The row index of the matrix for a particular simulation.
    """
    return (rp-1)*100 + (sim-1)

def make_vectors(row, points):
    """
    Turn 3D point information into BioPython Vector objects
    """
    vectors = []
    for i in [3*j for j in points]:
        vectors.append(Vector(*coords[row,i:i+3]))
    return vectors

def get_centroids(rp, sim):
    return make_vectors(get_row(rp, sim), [1,3,5,7,9])

def get_endpoints(rp, sim):
    return make_vectors(get_row(rp, sim), [0,2,4,6,8,10])

def get_centroids_and_endpoints(rp, sim):
    return make_vectors(get_row(rp, sim), range(11))


### Functions for calculating parameters of a representation ########

def get_lengths(vectors):
    """ Get the list of lengths of the vectors between points.
    vectors is a list of position vectors (from the origin). """
    vs = [vectors[i+1] - vectors[i] for i in range(len(vectors)-1)]  # get vectors between points from position vectors
    return [v.norm() for v in vs]  # return lengths of the vectors

def get_angles(vectors):
    """ Get the angles between each pair in the list of vectors """
    angles = []
    for i in range(len(vectors)-2):
        angles.append(calc_angle(vectors[i],
                                 vectors[i+1],
                                 vectors[i+2]))
    return angles

def get_dihedrals(vectors):
    """ Get the dihedral angle corresponding to the angle between
    the planes defined by each pair of vectors in sequence """
    dihedrals = []
    for i in range(len(vectors)-3):
        dihedrals.append(calc_dihedral(vectors[i],
                                       vectors[i+1],
                                       vectors[i+2],
                                       vectors[i+3]))
    return dihedrals


## Functions used to extract the data from the full set. Not needed after extraction
## since it is faster to use the data from representationdata.txt

def get_parameters_from_file(rp, sim, rep='CE'):
    """
    Get the parameters of a single protein simulation. The parameters for the
    centroid representation are

    L1, L2, L3, L4:  the lengths of the vectors between the centroids
    theta1, theta2, theta 3: the angles between pairs of vectors
    d1, d2:  the dihedral angles between the planes defined by pairs of vectors

    Other representations contain more parameters.
    """

    modules = get_CA_coords(rp, sim)
    if rep == 'C':
        vectors = get_centroids_from_modules(modules)
    elif rep == 'E':
        vectors = get_endpoints_from_modules(modules)
    elif rep == 'CE':
        _,_,_,vectors = get_centroids_and_endpoints_from_modules(modules)
    else:
        raise Exception('No representation called {}'.format(rep))

    lengths = get_lengths(vectors)
    angles = get_angles(vectors)
    dihedrals = get_dihedrals(vectors)

    return [*lengths, *angles, *dihedrals]

def get_mean_parameters_from_file(rp, rep='CE'):
    """
    Return the mean parameters over the 100 simulations
    """
    params = []
    for sim in range(1,101):
        params.append(get_parameters_from_file(rp, sim, rep))

    params = np.array(params)
    return np.mean(params, axis=0)

## functions used to get the vectors for each representation ##

vector_fns = {'C'  : get_centroids,
              'E'  : get_endpoints,
              'CE' : get_centroids_and_endpoints}

## Functions for getting parameters and distributions for each representation
## using the matrix in representationdata.txt (faster than above functions)

def get_parameters(rp, sim, rep='CE'):
    """
    Get the parameters of a single protein simulation. The parameters for the
    centroid representation are

    L1, L2, L3, L4:  the lengths of the vectors between the centroids
    theta1, theta2, theta 3: the angles between pairs of vectors
    d1, d2:  the dihedral angles between the planes defined by pairs of vectors

    Other representations contain more parameters.
    """
    try:
        vectors = vector_fns[rep](rp,sim)
    except:
        raise Exception('No representation called {}'.format(rep))

    lengths = get_lengths(vectors)
    angles = get_angles(vectors)
    dihedrals = get_dihedrals(vectors)

    return np.array([*lengths, *angles, *dihedrals])


def distribution(rp, param, rep='CE'):
    """
    Return all data points for a single parameter in a protein, for a given
    representation.

    e.g. try
    x = distribution(53, 'Length 1', 'C')
    """
    if type(param) == str:
        i = parameter_position(param, rep)
    else:
        i = param
    xs = []
    for sim in range(1, 100):
        p = get_parameters(rp, sim, rep)
        xs.append(p[i])
    return xs


def plotdistribution(rp, param, rep='CE', bins=15):
    """
    Get the distribution of a single parameter in a protein, and plot it
    in a histogram

    e.g. try
    x = distribution(53, 'Length 1', 'C')
    """
    xs = distribution(rp, param, rep)
    plt.hist(xs, bins)
    plt.title('rp{}'.format(rp))
    plt.ylabel('frequency')
    plt.xlabel(param)
    plt.show()


## Functions for analysing joints

def anglemean(thetas):
    """
    Calculate the mean of a list of angles. Think of the angles as corresponding
    to points on the unit circle, and take the mean of the x and y values located
    at the sines and cosines of the angles. Then take the inverse tangent of the
    ratio to find the mean angles.
    """
    x_mean = np.mean(np.cos(thetas))
    y_mean = np.mean(np.sin(thetas))
    return np.arctan2(y_mean, x_mean)


def anglepercentiles(thetas):
    """
    Calculate the 5th and 9th percentiles of a list of angles between -pi and pi.
    """
    # regular calculation of mean
    mean = np.mean(thetas)
    # angular calculation of mean
    amean = anglemean(thetas)
    # if the data range goes over the crossover (-pi,pi)
    if abs(mean - amean) > 0.1:
        # shift positives to negatives
        negatives = thetas[thetas > 0] - np.pi
        # and negatives to positives
        positives = thetas[thetas < 0] + np.pi
        # put data together and calculate the percentiles
        data = np.concatenate((negatives, positives))
        fifth = np.percentile(data, 5)
        ninetyfifth = np.percentile(data, 95)
        # shift back
        fifth -= np.sign(fifth)*np.pi
        ninetyfifth -= np.sign(ninetyfifth)*np.pi
    else:
        fifth = np.percentile(thetas, 5)
        ninetyfifth = np.percentile(thetas, 95)

    return fifth, ninetyfifth


def jointdistribution(joint, parameter, rep='CE'):
    """
    The distribution of a parameter that appears in more than one joint in the
    data.

    e.g.
    jointdistribution('D18 D18_j1_D14 CcapD14', 'Length 1') ->
    a list of 200 data points, corresponding to the two occurences of this
    joint in the data
    """
    pos = parameter_position(parameter, rep)
    params = [];
    for rp, j in joint_locations(joint):
        for sim in range(1, 101):
            # for each simulation, get the parameters for the representation
            p = get_parameters(rp, sim, rep)
            # only keep the the ones relevant to this joint
            params.append(p[pos])
    return np.array(params)



def jointinfo(joint, rep='CE'):
    """
    Print info relating to all occurences of this joint in the data, in the
    given representation. The ranges given are the 5th and 95th percentiles
    of the combined data set.

    e.g. try
    jointinfo(D18 D18_j1_D14 CcapD14')
    """
    print('\nJoint: {}'.format(joint))

    num_params = param_split(rep)

    for rp, j in joint_locations(joint):
        print('\nrp{}\n'.format(rp))
        print('{:12} {:10} {:10}'.format(
            'Parameter', 'Mean', 'Range'))
        params = [];
        names = joint_parameter_names(j, rep)
        for sim in range(1, 101):
            # for each simulation, get the parameters for the representation
            p = get_parameters(rp, sim, rep)
            # only keep the the ones relevant to this joint
            params.append(p[np.array(joint_parameter_locations(j, rep))])
        params = np.stack(params)

        for i in range(np.shape(params)[1]):
            if i < num_params[0]:
                mean = np.mean(params[:,i])
                fifth = np.percentile(params[:,i], 5)
                ninetyfifth = np.percentile(params[:,i], 95)
            else:
                mean = anglemean(params[:,i])
                fifth, ninetyfifth = anglepercentiles(params[:,i])
            print('{:12} {:<10.3f} ({:.3f},{:.3f})'.format(
                        names[i], mean, fifth, ninetyfifth))


## Functions for plotting simulations ##

def plotC(ax, centroids, modules):
    """
    Plot a protein by vectors in the 'C' representation with legend given by
    names in modules
    """
    xs = centroids[:,0]
    ys = centroids[:,1]
    zs = centroids[:,2]

    for x,y,z,mod in zip(xs,ys,zs, modules):
        ax.plot([x],[y],[z],marker='o', markersize=10, label=mod)

    for i in range(1, np.shape(centroids)[0]):
    # plot the lines between the points
        ax.plot3D([xs[i-1], xs[i]],
                  [ys[i-1],ys[i]],
                  [zs[i-1], zs[i]], 'k', alpha=0.5)

def plotE(ax, endpoints, modules):
    """
    Plot a protein by vectors in the 'E' representation with legend given by
    names in modules
    """
    xs = endpoints[:,0]
    ys = endpoints[:,1]
    zs = endpoints[:,2]

    for i, mod in zip(range(1, len(modules)+1), modules):
    # plot the lines between the points
        ax.plot3D([xs[i-1], xs[i]],
                  [ys[i-1],ys[i]],
                  [zs[i-1], zs[i]], linewidth=5, label=mod)

def plotCE(ax, centroids, c_and_e, modules):
    """
    Plot a protein by vectors in the 'CE' representation with legend given by
    names in modules
    """
    cxs = centroids[:,0]
    cys = centroids[:,1]
    czs = centroids[:,2]
    xs = c_and_e[:,0]
    ys = c_and_e[:,1]
    zs = c_and_e[:,2]

    ax.scatter(cxs, cys, czs, linewidths=5, c='k', alpha=1)

    for j, mod in enumerate(modules):
        i = 2*j
        ax.plot3D([xs[i], xs[i+1], xs[i+2]],
                  [ys[i], ys[i+1], ys[i+2]],
                  [zs[i], zs[i+1], zs[i+2]], linewidth=3, label=mod)


def plotsim(rp, sim, rep='CE', title='', xlabel='', ylabel='', save=False):
    """
    Plot a simulation given by rp and sim numbers from the data, in the given
    representation.
    """
    centroids = np.array([v.get_array() for v in get_centroids(rp, sim)])
    endpoints = np.array([v.get_array() for v in get_endpoints(rp, sim)])
    c_and_e = np.array([v.get_array() for v in get_centroids_and_endpoints(rp,sim)])
    modules = protein_string(rp).split()

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    if rep == 'C':
        plotC(ax, centroids, modules)
    elif rep == 'E':
        plotE(ax, endpoints, modules)
    elif rep == 'CE':
        plotCE(ax, centroids, c_and_e, modules)

    ax.legend()
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if save:
        plt.savefig('figrp{}.eps'.format(rp))
    else:
        plt.show()

# Functions for creating new proteins

def newproteindistributions(protein):
    """
    Get the distributions of all parameters for a new protein sequence,
    using the 'CE' representation.

    e.g.
    newproteindistribution('NcapD14 D14_j4_D79 D79_j2_D14 D14_j2_D79 CcapD79') ->
    a list of lists, each containing all the data points relevant to a single
    parameter (first is 'Length 1' etc.).
    """
    modules = protein.split()

    # construct a list of the joint names
    joints = [' '.join(modules[i:i+3]) for i in range(len(modules)-2)]

    # numbers of lengths, angles and dihedrals for new protein
    num_params = param_split('CE', len(modules))

    # initialise an empty array for the data for each parameter
    param_dists = [[] for i in range(sum(num_params))]

    for joint_num, joint_name in enumerate(joints):
        params = [];
        for rp, j in joint_locations(joint_name):
            for sim in range(1, 101):
                # for each simulation, get the parameters for the representation
                p = get_parameters(rp, sim, 'CE')
                # only keep the the ones relevant to this joint
                params.append(p[np.array(joint_parameter_locations(j, 'CE'))])

        # turn params into a matrix with columns containing the distribution
        # of each parameter
        params= np.stack(params)

        # append the parameters in the correct place in the new protein
        param_locs = joint_parameter_locations(joint_num,
                                               'CE',
                                               protein_length=len(modules))
        for i in range(len(param_locs)):
            param_dists[param_locs[i]].extend(params[:,i])

    return param_dists

def newproteinranges(protein, display=False):
    """
    Get the means, 5th and 95th percentiles for all parameters in a new protein
    sequence, using the 'CE' representation
    """
    param_dists = newproteindistributions(protein)
    modules = protein.split()
    num_params = param_split('CE', len(modules))

    if display:
        print('{:12} {:10} {:17} {:10}'.format(
                'Parameter','Mean','Range','Data Points'))

    means = []; fifths = []; ninetyfifths = [];

    for i, param in enumerate(param_dists):
        name = parameter_name(i, 'CE', len(modules))
        param = np.array(param)
        mean = np.mean(param)
        # calculate means and percentiles differently for lengths..
        if i < num_params[0]:
            fifth = np.percentile(param, 5)
            ninetyfifth = np.percentile(param, 95)
        # or angles
        else:
            mean = anglemean(param)
            fifth, ninetyfifth = anglepercentiles(param)

        means.append(mean)
        fifths.append(fifth)
        ninetyfifths.append(ninetyfifth)

        if display:
            print('{:12} {:<10.3f} ({:6.3f},{:6.3f})   {:<10d}'
                  .format(name, mean, fifth, ninetyfifth, len(param)))

    return means, fifths, ninetyfifths

def newprotein(protein):
    """
    Predict and display information about a new protein structure, using the 'CE'
    representation.
    """
    modules = protein.split()
    print('Protein:\n{}\n'.format(protein))

    # get the ranges of the new protein
    means, fifths, ninetyfifths = newproteinranges(protein,True)

    # some sorting required before reconstruction in space
    l,a,d,a5,a95,d5,d95 = sortparameters(means,
                                         fifths,
                                         ninetyfifths,
                                         len(modules))
    # construct the new protein in 3D space
    v,va5,va95,vd5,vd95 = protein_synthesis(l,a,d,a5,a95,d5,d95)

    # plot the new protein
    plotnewprotein(v,va5,va95,vd5,vd95, modules)


def sortparameters(m, f, n, protein_length=5):
    """
    This sorting is required of the means, 5th and 95th percentiles before
    passing to the function protein_synthesis
    """
    lnum, anum, dnum = param_split(protein_length=protein_length)
    l = m[0:lnum]
    a = m[lnum:lnum+anum]
    d = m[lnum+anum:]
    a5 = f[lnum:lnum+anum]
    a95 = n[lnum:lnum+anum]
    d5 = f[lnum+anum:]
    d95 = n[lnum+anum:]
    return l, a, d, a5, a95, d5, d95


def plotnewprotein(V,VA5,VA95,VD5,VD95,modules,title='',xlabel='',ylabel=''):
    """
    Plot a new protein structure given position vectors of points, and position
    vectors of the 5th and 95th percentiles for plotting a cone of movement
    possibilities. Plot a legend with the names in modules.
    """
    centroids = np.array([V[i].get_array() for i in range(1,len(V),2)])
    c_and_e = np.array([V[i].get_array() for i in range(len(V))])
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    plotCE(ax, centroids, c_and_e, modules)
    ax.legend()
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    for i in range(len(V)-1):
        ellipse_cone(ax,
                     V[i].get_array(),
                     V[i+1].get_array(),
                     (VA95[i+1]-VA5[i+1]).norm() / 2,
                     (VD95[i+1]-VD5[i+1]).norm() / 2,
                     'b')
    plt.show()

def protein_synthesis(l,t,d,t5,t95,d5,d95):
    """
    protein_synthesis takes all the lengths, angles and dihedrals and returns points in 3-D space using rotation matrices, as well as returning the points
    in space of the 5th and 95th percentiles of the angles and dihedrals
    """
    V=[]
    VA5=[]
    VA95=[]
    VD5=[]
    VD95=[]


    #rotation matrices for the second vector, rotating about a vector perpendicular to the first vector
    if (t[0] >= 0):
        RT = rotaxis(math.pi - t[0], Vector(0, 1, 0))
    elif (t[0] < 0):
        RT = rotaxis(-(math.pi - t[0]), Vector(0, 1, 0))
    if (t5[0] >= 0):
        RT5 = rotaxis(math.pi - t5[0], Vector(0, 1, 0))
    elif (t[0] < 0):
        RT5 = rotaxis(-(math.pi - t5[0]), Vector(0, 1, 0))
    if (t5[0] >= 0):
        RT95 = rotaxis(math.pi - t95[0], Vector(0, 1, 0))
    elif (t[0] < 0):
        RT95 = rotaxis(-(math.pi - t95[0]), Vector(0, 1, 0))

    #First 3 points for the representation
    #point 0
    V.append(Vector(0,0,0))
    #point 1
    V.append(Vector(l[0],0,0))
    #point 2, found by rotating the original vector and changing the length
    V.append(V[1]+(V[1].left_multiply(RT)).normalized()**l[1])


    #Points for the 5th and 95th percentiles

    # point 0
    VA5.append(Vector(0, 0, 0))
    # point 1
    VA5.append(Vector(l[0], 0, 0))
    # point 2
    VA5.append(V[1] + (V[1].left_multiply(RT5)).normalized() ** l[1])

    # point 0
    VA95.append(Vector(0, 0, 0))
    # point 1
    VA95.append(Vector(l[0], 0, 0))
    # point 2
    VA95.append(V[1] + (V[1].left_multiply(RT95)).normalized() ** l[1])

    # point 0
    VD5.append(Vector(0, 0, 0))
    # point 1
    VD5.append(Vector(l[0], 0, 0))
    # point 2
    VD5.append(V[1] + (V[1].left_multiply(RT)).normalized() ** l[1])

    # point 0
    VD95.append(Vector(0, 0, 0))
    # point 1
    VD95.append(Vector(l[0], 0, 0))
    # point 2
    VD95.append(V[1] + (V[1].left_multiply(RT)).normalized() ** l[1])

    #loops through all the lengths, angles and dihedrals, creating new vectors and points in 3-D space
    i=2
    while (i<=len(l)-1):

        #New angle rotation matrices
        if (t[i-1] >= 0):
            RT = rotaxis(math.pi - t[i-1], (V[i] - V[i-1]) ** V[i-1])
        elif (t[i-1] < 0):
            RT = rotaxis(-(math.pi - t[i-1]), (V[i] - V[i-1]) ** V[i-1])

        if (t5[i-1] >= 0):
            RT5 = rotaxis(math.pi - t5[i-1], (V[i] - V[i-1]) ** V[i-1])
        elif (t5[i-1] < 0):
            RT5 = rotaxis(-(math.pi - t5[i-1]), (V[i] - V[i-1]) ** V[i-1])

        if (t95[i-1] >= 0):
            RT95 = rotaxis(math.pi - t95[i-1], (V[i] - V[i-1]) ** V[i-1])
        elif (t95[i-1] < 0):
            RT95 = rotaxis(-(math.pi - t95[i-1]), (V[i] - V[i-1]) ** V[i-1])

        #New dihedral rotation matrices
        if (d[i-2] >= 0):
            RD = rotaxis((math.pi - d[i-2]), V[i] - V[i-1])
        elif (d[i-2] < 0):
            RD = rotaxis(-(math.pi - d[i-2]), V[i] - V[i-1])

        if (d5[i-2] >= 0):
            RD5 = rotaxis((math.pi - d5[i-2]), V[i] - V[i-1])
        elif (d5[i-2] < 0):
            RD5 = rotaxis(-(math.pi - d5[i-2]), V[i] - V[i-1])

        if (d95[i-2] >= 0):
            RD95 = rotaxis((math.pi - d95[i-2]), V[i] - V[i-1])
        elif (d95[i-2] < 0):
            RD95 = rotaxis(-(math.pi - d95[i-2]), V[i] - V[i-1])
        #point i+1
        #rotates by angles and by dihedral
        V.append(V[i]+(((V[i]-V[i-1]).left_multiply(RT)).left_multiply(RD)).normalized()**l[i])
        VA5.append(V[i] + (((V[i] - V[i - 1]).left_multiply(RT5)).left_multiply(RD)).normalized() ** l[i])
        VA95.append(V[i] + (((V[i] - V[i - 1]).left_multiply(RT95)).left_multiply(RD)).normalized() ** l[i])
        VD5.append(V[i] + (((V[i] - V[i - 1]).left_multiply(RT)).left_multiply(RD5)).normalized() ** l[i])
        VD95.append(V[i] + (((V[i] - V[i - 1]).left_multiply(RT)).left_multiply(RD95)).normalized() ** l[i])
        i=i+1
    return V,VA5,VA95,VD5,VD95



def ellipse_cone(ax, p0, p1, R0, R1, color):
    """
    Takes two radii and two points and plots an elipse cone
    """

    # vector in direction of axis
    v = p1 - p0
    # find magnitude of vector
    mag = norm(v)
    # unit vector in direction of axis
    v = v / mag
    # make some vector not in the same direction as v
    not_v = np.array([1, 1, 0])
    if (v == not_v).all():
        not_v = np.array([0, 1, 0])
    # make vector perpendicular to v
    n1 = np.cross(v, not_v)
    # print n1,'\t',norm(n1)
    # normalize n1
    n1 /= norm(n1)
    # make unit vector perpendicular to v and n1
    n2 = np.cross(v, n1)
    # surface ranges over t from 0 to length of axis and 0 to 2*pi
    n = 80
    t = np.linspace(0, mag, n)
    theta = np.linspace(0, 2 * np.pi, n)
    # use meshgrid to make 2d arrays
    t, theta = np.meshgrid(t, theta)
    RA = np.linspace(0, R0, n)
    RD = np.linspace(0, R1, n)
    # generate coordinates for surface
    X, Y, Z = [p0[i] + v[i] * t + RA *
               np.sin(theta) * n1[i] + RD * np.cos(theta) * n2[i] for i in [0, 1, 2]]
    ax.plot_surface(X, Y, Z, color=color,alpha=0.05, linewidth=0, antialiased=False)
