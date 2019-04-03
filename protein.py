# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 08:55:41 2019

@author: James Cass, Alex Brown, Imogen Taylor, Weicheng Zhu, Cheuk Ho Chan

This module contains functions for interacting with the full data set of
644 proteins. It is assumed that the files are located in the path specified
by PDBPATH, for example with file structure:

PDBPATH/32/rp32_0001.pdb

where all simulations for rp1 live in the folder '32'. It is also assumed that
the files full_rplist.txt and modules-length.txt are in the top directory.
"""

from Bio.PDB import PDBParser
from Bio.PDB.vectors import calc_angle, calc_dihedral, Vector

import numpy as np
import random

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt


## PUT PATH TO PDB FILES HERE ####

PDBPATH = r'F:\Downloads\eng_data'

## Make a dictionary of module start and end points ##

PROTEINS = {} # dictionary of the modules in each protein
LENDICT = {} # dictionary for the length of each module
MODULE_LOCS = {} # start/end points of each module in a protein
NCAP_JOINTS = set() # all the names of ncaps
CCAP_JOINTS = set() # all ccaps
INTERIOR_JOINTS = set() # all interior joints
JOINT_LOCS = {} # all locations of a joint in the data set


# create the length dictionary
with open('modules-length.txt','r') as lenfile:
    LENDICT = {}
    for line in lenfile.readlines():
        module, length = line.split()   # get module name and length
        LENDICT[module] = int(length)   # make a dictionary of lengths


# create the dictionaries of module length and joint locations
# fill sets of ncaps, ccaps and joints

with open('full_rplist.txt','r') as rpfile:
    proteins = []
    for i, line in enumerate(rpfile.readlines()):  # go through each line of full_rplist
        protein = line.split()  # split the line up and put it in a list
        rp, modules = protein[0], protein[1:] # split into protein name and module names
        PROTEINS[rp] = ' '.join(modules)

        # store names of ncaps, joints and ccaps for protein generation
        NCAP_JOINTS.add(' '.join(modules[0:3]))
        INTERIOR_JOINTS.add(' '.join(modules[1:4]))
        CCAP_JOINTS.add(' '.join(modules[2:5]))

        # create the dictionary of module locations
        module_locs = []
        aa = 1  # start at amino acid 1
        for mod in modules:   # loop thorugh the modules
            module_locs.append((aa, aa + LENDICT[mod] - 1)) # add a pair (start, end)
            aa += LENDICT[mod] # move forward the length of the module
        MODULE_LOCS[rp] = module_locs

        # create the dictionary of joint locations
        for j in range(3):
            joint = ' '.join(modules[j:j+3])
            if joint in JOINT_LOCS:
                JOINT_LOCS[joint].append((i+1, j))
            else:
                JOINT_LOCS[joint] = [(i+1,j)]

# functions for getting rp numbers, protein and joint strings

def joint_locations(joint):
    """
    All instances of a joint in the data set. The three joints in a protein
    are indexed by 0 (Ncap), 1 and 2 (Ccap).

    eg. joint_locations('D54_j1_D79 D79 D79') -> (87, 1)

    means the joint D54_j1_D79 D79 D79 appears in rp87 joint 1
    """
    return JOINT_LOCS[joint]

def protein_string(rp):
    """
    Get the modules of the protein as a string from the rp number

    e.g. protein_string(343) -> 'NcapD14 D14_j1_D14 D14_j1_D14 D14_j1_D81 CcapD81'
    """
    return PROTEINS['rp{}'.format(rp)]

def get_rp(protein_string):
    """
    Get the rp number from the string containing all modules of the protein
    separated by spaces.

    e.g. get_rp('NcapD14 D14_j1_D14 D14_j1_D14 D14_j1_D81 CcapD81') -> 'rp343'
    """
    for rp, string in PROTEINS.items():
        if protein_string == string:
            return rp

## Functions for interacting with pdb files ########################

def pdb_filename(rp, sim):
    """
    Get the filename of a simulation given the protein number 'rp'
    and simulation number 'sim'
    e.g. pdb_filename(1,1) -> '.../1/rp1_0001.pdb'
    """
    path = r'{}\{}\rp{}_{:04}.pdb'.format(PDBPATH, rp, rp, sim)
    #print(path)
    return path


def get_CA_coords(rp, sim):
    """
    Get the coordinates of all the CA atoms in the pdbfile. Returns
    a list of 5 matrices containing the locations of the CA atoms
    in each module.
    """
    # import the pdb file
    parser = PDBParser()
    structure = parser.get_structure('rp', pdb_filename(rp, sim))
    model = structure[0]

    # residues contains each amino acid
    # residues[1] is the first amino acid
    residues = model['a']

    CA_coords = []
    protein_name = 'rp{}'.format(rp)

    for start, end in MODULE_LOCS[protein_name]:
        module_CAs = []
        for i in range(start, end + 1):
            # go through each amino acid
            aa = residues[i]
            # find the CA atom
            atom = aa['CA']
            # add the x,y,z coordinates of CA to the list
            module_CAs.append(atom.get_coord())
        # add the module to the full matrix of coordinates
        CA_coords.append(np.stack(module_CAs))
    return CA_coords


### Functions for extracting centroids and endpoints of modules ######

def get_centroid_vector(module):
    """
    Get the centroid of a module (a numpy matrix of coordinates) as a vector

    eg. x = get_CA_coords(1,1)
    v = get_centroid_vector(x[0])
    """
    x = np.mean(module[:,0])
    y = np.mean(module[:,1])
    z = np.mean(module[:,2])
    return Vector(x, y, z)


def get_centroids_from_modules(modules):
    """
    Get the centroids of the list of modules as a list
    of vectors

    e.g
    x = get_CA_coords(3,50)
    vs = get_centroids_from_modules(x)
    """
    # this is just a quick way of making a list
    return [get_centroid_vector(m) for m in modules]


def get_endpoints_from_modules(modules):
    """
    Get the averaged endpoints of the modules as a list of vectors
    """
    # starting point
    endpoints = [Vector(*modules[0][0])]

    for i in range(len(modules)-1):
        # calculate the midpoint of start/end of interior modules
        midpoint = Vector(*((modules[i][-1] + modules[i+1][0]) / 2))
        endpoints.append(midpoint)

    # ending point
    endpoints.append(Vector(*modules[-1][-1]))
    return endpoints


def get_centroids_and_endpoints_from_modules(modules):
    """
    Get the averaged endpoints and centroids of each module as a
    list of vectors
    """
    centroids = []

    # calculate centroids for each module
    for module in modules:
        centroids.append(np.mean(module, axis=0))

    points_first = modules[0][0]             #very first point of the protein
    points_last = modules[4][-1]             #very last points of the protein

    midpoints = []
    for i in range(0,4):
        # calculate midpoint between start and end of neighbouring modules
        midpoint = (modules[i][-1] + modules[i+1][0])/2

        # add midpoints to a list
        midpoints.append(midpoint)

    points = points_first                   #begin assembling vector with first point of protein
    points = np.append(points,centroids[0]) #add first centroid
    centroids_four = centroids[1:]          #remove the first centroid from the list of other centroids


    # add the rest of the centroids and midpoints
    for i in range(0,4):
        # here, we add the midpoint twice so that list contains groups of three
        points = np.append(points,midpoints[i])
        points = np.append(points,centroids_four[i])

    points = np.append(points,points_last) #finish protein with last point

    # turn list of points into a list of lists, one list for each module representation
    i=0
    group_points=[]
    while i < len(points):
      group_points.append(points[i:i+3])
      i = i + 3

    # isolate x,y,z coordinates
    group_points = np.array(group_points)
    x2 = group_points[:,0]
    y2 = group_points[:,1]
    z2 = group_points[:,2]
    """Vectorising the points """
    vs = []
    for x,y,z in zip(x2,y2,z2):
        v = Vector(x,y,z)
        vs.append(v)

    return x2,y2,z2,vs


### Functions for producing new random protein sequences #################


def randjoint(include_caps=True):
    """
    Returns a random joint as a string. include_caps will include joints
    containing Ncap and Ccap.

    e.g.
    randjoint() -> 'D79 D79_j1_D54 CcapD54'
    """
    if include_caps:
        return random.sample(
                INTERIOR_JOINTS.union(NCAP_JOINTS, CCAP_JOINTS), 1)[0]
    else:
        return random.sample(INTERIOR_JOINTS, 1)[0]

def randnextjoint(joint1):
    """
    Returns a random compatible next joint after joint1 i.e. the last two modules
    of joint1 overlap with the returned value.

    e.g.
    randnextjoint('D14_j1_D14 D14_j2_D14 D14_j1_D76') ->
    'D14_j2_D14 D14_j1_D76 D76'
    """
    possibles = []
    overlap1 = ' '.join(joint1.split()[-2:])
    for joint2 in INTERIOR_JOINTS:
        overlap2 = ' '.join(joint2.split()[:2])
        if overlap1 == overlap2:
           possibles.append(joint2)
    if not possibles:
        print('no connecting joints found')
        return None
    else:
        return random.choice(possibles)

def randprotein(num_modules=5):
    """
    Returns a random protein sequence as a string of the required length of
    modules (>=5).
    e.g.
    randprotein(8) ->
    'NcapD14 D14_j5_D79 D79 D79_j1_D54 D54_j1_D79 D79_j2_D14 D14_j3_D54 CcapD54'
    """
    j1 = randjoint(include_caps=False)
    if num_modules < 5:
        print('Minimum of 5 modules')
        return
    elif num_modules == 5:
        last_module = j1.split()[-1]
    modules = j1.split()
    ncap = 'Ncap{}'.format(modules[0].split('_')[0])
    protein = [ncap, *modules]
    for i in range(num_modules-5):
        j2 = randnextjoint(j1)
        if not j2:
            print('failed')
            return
        last_module = j2.split()[-1]
        protein.append(last_module)
        j1 = j2

    protein.append('Ccap{}'.format(last_module.split('_')[-1]))
    return ' '.join(protein)


### Functions for plotting the full protein

def plot_modules(modules):
    """
    Make a 3D scatter plot of the atoms in the list of modules,
    with different colours for each module. Each module is a list
    of 3D coordinates of atoms.

    e.g.
    x = get_CA_coords(3,5)
    plot_modules(x)
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    colors = ['r','b','g','c','m']
    for i, module in enumerate(modules): # loop through modules and get index i
        x, y, z = module[:,0], module[:,1], module[:,2]
        ax.scatter(x, y, z, color=colors[i])
    plt.show()


def plot_CAs(rp, sim):
    """
    Plot all the CA atoms in a single protein simulation
    eg. plot_CAS(10,15) -> plots rp10 simulation 15
    """
    coords = get_CA_coords(rp, sim)
    plot_modules(coords)
