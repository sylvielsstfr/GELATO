#! /usr/bin/env python

""" Wrapper for mulitple gelato runs """

# Packages
import os
import copy
import argparse
import numpy as np
from pathlib import Path
from astropy.table import Table

# gelato supporting files
import gelato
import gelato.ConstructParams as CP

# Main Function
if __name__ == "__main__":

    ## Parse Arguements to find Parameter File ##
    parser = argparse.ArgumentParser()
    parser.add_argument('Parameters', type=str, help='Path to parameters file')

    
    parser.add_argument('ObjectList', type=str, help='Path to object list with paths to spectra and their redshifts.')
    args = parser.parse_args()
    p = CP.construct(args.Parameters)
    ## Parse Arguements to find Parameter File ##

    ## Create Directory for Output
    if not os.path.exists(p["OutFolder"]):
        Path(p["OutFolder"]).mkdir(parents=True)

    if p['Verbose']:
        now = gelato.header()

    ## Assemble Objects
    objects = gelato.loadObjects(args.ObjectList)
    ## Assemble Objects

    ## Run gelato ##
    if p['NProcess'] > 1: # Mutlithread
        import multiprocessing as mp
        pool = mp.Pool(processes=p['NProcess'])
        inputs = [(copy.deepcopy(args.Parameters),o['Path'],o['z']) for o in objects]
        pool.starmap(gelato.gelato, inputs)
        pool.close()
        pool.join()
    else: # Single Thread
        for o in objects:
            gelato.gelato(copy.deepcopy(args.Parameters),o['Path'],o['z'])
    ## Run gelato ##

    ## Concatenate Results ##
    if p['Concatenate']:
        import gelato.Concatenate as C
        C.concatfromresults(p,objects)
    ## Concatenate Results ##

    if p['Verbose']:
        gelato.footer(now)