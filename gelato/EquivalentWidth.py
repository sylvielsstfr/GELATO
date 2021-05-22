""" Generate Rest Equivalent Widths """

# Import packages
import numpy as np
from astropy.io import fits
from astropy.table import Table,hstack

# GELATO
import gelato.CustomModels as CM
import gelato.SpectrumClass as SC

# Constants
C = 299792.458 # km/s

# Calculate Equivalent Width
def EquivalentWidth(spectrum,model,parameters,param_names):

    # Continuum
    if 'PL_Continuum_Coefficient' in param_names:
        continuum = model[0:2]
    else: continuum = model[0]

    # Index where emission lines start
    ind = len(continuum.parameters)

    # Get all continuum heights
    continua = np.ones((parameters.shape[0],len(spectrum.wav)))
    for i,params in enumerate(parameters):
        continuum.parameters = params[:ind]
        continua[i] = continuum(spectrum.wav)

    # Empty EW array
    REWs = np.ones((parameters.shape[0],int((len(param_names)-1-ind)/3),))
    REWs_names = []

    # Iterate over Emission lines
    for i in range(ind,len(param_names)-1,3):

        # Get line parameters
        REWs_names.append(param_names[i].replace('-Redshift','-REW'))
        center = float(REWs_names[-1].split('-')[-2]) 
        opz = 1 + parameters[:,i] # 1 + z
        flux = parameters[:,i+1] # Line flux

        # Get line
        linewav = center*opz
        linewidth = linewav*spectrum.p['LineRegion']/(2*C)

        # Continuum height
        heights = np.ones(parameters.shape[0])
        for j,lwv,lwd,ctm in zip(range(parameters.shape[0]),linewav,linewidth,continua):
            # Continuum height region
            region = np.logical_and(spectrum.wav > lwv - lwd,spectrum.wav < lwv + lwd)
            if region.sum() == 0:
                heights[j] = np.nan
            else:
                heights[j] = np.median(ctm[region])

        # Get REW
        REWs[:,int((i-ind)/3)] = np.abs(flux/(heights*opz))

    # Return combined Results
    return np.hstack((parameters,REWs)),param_names+REWs_names

# Plot from results
def EWfromresults(params,path,z):

    if params["Verbose"]:
        print("Measuring texture:",path.split('/')[-1])

    ## Load in Spectrum ##
    spectrum = SC.Spectrum(path,z,params)

    if spectrum.regions != []:

        # Load name and parameters
        fname = params['OutFolder']+path.split('/')[-1].replace('.fits','')+'-results.fits'
        parameters = Table.read(fname)
        names = parameters.colnames

        # Dont add if already has EWs and no overwrite
        for n in names:
            if 'EW' in n:
                if not params['Overwrite']:
                    print('Texture already measured:',path.split('/')[-1])
                    return
                else: 
                    parameters = parameters[[n for n in names if 'EW' not in n]]
                    names = parameters.colnames
                break
        
        # Put parameters into the form we need
        parameters = np.array([parameters[n] for n in names]).T

        ## Create model ##
        # Add continuum
        model = CM.SSPContinuum(spectrum)
        model.set_region(np.ones(spectrum.wav.shape,dtype=bool))
        if 'PL_Continuum_Coefficient' in names:
            model += CM.PowerLawContinuum(spectrum)

        # Add spectral lines
        ind = len(model.parameters) # index where emission lines begin
        for i in range(ind,len(names)-1,3):
            center = float(names[i].split('-')[-2])
            model += CM.SpectralFeature(center,spectrum)
        
        # Calculate REWs
        parameters,param_names = EquivalentWidth(spectrum,model,parameters,names)

        # Turn into FITS table
        parameters = Table(data=parameters,names=param_names)
        
    if params["Verbose"]:
        print("Texture measured:",path.split('/')[-1])