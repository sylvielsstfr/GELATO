""" Statistical F-Test Package """
import numpy as np
import scipy.stats as stats

debug_level =  1

# Preform an f-test
def FTest(model1,x1,model2,x2,spectrum,args):

    if spectrum.p["Verbose"] and debug_level> 0:
        n1 = model1.nparams()
        n2 = model2.nparams()
        msg = f"++++ FTest model1 : n = {n1}, model2 : n = {n2}"
        print(msg)

    # Find full and reduced model
    if (model1.nparams() < model2.nparams()):
        model_full = model2
        x_full = x2
        model_redu = model1
        x_redu = x1
        if spectrum.p["Verbose"] and debug_level> 0:
            print("++++ FTest : reduced model is model 1, full model is model 2")
    else: 
        model_full = model1
        x_full = x1
        model_redu = model2
        x_full = x2
        if spectrum.p["Verbose"] and debug_level> 0:
            print("++++ FTest reduced model is model 2, full model is model 1")

    # Degrees of freedom
    N = args[0].size

    df_redu = N - model_redu.nparams()
    df_full = N - model_full.nparams()

    # Calculate Chi2
    RSS_redu = Chi2(model_redu,x_redu,args)
    RSS_full = Chi2(model_full,x_full,args)

    if spectrum.p["Verbose"] and debug_level> 0:
        print(f"++++ FTest :: Chi2 : redu model = {RSS_redu}, full model = {RSS_full}")

    # F-test
    F_value = ((RSS_redu - RSS_full)/(df_redu - df_full))/(RSS_full/df_full)
    F_distrib = stats.f(df_redu - df_full,df_full)
    acceptFlag = F_distrib.cdf(F_value) > spectrum.p['FThresh']

    if spectrum.p["Verbose"] and debug_level> 0:
        print(f"++++ FTest :: F_value = {F_value}, acceptFlag = {acceptFlag}")

    # Greater than threshold?
    return F_distrib.cdf(F_value) > spectrum.p['FThresh']

# Calculate Akaike Information Criterion
def AIC(model,p,args):

    # Correct up to a constant
    return Chi2(model,p,args) + 2*model.nparams()

def Chi2(model,p,args):

    return np.square(model.residual(p,*args)).sum()