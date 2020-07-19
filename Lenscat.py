#
# lenscat.py
#
# Expanded Function module of catalogue for strong lensing demos
#
#
# Copyright 2018 by Antonio Herrera Martin 
# Creative Commons Attribution-Noncommercial-ShareAlike 3.0 license applies:
# http://creativecommons.org/licenses/by-nc-sa/3.0/
# All redistributions, modified or otherwise, must include this
# original copyright notice, licensing statement, and disclaimer.
# DISCLAIMER: ABSOLUTELY NO WARRANTY EXPRESS OR IMPLIED.
# AUTHOR ASSUMES NO LIABILITY IN CONNECTION WITH THIS COMPUTER CODE.
#
# Copyright 2009 by Adam S. Bolton
# Creative Commons Attribution-Noncommercial-ShareAlike 3.0 license applies:
# http://creativecommons.org/licenses/by-nc-sa/3.0/
# All redistributions, modified or otherwise, must include this
# original copyright notice, licensing statement, and disclaimer.
# DISCLAIMER: ABSOLUTELY NO WARRANTY EXPRESS OR IMPLIED.
# AUTHOR ASSUMES NO LIABILITY IN CONNECTION WITH THIS COMPUTER CODE.
#
# 

import numpy as N

def xy_rotate(x, y, xcen, ycen, phi):
    """
    NAME: xy_rotate

    PURPOSE: Transform input (x, y) coordiantes into the frame of a new
             (x, y) coordinate system that has its origin at the point
             (xcen, ycen) in the old system, and whose x-axis is rotated
             c.c.w. by phi degrees with respect to the original x axis.

    USAGE: (xnew,ynew) = xy_rotate(x, y, xcen, ycen, phi)

    ARGUMENTS:
      x, y: numpy ndarrays with (hopefully) matching sizes
            giving coordinates in the old system
      xcen: old-system x coordinate of the new origin
      ycen: old-system y coordinate of the new origin
      phi: angle c.c.w. in degrees from old x to new x axis

    RETURNS: 2-item tuple containing new x and y coordinate arrays

    WRITTEN: Adam S. Bolton, U. of Utah, 2009
    """
    phirad = N.deg2rad(phi)
    xnew = (x - xcen) * N.cos(phirad) + (y - ycen) * N.sin(phirad)
    ynew = (y - ycen) * N.cos(phirad) - (x - xcen) * N.sin(phirad)
    return (xnew,ynew)

def gauss_2d(x, y, par):
    """
    NAME: gauss_2d

    PURPOSE: Implement 2D Gaussian function

    USAGE: z = gauss_2d(x, y, par)

    ARGUMENTS:
      x, y: vecors or images of coordinates;
            should be matching numpy ndarrays
      par: vector of parameters, defined as follows:
        par[0]: amplitude
        par[1]: intermediate-axis sigma
        par[2]: x-center
        par[3]: y-center
        par[4]: axis ratio
        par[5]: c.c.w. major-axis rotation w.r.t. x-axis
        
    RETURNS: 2D Gaussian evaluated at x-y coords

    NOTE: amplitude = 1 is not normalized, but rather has max = 1

    WRITTEN: Adam S. Bolton, U. of Utah, 2009
    """
    (xnew,ynew) = xy_rotate(x, y, par[2], par[3], par[5])
    r_ell_sq = ((xnew**2)*par[4] + (ynew**2)/par[4]) / N.abs(par[1])**2
    return par[0] * N.exp(-0.5*r_ell_sq)

def sie_grad(x, y, par):
    """
    NAME: sie_grad

    PURPOSE: compute the deflection of an SIE potential

    USAGE: (xg, yg) = sie_grad(x, y, par)

    ARGUMENTS:
      x, y: vectors or images of coordinates;
            should be matching numpy ndarrays
      par: vector of parameters with 1 to 5 elements, defined as follows:
        par[0]: lens strength, or 'Einstein radius'
        par[1]: (optional) x-center (default = 0.0)
        par[2]: (optional) y-center (default = 0.0)
        par[3]: (optional) axis ratio (default=1.0)
        par[4]: (optional) major axis Position Angle
                in degrees c.c.w. of x axis. (default = 0.0)

    RETURNS: tuple (xg, yg) of gradients at the positions (x, y)

    NOTES: This routine implements an 'intermediate-axis' convention.
      Analytic forms for the SIE potential can be found in:
        Kassiola & Kovner 1993, ApJ, 417, 450
        Kormann et al. 1994, A&A, 284, 285
        Keeton & Kochanek 1998, ApJ, 495, 157
      The parameter-order convention in this routine differs from that
      of a previous IDL routine of the same name by ASB.

    WRITTEN: Adam S. Bolton, U of Utah, 2009
    """
    # Set parameters:
    b = N.abs(par[0]) # can't be negative!!!
    xzero = 0. if (len(par) < 2) else par[1]
    yzero = 0. if (len(par) < 3) else par[2]
    q = 1. if (len(par) < 4) else N.abs(par[3])
    phiq = 0. if (len(par) < 5) else par[4]
    eps = 0.001 # for sqrt(1/q - q) < eps, a limit expression is used.
    # Handle q > 1 gracefully:
    if (q > 1.):
        q = 1.0 / q
        phiq = phiq + 90.0
    # Go into shifted coordinats of the potential:
    phirad = N.deg2rad(phiq)
    xsie = (x-xzero) * N.cos(phirad) + (y-yzero) * N.sin(phirad)
    ysie = (y-yzero) * N.cos(phirad) - (x-xzero) * N.sin(phirad)
    # Compute potential gradient in the transformed system:
    r_ell = N.sqrt(q * xsie**2 + ysie**2 / q)
    qfact = N.sqrt(1./q - q)
    # (r_ell == 0) terms prevent divide-by-zero problems
    if (qfact >= eps):
        xtg = (b/qfact) * N.arctan(qfact * xsie / (r_ell + (r_ell == 0)))
        ytg = (b/qfact) * N.arctanh(qfact * ysie / (r_ell + (r_ell == 0)))
    else:
        xtg = b * xsie / (r_ell + (r_ell == 0))
        ytg = b * ysie / (r_ell + (r_ell == 0))
    # Transform back to un-rotated system:
    xg = xtg * N.cos(phirad) - ytg * N.sin(phirad)
    yg = ytg * N.cos(phirad) + xtg * N.sin(phirad)
    # Return value:
    return (xg, yg)

def pm_grad(x, y, par):
    """
    NAME: pm_grad

    PURPOSE: compute the deflection of an point mass potential

    USAGE: (xg, yg) = pm_grad(x, y, par)

    ARGUMENTS:
      x, y: vectors or images of coordinates;
            should be matching numpy ndarrays
      par: vector of parameters with 1 to 3 elements, defined as follows:
        par[0]: lens strength, or 'Einstein radius'
        par[1]: (optional) x-center (default = 0.0)
        par[2]: (optional) y-center (default = 0.0)
 

    RETURNS: tuple (xg, yg) of gradients at the positions (x, y)


    WRITTEN: Antonio Herrera Martin, U of Glasgow, 2017
    """
    # Set parameters:
    b = N.abs(par[0]) # can't be negative!!!
    xzero = 0. if (len(par) < 2) else par[1]
    yzero = 0. if (len(par) < 3) else par[2]
    # Go into shifted the center of potential:
    xpm = (x-xzero) 
    ypm = (y-yzero)  
    # Arrange in polar coordinates for simplicity:
    r_pm = N.sqrt( xpm**2 + ypm**2) 
    phi = N.arctan2(ypm, xpm)
    # (r_pm == 0 ) prevents division by zero problems
    # alpha is the deflection angle from beta = theta - alpha
    alpha = b**2/(r_pm+(r_pm == 0))
    # obtain the carthesian positions.
    xg = alpha*N.cos(phi)
    yg = alpha*N.sin(phi)
    # Return value:
    return (xg, yg)

def sol_grad(x, y, par):
    """
    NAME: sol_grad

    PURPOSE: compute the deflection of an soliton-only potential

    USAGE: (xg, yg) = sol_grad(x, y, par)

    ARGUMENTS:
      x, y: vectors or images of coordinates;
            should be matching numpy ndarrays
      par: vector of parameters with 1 to 5 elements, defined as follows:
        par[0]: lens strength, or parameter 'Einstein radius' (Only used if no lambda is provided)
        par[1]: Parameter lambda. ( If it is not provided or set to 0.0, it will be obtained from Einstein radius. )
        par[2]: (optional) x-center (default = 0.0)
        par[3]: (optional) y-center (default = 0.0)
         

    RETURNS: tuple (xg, yg) of gradients at the positions (x, y)


    WRITTEN: Antonio Herrera Martin, U of Glasgow, 2017
    """
    # Set Constants:
    lambdacrit = 2048./(429.*N.pi**2)
    # Set parameters:
    b = N.abs(par[0]) # can't be negative!!!
    me = (2./(13.*lambdacrit))*((1+b**2)**(13./2)-1.)/(1.+b**2)**(13./2) 
    lambdapar = b**2/me if (len(par) < 2 or par[1] == 0. ) else par[1]
    if (len(par) < 2 or par[1] == 0. ):
        print "Lambda for Einstein radius = ", lambdapar
    xzero = 0. if (len(par) < 3) else par[2]
    yzero = 0. if (len(par) < 4) else par[3]
    
    # Go into shifted coordinats of the potential:
    xpm = (x-xzero) 
    ypm = (y-yzero)  
    # Arrange in polar coordinates for simplicity:
    theta = N.sqrt( xpm**2 + ypm**2) 
    #theta = thetau/rs
    phi = N.arctan2(ypm, xpm)
    # (theta == 0 ) prevents division by zero problems
    # m(theta) is the reduced mass in and theta is in polar coordinates
    m = (2./(13.*lambdacrit))*((1.+theta**2)**(13./2)-1.)/(1.+theta**2)**(13./2)
     # alpha is the deflection angle from beta = theta - alpha
    alpha = lambdapar*m/(theta+(theta == 0))
    # obtain the carthesian positions.
    xg = alpha*N.cos(phi)
    yg = alpha*N.sin(phi)
    # Return value:
    return (xg, yg)

def wave_grad(x, y, par):
    """
    NAME: wave_grad

    PURPOSE: compute the deflection of an soliton+NFW tail potential

    USAGE: (xg, yg) = wave_grad(x, y, par)

    ARGUMENTS:
      x, y: vectors or images of coordinates;
            should be matching numpy ndarrays
      par: vector of parameters with 1 to 6 elements, defined as follows:
        par[0]: lens strength, or parameter 'Einstein radius'
        par[1]: Parameter lambda. ( If it is not provided or set to 0.0, it will be obtained from Einstein radius. )
        par[2]: Transition radius. It must be different from zero
        par[3]: Alpha. Cannot be Zero nor Negative
        par[4]: (optional) x-center (default = 0.0)
        par[5]: (optional) y-center (default = 0.0)

    RETURNS: tuple (xg, yg) of gradients at the positions (x, y)


    WRITTEN: Antonio Herrera Martin, U of Glasgow, 2018
    """
    # Set parameters:
    b = N.abs(par[0]) # can't be negative!!!
    repsilon =  N.abs(par[2]) #can't be negative!!
    alpha = 1e-11 if(N.abs(par[3]) == 0. ) else N.abs(par[3])
    me = mwave(b,repsilon,alpha)
    if (me) == 0.:
        raise ValueError("No deflection at Einstein angle. Check parameters.")
    lambdapar = b**2/me if (len(par) < 4 or par[1] == 0. ) else par[1]
    if (len(par) < 4 or par[1] == 0. ):
        print "Lambda for Einstein radius = ", lambdapar
    xzero = 0. if (len(par) < 5) else par[4]
    yzero = 0. if (len(par) < 6) else par[5]

    # Go into shifted coordinats of the potential:
    xpm = (x-xzero)  
    ypm = (y-yzero)   
    # Arrange in polar coordinates for simplicity:
    theta = N.sqrt( xpm**2 + ypm**2) 
    #theta = thetau/rs
    phi = N.arctan2(ypm, xpm)
    # (theta == 0 ) prevents division by zero problems
    # m(theta) is the reduced mass in and theta is in polar coordinates
    if (type(theta).__module__ == N.__name__):   
        mwavef = N.vectorize(mwave)
    else:
        mwavef = mwave
    m = mwavef(theta,repsilon,alpha)
    # alpha is the deflection angle from beta = theta - alpha
    alpha = lambdapar*m/(theta+(theta == 0))
    # obtain the carthesian positions.
    xg = alpha*N.cos(phi)
    yg = alpha*N.sin(phi)
    # Return value:
    return (xg, yg)


#########################################################
#  The following the lensing functions for              #
#  the wave dark matter profile.                        #  
#  More info: http://theses.gla.ac.uk/9027/             #
#########################################################               


def Bfunc(u):
    """
    NAME: Bfunc

    PURPOSE: Calculate the integral for the soliton part for
             a wave dark matter. 
             Only the integral cos^14 u is obtained. 
             Several constants are needed later for complete
             the soliton part.

    USAGE: x = Bfunc(u)

    ARGUMENTS: u is a number

    RETURNS: The value of the integral evaluated from 0 to u

    WRITTEN: Antonio Herrera Martin, U of Glasgow, 2017
    """
    factor = 1./122880.
    out = factor*( \
                  27720.*u \
                  + 23760.*N.sin(2.*u)\
                  + 7425.*N.sin(4.*u)\
                  + 2200.*N.sin(6.*u)\
                  + 496.*N.sin(8.*u)\
                  + 72.*N.sin(10.*u)\
                  + 5.*N.sin(12.*u)\
                 )
    return out;

def func(x,y): 
    """
    NAME: func

    PURPOSE: Calculate the incomplete nfw part of the profile.
             Several constants are added later.
             

    USAGE: x = func(x,y)

    ARGUMENTS: These must be provided as:
               x = alpha*N.absolute(xi)
               y = alpha*re
               where alpha is a positive parameter,
                     re is the transition radius,
                     xi is the value of evaluation.

    RETURNS: The valuation of the function to be added for the complete
             profile.

    WRITTEN: Antonio Herrera Martin, U of Glasgow, 2017
    """
    if x>y: 
        raise ValueError("The value of x cannot be bigger than y.")
    if 0 < x < 1.:
        out = (2./ N.sqrt(1.-x**2))*N.arctanh(N.sqrt((1.-x**2))/(1.+y+N.sqrt(y**2-x**2)))
        return out;
    elif x > 1. :
        out = (2./ N.sqrt(x**2-1.))*N.arctan(N.sqrt((x**2-1.))/(1.+y+N.sqrt(y**2-x**2)))
        return out;
    elif x == 1. :
        out =  1. - N.sqrt(y-1.)/N.sqrt(y+1)
        return out;
    else :
        return 0.;

def mnfw(x): 
    """
    NAME: mnfw

    PURPOSE: Calculates the surface surface mass for a
             NFW universal profile.
             
    USAGE: x = mnfw(x)

    ARGUMENTS: x is the normalized radius to evaluate.

    RETURNS: Surface mass for NFW at the point x.

    WRITTEN: Antonio Herrera Martin, U of Glasgow, 2017
    """
    if 0< x < 1.:
        out = (2./ N.sqrt(1.-x**2))*N.arctanh(N.sqrt((1.-x)/(1.+x)))+N.log(x/2.)
        return out;
    elif x == 1. :
        out = 1. + N.log(1./2.)
        return out;
    elif x > 1. :
        out = (2./ N.sqrt(x**2-1.))*N.arctan(N.sqrt((x-1)/(1.+x)))+N.log(x/2.)
        return out;
    else :
        return 0.;

def rhonfw(re,alpha): 
    """
    NAME: rhonfw

    PURPOSE: Calculate the ratio of the NFW and soliton sections.

    USAGE: x = rhonfw(re,alpha)

    ARGUMENTS: alpha is a positive parameter,
                re is the transition radius.

    RETURNS: The value of the ratio of the sections.

    WRITTEN: Antonio Herrera Martin, U of Glasgow, 2017
    """
    return alpha*re*(1.+alpha*re)**2/(1.+re**2)**8;

def mlow(xi,re,alpha): 
    """
    NAME: mlow

    PURPOSE: Calculate the surface mass for the complete
             wave dark matter profile for xi < re
             

    USAGE: x = mlow(xi,re,alpha)

    ARGUMENTS: These must be provided as:
                alpha is a positive parameter,
                re is the transition radius,
                xi is the value of evaluation.

    RETURNS: The surface mass density at a normalized radius xi, 
             when xi < re.

    WRITTEN: Antonio Herrera Martin, U of Glasgow, 2017
    """ 
    if xi > re:
        raise ValueError("xi cannot be bigger than re in this section.")
    if (re or alpha) < 0.:
        raise ValueError("nor alpha nor re can be negative.")
    x = alpha*N.absolute(xi)
    y = alpha*re
    fac1 = N.arctan(N.sqrt((re**2-xi**2)/(1.+xi**2)))
    out = 4.*N.pi*(\
             (1./14.)*(\
                      Bfunc(N.arctan(re))\
                       -Bfunc(fac1)/(1.+xi**2)**(13./2.)\
                       +(N.sqrt(re**2-xi**2)-re)/(1.+re**2)**7\
                      )\
              +rhonfw(re,alpha)/(alpha**3)*(\
                                           N.log((y+N.sqrt(y**2-x**2))/(2.*(y+1.)))\
                                           +(y-N.sqrt(y**2-x**2))/(y+1.)+func(x,y))\
             )
    return out;

def mhigh(xi,re,alpha):
    """
    NAME: mhigh

    PURPOSE: Calculate the surface mass for the complete
             wave dark matter profile for xi > re
             

    USAGE: x = mlow(xi,re,alpha)

    ARGUMENTS: These must be provided as:
                alpha is a positive parameter,
                re is the transition radius,
                xi is the value of evaluation.

    RETURNS: The surface mass density at a normalized radius xi, 
             when xi > re.

    WRITTEN: Antonio Herrera Martin, U of Glasgow, 2017
    """
    if xi < re:
        raise ValueError("xi cannot be smaller than re in this section.")
    if (re or alpha) < 0.:
        raise ValueError("nor alpha nor re can be negative.")
    x = alpha*N.absolute(xi)
    y = alpha*re
    out = 4.*N.pi*(\
             (1./14.)*(Bfunc(N.arctan(re))-re/(1.+re)**7)\
              +rhonfw(re,alpha)/(alpha**3)*(N.log(1./(y+1.))+(y)/(y+1.)+mnfw(x)))
    return out;

def mwave(xi,re,alpha):
    """
    NAME: mwave

    PURPOSE: Driver function for the surface mass for the complete
             wave dark matter profile.

    USAGE: x = mwave(xi,re,alpha)

    ARGUMENTS: These must be provided as:
                alpha is a positive parameter,
                re is the transition radius,
                xi is the value of evaluation.

    RETURNS: The surface mass density at a normalized radius xi for 
             wave dark matter profile.
    NOTE: if re == 0, it will return the NFW evaluated at xi disregarding
          alpha.

    WRITTEN: Antonio Herrera Martin, U of Glasgow, 2017
    """
    if (xi < re) and (re != 0) :
        return mlow(xi,re,alpha)
    if (xi >= re) and (re != 0) :
        return mhigh(xi,re,alpha)
    if re == 0:
        return mnfw(xi)

