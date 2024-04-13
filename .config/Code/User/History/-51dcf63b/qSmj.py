import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from dataclasses import dataclass

@dataclass
class AircraftParams:
    bw: float
    cw: float
    a2dw: float
    alpha_0w: float
    ew: float
    cd0w: float
    rw: float
    Cmw: float
    bt: float
    ct: float
    cef: float
    lt: float
    tt: float
    ih: float
    et: float
    a2dt: float
    alpha_0t: float
    epsilon0: float
    tau_e: float
    cd0t: float
    rt: float
    Vf: float
    df: float
    lf: float
    htv: float
    ltv: float


class super_cub():

    def __init__(self, params):
        #Wing Parameters
        self.bw = params.bw
        self.cw = params.cw
        self.a2dw = params.a2dw
        self.alpha_0w = params.alpha_0w
        self.ew = params.ew
        self.cd0w = params.cd0w
        self.rw = params.rw
        self.Cmw = params.Cmw

        self.Sw = self.bw*self.cw
        self.ARw = self.bw**2/self.Sw
        self.a3dw = self.a2dw/(1+(self.a2dw)/(np.pi*self.ew*self.ARw))

        #Tail Parameters
        self.bt = params.bt
        self.ct = params.ct
        self.ce = params.cef*self.ct
        self.lt = params.lt
        self.tt = params.tt
        self.ih = params.ih
        self.tt = params.tt
        self.a2dt = params.a2dt
        self.alpha_0t = params.alpha_0t
        self.epsilon0 = params.epsilon0
        self.tau_e = params.tau_e
        self.cd0t = params.cd0t
        self.rt = params.rt

        self.St = self.bt*self.ct
        self.ARt = self.bt**2/self.St
        self.a3dt = self.a2dt/(1+(self.a2dt)/(np.pi*self.et*self.ARt))

        #Fuselage Parameters
        self.df = params.df
        self.lf = params.lf
        self.Swetf = 2*np.pi*self.df*self.lf
        self.Swetf = 2.544; self.Vf =0.062831
        self.df = 0.2; self.lf = 2

        #Vertical Tail Parameters
        self.htv = 0.2; self.ltv = 0.8
        self.Swetv = self.htv*self.ltv
        
    def get_nh(self, alpha_deg):
        if alpha_deg<15:
            return 0.9
        else: 
            return 0.9-0.035*(alpha_deg-15)

    def get_fuselage_Cd(self, Re_c):
        """
        Returns the fuselage drag coefficient given the Reynolds number and the altitude.
        """
        
        if Re_c<1e5:
            return (1.328/Re_c**0.5)*(1+(self.df/self.lf)**1.5)+0.11*(self.df/self.lf)
        else:
            return (0.074/Re_c**0.2)*(1+ 1.5*(self.df/self.lf)**1.5 +  7*(self.df/self.lf)**3)

        
    def get_vertical_tail_Cd(self, Re_c):
        """
        Returns the vertical tail drag coefficient given the angle of attack (degrees).
        """
        if Re_c<1e6:
            return 1e-3 
        else:
            return 5e-3

    def get_coefficients(self, alpha_deg, del_e_deg, Re_c, h):
        """
        Returns the whole aircraft CL, CD, Cm, given the angle of attack (degrees), elevator deflection (degrees), Reynolds number, and center of gravity location(Xcg/c).
        """
        Clw = self.a3dw*(alpha_deg-self.alpha_0w)
        Clt = self.a3dt*(2*Clw/(np.pi*self.et*self.ARw) + self.ih +0.7*del_e_deg)
        Cl = Clw + self.get_nh(alpha_deg)*Clt*(self.St/self.Sw)
    
        Cdw = self.cd0w + (self.rw + 1/(np.pi*self.ARw*self.ew))*Clw**2
        Cdt = self.cd0t + (self.rt + 1/(np.pi*self.ARt*self.et))*Clt**2
        Cd = Cdw + Cdt\
            +self.get_fuselage_Cd(Re_c)*self.Swetf/self.Sw 
            # +self.get_vertical_tail_Cd(Re_c)*(self.Swetv/self.Sw)
        Cm = self.Cmw + (2*self.Vf*np.pi/180)*alpha_deg/(self.Sw*self.cw) \
            +self.a3dw*(alpha_deg-self.alpha_0w)*(h-0.25) \
            -self.lt*self.get_nh(alpha_deg)*self.a3dt*(alpha_deg + 2*Clw/(np.pi*self.et*self.ARt)+0.7*del_e_deg)*self.St/(self.Sw*self.cw)

        return Cm, Cl, Cd
    
if __name__ == "__main__":
    #Test
    alphas = np.linspace(-6.27, 15, 100)
    del_e = np.linspace(-10, 10, 5)
    Re_c = 500000
    airplane = super_cub('xf-n2415-il-500000.csv')

    Cms = np.zeros((len(del_e), len(alphas)))
    Cls = np.zeros((len(del_e), len(alphas)))
    Cds = np.zeros((len(del_e), len(alphas)))

    for i in range(len(del_e)):
        for j in range(len(alphas)):
            Cms[i,j] = airplane.get_coefficients(alphas[j], del_e[i], Re_c, 0.4)[0]
            Cls[i,j] = airplane.get_coefficients(alphas[j], del_e[i], Re_c, 0.4)[1]
            Cds[i,j] = airplane.get_coefficients(alphas[j], del_e[i], Re_c, 0.4)[2]
        
    plt.style.use('default')   # Set the aesthetic style of the plots

    # Define the del_e values
    del_e_values = np.linspace(-10, 10, 5)

    fig = plt.figure()
    # Plot Cm vs alphas for each del_e value
    for i, del_e in enumerate(del_e_values):
        plt.plot(alphas, Cms[i], label=f"$\delta_e$ = {del_e}")
        
    # Set the plot title and labels
    plt.title("$C_m$ vs $\\alpha$")
    plt.xlabel("$\\alpha$")
    plt.ylabel("$C_m$")

    # Add a legend
    plt.legend()
    plt.grid(True)
    # Show the plot
    plt.savefig('Cm_vs_alpha.jpg', dpi=300)
    plt.show()

    fig = plt.figure()
    # Plot Cl vs alphas for each del_e value
    for i, del_e in enumerate(del_e_values):
        plt.plot(alphas, Cls[i], label=f"$\delta_e$ = {del_e}")
        
    # Set the plot title and labels
    plt.title("$C_l$ vs $\\alpha$")
    plt.xlabel("$\\alpha$")
    plt.ylabel("$C_l$")

    # Add a legend
    plt.legend()
    plt.grid(True)
    # Show the plot
    plt.savefig('Cl_vs_alpha.jpg', dpi=300)
    plt.show()

    fig = plt.figure()
    # Plot Cl vs Cd for each del_e value
    for i, del_e in enumerate(del_e_values):
        plt.plot(Cds[i], Cls[i], label=f"$\delta_e$ = {del_e}")
        
    # Set the plot title and labels
    plt.title("$C_l$ vs $C_d$")
    plt.xlabel("$C_d$")
    plt.ylabel("$C_l$")

    # Add a legend
    plt.legend()
    plt.grid(True)
    # Show the plot
    plt.savefig('Cl_vs_Cd.jpg', dpi=300)
    plt.show()