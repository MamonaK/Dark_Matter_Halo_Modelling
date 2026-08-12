# Dark_Matter_Halo_Modelling

## Project Description
This project investigates dark matter halo models and their ability to reproduce observed galactic rotation curves. The enclosed mass of each halo is formulated as a first-order ODE and solved numerically using the fourth-order Runge--Kutta (RK4) method. Cored-isothermal, Navarro--Frenk--White (NFW), and Burkert density profiles are compared with rotation-curve data from the SPARC database using reduced $\chi^2$ fitting. The project also validates the RK4 implementation against the analytical solution for the cored-isothermal profile and examines how different halo density structures affect galactic rotation curves.

## Getting Started:
1. Clone the repository.
2. Install the required Python packages.
3. Run main.py to verify the SPARC data loads correctly.
4. See DARKMATTERMODELLING(1) for code of density profiles, residuals, and chi-squared fitting
5. See DM_Halo_rk4(1) for source code RK4 method on Cored-Isothermal density profile


## Contributors
Henna Sohail, Chloe Skinner, Mamona Khan
