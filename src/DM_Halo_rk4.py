# ode isothermal
def dMdr(r, rho, rc):
    return 4*np.pi*r**2*rho / (1 + (r/rc)**2)

# rk4
def rk4(r, rho, rc):
    M = np.zeros(len(r))
    for i in range(len(r)-1):
        h = r[i+1] - r[i]
        k1 = dMdr(r[i], rho, rc)
        k2 = dMdr(r[i] + h/2, rho, rc)
        k3 = dMdr(r[i] + h/2, rho, rc)
        k4 = dMdr(r[i] + h, rho, rc)
        M[i+1] = M[i] + h/6*(k1 + 2*k2 + 2*k3 + k4)

    return M

# apply galaxy
galaxy_ode = next(g for g in galaxies if g["name"] == "NGC3741")
rho_NGC3741, rc_NGC3741 = fitgalaxy(galaxy_ode, isothermal, p0=(1e7, 5)).x

r = np.linspace(0, max(galaxy_ode["R"]), 100)
M_rk4 = rk4(r, rho_NGC3741, rc_NGC3741)

# convert mass to velocitrt
V_rk4 = np.zeros(len(r))

V_rk4[1:] = np.sqrt(
    G*M_rk4[1:] / r[1:]
)

# compare analytical w rk4
V_analytic = np.zeros(len(r))
V_analytic[1:] = isothermal(
    r[1:], rho_NGC3741, rc_NGC3741
)

plt.figure(figsize=(8, 5))
plt.plot(r, V_analytic, linewidth=2.5, label="Analytical")
plt.plot(r, V_rk4, "--", linewidth=2, label="RK4")
plt.xlabel("Radius (kpc)", fontsize=11)
plt.ylabel("Halo Velocity (km/s)", fontsize=11)
plt.title("NGC3741 Cored Isothermal Halo", fontsize=13)
plt.grid(True, linestyle="--", alpha=0.4)
plt.legend(
    loc="lower right",
    frameon=True,
    fontsize=10
)
plt.xlim(0, max(r))
plt.ylim(bottom=0)
plt.tight_layout()
plt.show()


def isothermal (r,rho,rc): #Isothermal profile for scipy
    innerterm = 1-(rc/r)*np.arctan(r/rc)
    outerterm = 4*np.pi*G*rho*rc**2*innerterm
    return np.sqrt(outerterm)


# RK4 convergence test
Rmax = max(galaxy_ode["R"])

# #of equal radial intervals
n_steps_list = [20, 40, 80, 160]

M_exact = isothermal(
    Rmax,
    rho_NGC3741,
    rc_NGC3741
)
rows = []

for n_steps in n_steps_list:
    r_test = np.linspace(0, Rmax, n_steps + 1)
    M_test = rk4(
        r_test,
        rho_NGC3741,
        rc_NGC3741
    )
    h = Rmax / n_steps

    # compare numerical and analytical mass at the final radius
    error = abs(M_test[-1] - M_exact)

    rows.append({
        "N": n_steps,
        "h (kpc)": h,
        "Absolute mass error (Msun)": error
    })


# error ratios and convergence order
for i in range(len(rows)):

    if i < len(rows) - 1:

        E_h = rows[i]["Absolute mass error (Msun)"]
        E_h2 = rows[i + 1]["Absolute mass error (Msun)"]

        ratio = E_h / E_h2
        order = np.log2(ratio)

        rows[i]["Error ratio"] = ratio
        rows[i]["Observed order"] = order

    else:
        rows[i]["Error ratio"] = np.nan
        rows[i]["Observed order"] = np.nan

convergence_table = pd.DataFrame(rows)

print(convergence_table.to_string(index=False))

#latex for report
print(
    convergence_table.to_latex(
        index=False,
        float_format="%.4g",
        na_rep="--",
        caption="RK4 convergence test for the cored-isothermal enclosed-mass ODE using NGC3741.",
        label="tab:rk4_convergence"
    )
)