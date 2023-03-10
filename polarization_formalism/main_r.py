from dependencies_r import *
from scipy.integrate import odeint


def initiate(nu_types, r_i, r_f, E_i, E_f, E_step, E_0, Amplitude):
    y0 = []  # Initial state
    omega = []
    flavor_sign = 1

    E_vec = np.arange(E_i, E_f, E_step)
    n_E = len(E_vec)

    n_f = len(nu_types)

    for i in range(n_E):
        omega.append(Delta_m2/(2*E_vec[i]*10**6))  # eV
        for j in range(n_f):
            if nu_types[j] == "nu_x":
                flavor_sign = -1
            if nu_types[j] == "nu_e":
                flavor_sign = 1
            # nu
            nu_spec = Amplitude[n_f*j] * phi_vec(E_vec[i], E_0[n_f*j], 2.3)*E_step
            y0.append(0)
            y0.append(0)
            y0.append(flavor_sign*nu_spec)
            # nubar
            nu_spec = Amplitude[n_f*j+1] * phi_vec(E_vec[i], E_0[n_f*j+1], 2.3)*E_step
            y0.append(0)
            y0.append(0)
            y0.append(flavor_sign*nu_spec)

    # mu
    mu_0 = (10)*max(omega)
    # r array
    r_step = (2*np.pi/max(omega))/20  # eV⁻¹
    r_i = r_i*eV_to_inv_km  # eV⁻¹
    r_f = r_f*eV_to_inv_km  # eV⁻¹
    r = np.arange(r_i, r_f, r_step)  # eV⁻¹

    return y0, omega, E_vec, r, mu_0, n_f, n_E


def func_Collective_nu(y, time, params):
    omega, mu_opt, mu_0, n_f, n_dim, n_E = params  # unpack parameters
    B = np.array(B(theta_31))
    L = np.array(L())

    r = time/eV_to_inv_km  # From eV⁻¹ to km
    mu = mu(r, mu_opt, mu_0)
    lamb = lam(r, "no", 0)

    derivs = []
    nu, nubar = [], []
    num_diff_nu_compnents = 2*n_f*n_dim

    # Filling [Energy bin][Nu_types][3components]
    for i in range(n_E):
        nu.append([])
        nubar.append([])
        for j in range(n_f):
            nu[i].append([])
            nubar[i].append([])
            for k in range(n_dim):
                # nu
                nu_index = (i*num_diff_nu_compnents)+k+2*j*n_dim
                nu[i][j].append(y[nu_index])
                # nubar
                nubar_index = (i*num_diff_nu_compnents)+(k+n_dim)+2*j*n_dim
                nubar[i][j].append(y[nubar_index])

    # Summed nu and nubar components
    nu_sum, nubar_sum = [], []
    nu_aux = np.transpose(nu, (2, 0, 1))
    nubar_aux = np.transpose(nubar, (2, 0, 1))

    for i in range(n_dim):
        nu_sum.append(sum(map(sum, nu_aux[i])))
        nubar_sum.append(sum(map(sum, nubar_aux[i])))

    B = np.array(B)
    nu_sum = np.array(nu_sum)
    nubar_sum = np.array(nubar_sum)
    # list of dy/dt=f functions
    for i in range(n_E):
        for j in range(n_f):
            # nu
            P_aux = cross_prod(nu[i][j], (B*omega[i]+L*lamb-mu*((nu_sum-nu[i][j])-nubar_sum)))
            # P_aux= cross_prod(nu[i][j],(B*omega[i]+L*lamb-mu*(nu_sum-nubar_sum)))
            for k in range(n_dim):
                derivs.append(P_aux[k])

            # nubar
            P_aux = cross_prod(nubar[i][j], (-1*B*omega[i]+L*lamb-mu*(nu_sum-(nubar_sum-nubar[i][j]))))
            # P_aux= cross_prod(nubar[i][j],(-1*B*omega[i]+L*lamb-mu*(nu_sum-nubar_sum)))
            for k in range(n_dim):
                derivs.append(P_aux[k])

    return derivs


def solver_two_families(nu_types, r_i, r_f, E_i, E_f, E_step, E_0, Amplitude, mass_ord):

    y0, omega, E_vec, r, mu_0, n_f, n_dim, n_E = initiate(
        nu_types, r_i, r_f, E_i, E_f, E_step, E_0, Amplitude)

    if mass_ord == "NH":
        params = np.array(omega), "SN", mu_0, n_f, n_dim, n_E
    elif mass_ord == "IH":
        params = -1*np.array(omega), "SN", mu_0, n_f, n_dim, n_E
    else:
        print("Not a mass ordering option!")
        return 0
    psoln = odeint(func_Collective_nu, y0, r, args=(params,))

    nu, nubar = read_output(psoln, (n_f, n_dim, n_E))
    nu_e_time, nubar_e_time, nu_x_time, nubar_x_time = read_two_flavor_v1(
        nu, nubar)

    r = r/eV_to_inv_km  # From eV⁻¹ to km

    # return nu_e_time,nubar_e_time, nu_x_time,nubar_x_time
    return E_vec, r, mu_0, nu_e_time, nubar_e_time, nu_x_time, nubar_x_time, nu, nubar
