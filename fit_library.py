# Includes
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import pickle
from scipy.optimize import curve_fit
from scipy import integrate
import cv2

# Print precision and DPI precision and TEX rendering in plots

plt.rc('text', usetex=True)
plt.rc('font', family='serif')

np.set_printoptions(precision=3)

DPI = 300

# Parameters placed in the simulation

dx = 0.01
n_scanned_angles = 101
angles = np.linspace(0, np.pi / 2, n_scanned_angles + 1)
dtheta = angles[1] - angles[0]

# Scanned N_turns, here we define at which number of turns we are going to
# "see what happened with the beam"

n_turns = np.array([
    1000, 1200, 1400, 1600, 1800, 2000, 2500, 3000, 3500, 4000, 4500, 5000,
    5500, 6000, 6500, 7000, 7500, 8000, 8500, 9000, 9500, 10000, 12000, 14000,
    16000, 18000, 20000, 25000, 30000, 35000, 40000, 45000, 50000, 55000,
    60000, 65000, 70000, 75000, 80000, 85000, 90000, 95000, 100000, 120000,
    140000, 160000, 180000, 200000, 250000, 300000, 350000, 400000, 450000,
    500000, 550000, 600000, 650000, 700000, 750000, 800000, 850000, 900000,
    950000, 1000000, 1200000, 1400000, 1600000, 1800000, 2000000, 2500000,
    3000000, 3500000, 4000000, 4500000, 5000000, 5500000, 6000000, 6500000,
    7000000, 7500000, 8000000, 8500000, 9000000, 9500000, 10000000
])

# Partition lists for basic angle partitioning, since we will want to check for
# different angle partitions, here we list how deep we want to go.
# if we want to neglect this analysis, keep only the first line and comment the
# others.

partition_lists = [
    [0, np.pi / 2],  # Always always keep this one
    [x for x in np.linspace(0, np.pi / 2, num=3)],
    [x for x in np.linspace(0, np.pi / 2, num=4)],
    [x for x in np.linspace(0, np.pi / 2, num=5)],
    [x for x in np.linspace(0, np.pi / 2, num=6)],
    [x for x in np.linspace(0, np.pi / 2, num=7)],
    [x for x in np.linspace(0, np.pi / 2, num=8)],
    [x for x in np.linspace(0, np.pi / 2, num=9)],
    [x for x in np.linspace(0, np.pi / 2, num=10)],
    [x for x in np.linspace(0, np.pi / 2, num=11)],
    [x for x in np.linspace(0, np.pi / 2, num=12)],
    [x for x in np.linspace(0, np.pi / 2, num=13)]
]

# Not perfect, needs improvement for proper scientific notation!
def error_formatter(meas, err):
    err *= 1.0001
    mag = np.absolute(int(np.log10(err)) - 1)
    return "${0:.{1}f} \pm ".format(meas, mag) + "{0:.{1}f}$".format(err,mag)

################################################################################
################################################################################
### DYNAMIC APERTURE COMPUTATION FUNCTIONS  ####################################
################################################################################
################################################################################


def compute_D(contour_data, section_lenght, d_angle=dtheta):
    '''
    Given an array of distances from an angular scan
    and the total lenght of the section.
    Returns the defined Dynamic Aperture
    '''
    # return integrate.simps(contour_data, dx=d_angle) / (section_lenght)
    return np.average(contour_data)


def compute_D_error(contour_data, d_lenght=dx, d_angle=dtheta):
    '''
    Given the array of distances from an angular scan,
    computes the error estimation
    '''
    first_derivatives = np.asarray(
        [(contour_data[k + 1] - contour_data[k]) / d_angle
         for k in range(len(contour_data) - 1)])
    average_derivative = np.average(first_derivatives)
    return (np.sqrt(d_lenght * d_lenght / 4 +
                    average_derivative**2 * d_angle**2 / 4))


def make_countour_data(data, n_turns, d_lenght):
    """
    Given the simulation data, makes a dictionary with the contour data for
    the given N turns list.
    """
    angle = {}
    for theta in data:
        temp = {}
        for time in n_turns:
            j = 0
            while data[theta][j] >= time:
                j += 1
            temp[time] = ((j - 1) * dx)
        angle[theta] = temp
    return angle


def divide_and_compute(data,
                       n_turns,
                       partition_list=[0, np.pi / 2],
                       d_lenght=dx,
                       d_angle=dtheta):
    '''
    data is a dictionary containing the simulation data.
    n_turns is an array of the times to explore.
    partition_list is the list of partition to analyze separately.
    Returns a dictionary per angles analyzed
    '''
    angle = make_countour_data(data, n_turns, d_lenght)
    result = {}
    for i in range(len(partition_list) - 1):
        n_angles = 0
        D = {}
        Err = {}
        for t in n_turns:
            limit = []
            for theta in sorted(angle):
                if theta >= partition_list[i] and theta <= partition_list[i +
                                                                          1]:
                    limit.append(angle[theta][t])
                    n_angles += 1
            assert n_angles >= 5
            limit = np.asarray(limit)
            D[t] = compute_D(limit, partition_list[i + 1] - partition_list[i],
                             d_angle)
            Err[t] = compute_D_error(limit, d_lenght, d_angle)
        result[(partition_list[i] + partition_list[i + 1]) / 2] = (D, Err)
    return result


################################################################################
################################################################################
###  FIT1 FUNCTIONS  ###########################################################
################################################################################
################################################################################


def FIT1(x, D_inf, B, k):
    """
    The first fitting formula used in the first papers.
    """
    return D_inf + B / np.exp(k * np.log(np.log(x)))

# I needed something here for drawing the error band for the fit parameters.
# this is an easy and lazy solution.

def FIT1_error_max(x, D_inf, D_inf_err, B, B_err, k, k_err):
    #print(D_inf, D_inf_err, B, B_err, k, k_err)
    return max([FIT1(x, D_inf + D_inf_err, B + B_err, k + k_err),
                FIT1(x, D_inf + D_inf_err, B + B_err, k - k_err),
                FIT1(x, D_inf + D_inf_err, B - B_err, k + k_err),
                FIT1(x, D_inf + D_inf_err, B - B_err, k - k_err)])
   

def FIT1_error_min(x, D_inf, D_inf_err, B, B_err, k, k_err):
    #print(D_inf, D_inf_err, B, B_err, k, k_err)
    return min([FIT1(x, D_inf - D_inf_err, B + B_err, k + k_err),
                FIT1(x, D_inf - D_inf_err, B + B_err, k - k_err),
                FIT1(x, D_inf - D_inf_err, B - B_err, k + k_err),
                FIT1(x, D_inf - D_inf_err, B - B_err, k - k_err)])


def non_linear_fit1(data, err_data, n_turns, k_min, k_max, dk, p0D=0, p0B=0):
    """
    Data is a dictionary in the form 
    data[number_of_turns] = current_dynamic_aperture,
    Err_data is a dictionary in the same form,
    n_turns is the list of scanned number of turns,
    k_min and k_max is the interval of exploration,
    dk is the scanning step,
    last two are just the starting points for the fit.
    """
    # I sadly have to redefine the fit function here for using curve_fit
    # properly, really unelegant.
    fit1 = lambda x, D_inf, B: D_inf + B / np.log(x)**k
    chi1 = lambda x, y, sigma, popt: ((1 / (len(n_turns) - 3)) *
                        np.sum(((y - fit1(x, popt[0], popt[1])) / sigma)**2))
    # we want to perform a scan for the different k values.
    explore_k = {}
    for number in np.arange(k_min, k_max + dk, dk):
        if np.absolute(number) > dk / 10.:
            k = number
            try:
                popt, pcov = curve_fit(
                    fit1,
                    n_turns, [data[i] for i in n_turns],
                    p0=[p0D, p0B],
                    sigma=[err_data[i] for i in n_turns])
                explore_k[k] = (popt, 
                                pcov,
                                chi1(n_turns, 
                                    [data[i] for i in n_turns],
                                    [err_data[i] for i in n_turns], 
                                    popt),
                                dk)
            except RuntimeError:
                print("Runtime Error at k = {}".format(k))
    assert len(explore_k) > 0
    return explore_k


def select_best_fit1(parameters):
    """
    Selects the best fit parameters by choosing the minimum chi-squared value.
    """
    best = sorted(parameters.items(), key=lambda kv: kv[1][2])[0]
    return (best[1][0][0],
            np.sqrt(best[1][1][0][0]),
            best[1][0][1],
            np.sqrt(best[1][1][1][1]),
            best[0],
            best[1][3])


def non_linear_fit1_iterated(data, err_data, n_turns,
                             k_min, k_max, dk, n_iterations, p0D=0, p0B=0):
    """
    With this method, we can refine the analysis on the k parameter in a smart
    way. Look on the report for more details.
    Signature is almost the same of non_linear_fit1 but we have now n_iterations
    to regulate.
    """
    all_fits = non_linear_fit1(data, err_data, n_turns,
                               k_min, k_max, dk, p0D, p0B)
    best_fit = select_best_fit1(all_fits)
    for i in range(n_iterations):
        best_fit = select_best_fit1(non_linear_fit1(
            data, err_data, n_turns,
            best_fit[4] - dk, best_fit[4] + dk, dk / 10, p0D, p0B))
        dk /= 10
    return all_fits, best_fit


# Some useful functions for plotting the fit curve by
# passing the parameters
def pass_params_fit1(x, params):
    return FIT1(x, params[0], params[2], params[4])


def pass_params_fit1_min(x, params):
    return FIT1_error_min(x, params[0], params[1],
                          params[2], params[3],
                          params[4], params[5])


def pass_params_fit1_max(x, params):
    return FIT1_error_max(x, params[0], params[1],
                          params[2], params[3],
                          params[4], params[5])

################################################################################
################################################################################
##  FIT2 FUNCTIONS  ############################################################
################################################################################
################################################################################


def FIT2(x, a, b, k):
    return b / np.exp(k * np.log(np.log(a * x)))


# Look in the report for more details about this alternative form
def FIT2_linearized(x, k, B, a): # b = exp(B); a = exp(A)
    return np.exp(B - k * np.log(np.log(a * np.asarray(x))))


# Same as Fit1
def FIT2_linearized_err_max(x, k, k_err, B, B_err, a, a_err):
    return FIT2_linearized(x, k - k_err, B + B_err, a - a_err)


def FIT2_linearized_err_min(x, k, k_err, B, B_err, a, a_err):
    return FIT2_linearized(x, k + k_err, B - B_err, a + a_err)

# This is mostly based on the same mentality of FIT1 fitting procedure,
# but because of many various analysis and different attempts, now it's
# indeed quite a messy spaghetti code.

def non_linear_fit2_fixed_a_and_k(data, err_data, n_turns, a, k, k_err=0., p0B=0.):
    fit2 = lambda x, B: B - k * np.log(np.log(float(a) * x))
    chi2 = lambda x, y, sigma, popt: ((1 / (len(n_turns) - 1)) *
                        np.sum(((y - fit2(x, popt[0])) / sigma)**2))
    working_data = {}
    working_err_data = {}
    # Preprocessing the data
    for label in data:
        working_data[label] = np.log(np.copy(data[label]))
        working_err_data[label] = ((1 / np.copy(data[label])) * 
                                            np.copy(err_data[label]))
    popt, pcov = curve_fit(fit2,
                           n_turns, [working_data[i] for i in n_turns],
                           p0=[p0B],
                           sigma=[working_err_data[i] for i in n_turns])
    return(k, k_err,
           popt[0], np.sqrt(pcov[0][0]),
           a, 0.,
           chi2(n_turns, [working_data[i] for i in n_turns],
                [working_err_data[i] for i in n_turns], popt))
            

def non_linear_fit2_fixedk(data, err_data, n_turns, a_min, a_max, da, k, p0B=0):
    fit2 = lambda x, B: B - k * np.log(np.log(a * x))
    chi2 = lambda x, y, sigma, popt: ((1 / (len(n_turns) - 3)) *
                        np.sum(((y - fit2(x, popt[0])) / sigma)**2))
    explore_a = {}

    working_data = {}
    working_err_data = {}
    # Preprocessing the data
    for label in data:
        working_data[label] = np.log(np.copy(data[label]))
        working_err_data[label] = ((1 / np.copy(data[label])) * 
                                            np.copy(err_data[label]))

    for number in np.arange(a_min, a_max + da, da):
        a = number
        try:
            popt, pcov = curve_fit(fit2,
                                   n_turns, [working_data[i] for i in n_turns],
                                   p0=[p0B],
                                   sigma=[working_err_data[i] for i in n_turns])
            explore_a[a] = (popt, 
                            pcov,
                            chi2(n_turns,
                                 [working_data[i] for i in n_turns],
                                 [working_err_data[i] for i in n_turns], 
                                 popt), 
                            da,
                            k)
        except RuntimeError:
            print("Runtime error with a = {}".format(a))
    assert len(explore_a) > 0
    return explore_a


def non_linear_fit2_final_fixedk(data, err_data, n_turns,
                                 a_min, a_max, da, a_bound, a_default,
                                 k, k_err=0., p0B=0.):
    scale_search = 1
    #print(scale_search)
    all_fits = non_linear_fit2_fixedk(data, err_data, n_turns,
                                      a_min, a_max, da, k, p0B)
    best_fit = select_best_fit2_fixedk(all_fits, k_err)
    while (best_fit[4] >= a_max * scale_search - da * scale_search and
                          scale_search <= a_bound):
        scale_search *= 10
        #print(scale_search)
        if scale_search > a_bound:
            print("Set a = {}".format(a_default))
            return all_fits, non_linear_fit2_fixed_a_and_k(
                                                    data, err_data, n_turns,
                                                    float(a_default), k, k_err, p0B)
        all_fits = non_linear_fit2_fixedk(data, err_data, n_turns,
                                          a_min, a_max * scale_search,
                                          da * scale_search, k, p0B)
        best_fit = select_best_fit2_fixedk(all_fits, k_err)
    return all_fits, best_fit


# At the end of the day, we use this one.
def non_linear_fit2_doublescan(data, err_data, n_turns,
                               k_min, dk, a_min, a_max, da, a_bound, a_default):
    best_fit_list = []
    fit_list = []
    chi_values = []
    k = k_min
    #print(k)
    fit, best_fit = non_linear_fit2_final_fixedk(data, err_data, n_turns,
                                                 a_min, a_max, da, a_bound,
                                                 a_default, k, dk)
    fit_list.append(fit)
    best_fit_list.append(best_fit)
    chi_values.append(best_fit[6])

    while best_fit[5] != 0.:
        k += dk
        #print(k)
        fit, best_fit = non_linear_fit2_final_fixedk(data, err_data, n_turns,
                                                 a_min, a_max, da, a_bound,
                                                 a_default, k, dk)
        if best_fit[5] != 0.:
            fit_list.append(fit)
            best_fit_list.append(best_fit)
            chi_values.append(best_fit[6])

    index = chi_values.index(min(chi_values))
    return fit_list[index], best_fit_list[index]

def select_best_fit2(parameters):
    best = sorted(parameters.items(), key=lambda kv: kv[1][2])[0]
    return (best[1][0][0], 
            np.sqrt(best[1][1][0][0]),
            best[1][0][1],
            np.sqrt(best[1][1][1][1]),
            best[0], 
            best[1][3],
            best[1][2]) # THE CHI SQUARED


def select_best_fit2_fixedk(parameters, k_err = 0.):
    best = sorted(parameters.items(), key=lambda kv: kv[1][2])[0]
    return (best[1][4], # k parameter
            k_err, # k error
            best[1][0][0], # B parameter
            np.sqrt(best[1][1][0][0]), # B error
            best[0], # a parameter
            best[1][3], # a error
            best[1][2]) # THE CHI SQUARED


def pass_params_fit2(x, params):
    return FIT2_linearized(x, params[0], params[2], params[4])


def pass_params_fit2_min(x, params):
    return FIT2_linearized_err_min(x, params[0], params[1],
                                   params[2], params[3],
                                   params[4], params[5])


def pass_params_fit2_max(x, params):
    return FIT2_linearized_err_max(x, params[0], params[1],
                                   params[2], params[3],
                                   params[4], params[5])

################################################################################
################################################################################
##  PLOTTING FUNCTIONS  ########################################################
################################################################################
################################################################################

# This is just stuff for plotting stuff, nothing special, very boring.

def plot_fit_basic1(fit_params, N, epsilon, angle, n_turns, dynamic_aperture,
                    imgpath="img/fit/fit1"):
    plt.errorbar(
        n_turns, [dynamic_aperture[epsilon][N][angle][0][i] for i in n_turns],
        yerr=[dynamic_aperture[epsilon][N][angle][1][i] for i in n_turns],
        linewidth=0,
        elinewidth=2,
        label='Data')
    plt.plot(
        n_turns,
        pass_params_fit1(n_turns, fit_params),
        'g--',
        linewidth=0.5,
        label='fit: $D_\infty={:6.3f}, b={:6.3f}, k={:6.3f}$'.format(
            fit_params[0], fit_params[2], fit_params[4]))
    # plt.plot(
    #     n_turns,
    #     [pass_params_fit1_min(x, fit_params) for x in n_turns],
    #     'g--',
    #     linewidth=0.5)
    # plt.plot(
    #     n_turns,
    #     [pass_params_fit1_max(x, fit_params) for x in n_turns],
    #     'g--',
    #     linewidth=0.5)
    plt.axhline(
        y=fit_params[0],
        color='r',
        linestyle='-',
        label='$y=D_\infty={:6.3f}$'.format(fit_params[0]))
    plt.xlabel("$N$ turns")
    #plt.xscale("log")
    plt.ylabel("$D (A.U.)$")
    plt.ylim(0., 1.)
    plt.title(
        "FIT1,\n$dx = {:2.2f}, dth = {:3.3f}, mid\,angle = {:3.3f}$,\n$N Parts = {}, \epsilon = {:2.0f}, \omega_x = {:3.3f}, \omega_y = {:3.3f}$".
        format(dx, dtheta, angle, N, epsilon[2], epsilon[0], epsilon[1]))
    # Tweak for legend.
    plt.plot(
        [], [],
        '',
        linewidth=0,
        label="$D_\infty = $" + error_formatter(fit_params[0], fit_params[1]))
    plt.plot(
        [], [],
        '',
        linewidth=0,
        label="$b = $" + error_formatter(fit_params[2], fit_params[3]))
    plt.plot(
        [], [],
        '',
        linewidth=0,
        label="$k = $" + error_formatter(fit_params[4], fit_params[5]))
    # And then the legend.
    plt.legend(prop={"size": 9})
    plt.tight_layout()
    plt.savefig(
        imgpath + "_eps{:2.0f}_wx{:3.3f}_wy{:3.3f}_angle{:3.3f}_Npart{}.png".
        format(epsilon[2], epsilon[0], epsilon[1], angle, N),
        dpi=DPI)
    plt.clf()


def plot_fit_basic2(fit_params, N, epsilon, angle, n_turns, dynamic_aperture,
                    imgpath="img/fit/fit2"):
    plt.errorbar(
        n_turns, [dynamic_aperture[epsilon][N][angle][0][i] for i in n_turns],
        yerr=[dynamic_aperture[epsilon][N][angle][1][i] for i in n_turns],
        linewidth=0,
        elinewidth=2,
        label='Data')
    plt.plot(
        n_turns,
        pass_params_fit2(n_turns, fit_params),
        'g--',
        linewidth=0.5,
        label='fit: $a={:6.3f}, b={:6.3f}, k={:6.3f}$'.format(
            fit_params[4], np.exp(fit_params[2]), fit_params[0]))
    # plt.plot(
    #     n_turns,
    #     [pass_params_fit2_min(x, fit_params) for x in n_turns],
    #     'g--',
    #     linewidth=0.5)
    # plt.plot(
    #     n_turns,
    #     [pass_params_fit2_max(x, fit_params) for x in n_turns],
    #     'g--',
    #     linewidth=0.5)
    plt.xlabel("$N$ turns")
    #plt.xscale("log")
    plt.ylabel("$D (A.U.)$")
    #plt.ylim(0., 1.)
    plt.title(
        "FIT2,\n$dx = {:2.2f}, dth = {:3.3f}, mid\,angle = {:3.3f}$,\n$N Parts = {}, \epsilon = {:2.0f}, \omega_x = {:3.3f}, \omega_y = {:3.3f}$".
        format(dx, dtheta, angle, N, epsilon[2], epsilon[0], epsilon[1]))
    # Tweak for legend.
    plt.plot(
        [], [],
        '',
        linewidth=0,
        label="$a = $" + error_formatter(fit_params[4], fit_params[5]))
    plt.plot(
        [], [],
        '',
        linewidth=0,
        label="$b = $" + error_formatter(
            np.exp(fit_params[2]),
            np.exp(fit_params[2]) * fit_params[3]))
    plt.plot(
        [], [],
        '',
        linewidth=0,
        label="$k = $" + error_formatter(fit_params[0], fit_params[1]))
    # And then the legend.
    plt.legend(prop={"size": 9})
    plt.tight_layout()
    plt.savefig(
        imgpath + "_eps{:2.0f}_wx{:3.3f}_wy{:3.3f}_angle{:3.3f}_Npart{}.png".
        format(epsilon[2], epsilon[0], epsilon[1], angle, N),
        dpi=DPI)
    plt.clf()


def fit_parameters_evolution1(fit_parameters, label="plot"):
    theta = []
    D = []
    D_err = []
    B = []
    B_err = []
    k = []
    k_err = []
    for N in fit_parameters:
        theta_temp = []
        D_temp = []
        B_temp = []
        k_temp = []
        D_temp_err = []
        B_temp_err = []
        k_temp_err = []
        for angle in fit_parameters[N]:
            theta_temp.append(angle / np.pi)
            D_temp.append(fit_parameters[N][angle][0])
            B_temp.append(fit_parameters[N][angle][2])
            k_temp.append(fit_parameters[N][angle][4])
            D_temp_err.append(fit_parameters[N][angle][1])
            B_temp_err.append(fit_parameters[N][angle][3])
            k_temp_err.append(fit_parameters[N][angle][5])
        theta.append(theta_temp)
        D.append(D_temp)
        B.append(B_temp)
        k.append(k_temp)
        D_err.append(D_temp_err)
        B_err.append(B_temp_err)
        k_err.append(k_temp_err)
    # print(A)
    # print(B)
    for i in range(len(D)):
        plt.errorbar(
            theta[i],
            D[i],
            yerr=D_err[i],
            xerr=(0.25 / len(D[i])),
            linewidth=0,
            elinewidth=1)
        plt.xlabel("Theta $(rad / \pi)$")
        plt.ylabel("Fit value " + "$D_\infty$ " + " (A.U.)")
        plt.title("fit1, " + label + ", " + "$D_\infty$ " + "parameter")
        plt.axhline(y=0, color='r', linestyle='-', linewidth=0.5)
        plt.tight_layout()
    plt.savefig("img/fit/" + "fit1" + label + "_Dinf.png", dpi=DPI)
    plt.clf()
    for i in range(len(B)):
        plt.errorbar(
            theta[i],
            B[i],
            yerr=B_err[i],
            xerr=(0.25 / len(B[i])),
            linewidth=0,
            elinewidth=1)
        plt.xlabel("Theta $(rad / \pi)$")
        plt.ylabel("Fit value B (A.U.)")
        plt.title("fit1, " + label + ", B parameter")
        plt.axhline(y=0, color='r', linestyle='-', linewidth=0.5)
        plt.tight_layout()
    plt.savefig("img/fit/" + "fit1" + label + "_B.png", dpi=DPI)
    plt.clf()
    for i in range(len(k)):
        plt.errorbar(
            theta[i],
            k[i],
            yerr=k_err[i],
            xerr=(0.25 / len(k[i])),
            linewidth=0,
            elinewidth=1)
        plt.xlabel("Theta $(rad / \pi)$")
        plt.ylabel("Fit value k (A.U.)")
        plt.title("fit1, " + label + ", k parameter")
        plt.axhline(y=0, color='r', linestyle='-', linewidth=0.5)
        plt.tight_layout()
    plt.savefig("img/fit/" + "fit1" + label + "_k.png", dpi=DPI)
    plt.clf()


def fit_parameters_evolution2(fit_parameters, label="plot"):
    theta = []
    a = []
    a_err = []
    B = []
    B_err = []
    k = []
    k_err = []
    for N in fit_parameters:
        theta_temp = []
        a_temp = []
        B_temp = []
        k_temp = []
        a_temp_err = []
        B_temp_err = []
        k_temp_err = []
        for angle in fit_parameters[N]:
            theta_temp.append(angle / np.pi)
            a_temp.append(fit_parameters[N][angle][4])
            B_temp.append(np.exp(fit_parameters[N][angle][2]))
            k_temp.append(fit_parameters[N][angle][0])
            a_temp_err.append(fit_parameters[N][angle][5])
            B_temp_err.append(np.exp(fit_parameters[N][angle][2]) * 
                              fit_parameters[N][angle][3])
            k_temp_err.append(fit_parameters[N][angle][1])
        theta.append(theta_temp)
        a.append(a_temp)
        B.append(B_temp)
        k.append(k_temp)
        a_err.append(a_temp_err)
        B_err.append(B_temp_err)
        k_err.append(k_temp_err)
    # print(a)
    # print(B)
    for i in range(len(a)):
        plt.errorbar(
            theta[i],
            a[i],
            yerr=a_err[i],
            xerr=(0.25 / len(a[i])),
            linewidth=0,
            elinewidth=1)
        plt.xlabel("Theta $(rad / \pi)$")
        plt.ylabel("Fit value " + "a " + " (A.U.)")
        plt.yscale("log")
        plt.title("fit2, " + label + ", " + "a " + "parameter")
        plt.axhline(y=0, color='r', linestyle='-', linewidth=0.5)
        plt.tight_layout()
    plt.savefig("img/fit/" + "fit2" + label + "_a.png", dpi=DPI)
    plt.clf()
    for i in range(len(B)):
        plt.errorbar(
            theta[i],
            B[i],
            yerr=B_err[i],
            xerr=(0.25 / len(B[i])),
            linewidth=0,
            elinewidth=1)
        plt.xlabel("Theta $(rad / \pi)$")
        plt.ylabel("Fit value b (A.U.)")
        plt.title("fit2, " + label + ", b parameter")
        plt.axhline(y=0, color='r', linestyle='-', linewidth=0.5)
        plt.tight_layout()
    plt.savefig("img/fit/" + "fit2" + label + "_B.png", dpi=DPI)
    plt.clf()
    for i in range(len(k)):
        plt.errorbar(
            theta[i],
            k[i],
            yerr=k_err[i],
            xerr=(0.25 / len(k[i])),
            linewidth=0,
            elinewidth=1)
        plt.xlabel("Theta $(rad / \pi)$")
        plt.ylabel("Fit value k (A.U.)")
        plt.title("fit2, " + label + ", k parameter")
        plt.axhline(y=0, color='r', linestyle='-', linewidth=0.5)
        plt.tight_layout()
    plt.savefig("img/fit/" + "fit2" + label + "_k.png", dpi=DPI)
    plt.clf()


def plot_chi_squared1(fit_params, epsilon, n_partitions=1, angle=np.pi / 4,
                      filename="img/fit/fit1_chisquared"):
    plt.plot(
        list(fit_params.keys()), 
        [x[2] for x in list(fit_params.values())],
        marker="o",
        markersize=0.5,
        linewidth=0.5)
    plt.xlabel("k value")
    plt.ylabel("Chi-Squared value")
    plt.title(
        "non linear FIT1 Chi-Squared evolution, $\epsilon = {:2.0f}$,\n number of partitions $= {}$, central angle $= {:2.2f}$".
        format(epsilon, n_partitions, angle))
    plt.tight_layout()
    plt.savefig(
        filename + "_eps{:2.0f}_npart{}_central{:2.2f}.png".
        format(epsilon, n_partitions, angle),
        dpi=DPI)
    plt.clf()


def plot_chi_squared2(fit_params, epsilon, n_partitions=1, angle=np.pi / 4,
                      filename="img/fit/fit2_chisquared"):
    plt.plot(
        list(fit_params.keys()),
        [x[2] for x in list(fit_params.values())],
        marker="o",
        markersize=0.5,
        linewidth=0.5)
    plt.xlabel("a value")
    #plt.xscale("log")
    plt.ylabel("Chi-Squared value")
    plt.yscale("log")
    plt.grid(True)
    plt.title(
        "non linear FIT2 Chi-Squared evolution, $\epsilon = {:2.0f}$,\n number of partitions $= {}$, central angle $= {:2.2f}$".
        format(epsilon, n_partitions, angle))
    plt.tight_layout()
    plt.savefig(
        filename + "_eps{:2.0f}_npart{}_central{:2.2f}.png".
        format(epsilon, n_partitions, angle),
        dpi=DPI)
    plt.clf()


def plot_chi_squared2_multiple(
                    fit_params_basic, fit_params, k_values, eps, n_part, angle,
                    filename="img/fit/fit2_fixedk_comparison_chisquared"):
    plt.plot(
        list(fit_params_basic.keys()),
        [x[2] for x in list(fit_params_basic.values())],
        marker="x",
        markersize=0.5,
        linewidth=0.5,
        label="Free k")
    for k in k_values:
        plt.plot(
            list(fit_params[k][eps][n_part][angle].keys()),
            [x[2] for x in list(fit_params[k][eps][n_part][angle].values())],
            marker="o",
            markersize=0.5,
            linewidth=0.5,
            label="$k = {:.2f}$".format(k))
    plt.xlabel("a value")
    plt.ylabel("Chi-Squared value")
    plt.yscale("log")
    plt.xscale("log")
    plt.grid(True)
    plt.legend(prop={"size":7}, ncol=2)
    plt.title(
        "FIT2 Chi-Squared evolution for different fixed k, $\epsilon = {:2.0f}$,\n number of partitions $= {}$, central angle $= {:2.2f}$".
        format(eps[2], n_part, angle))
    plt.tight_layout()
    plt.savefig(
        filename + "_eps{:2.0f}_npart{}_central{:2.2f}.png".
        format(eps[2], n_part, angle),
        dpi=DPI)
    plt.clf()


def fit_params_over_epsilon1(fit_params_dict, n_partitions=1, angle=np.pi / 4):
    ## D_inf
    plt.errorbar(
        [x[2] for x in sorted(fit_params_dict)],
        [fit_params_dict[x][n_partitions][angle][0] 
            for x in sorted(fit_params_dict)],
        yerr=[fit_params_dict[x][n_partitions][angle][1] 
            for x in sorted(fit_params_dict)],
        linewidth=0.5,
        elinewidth=0.5,
        marker="x",
        markersize=1)
    plt.xlabel("$\epsilon$")
    plt.ylabel("$D_\infty$ value")
    plt.title("FIT1 $D_\infty$ parameter evolution over $\epsilon$\n"+
              "N partitions $= {}$, central angle $= {:.3f}$".
              format(n_partitions, angle))
    plt.savefig("img/fit/f1param_eps_D_N{}_ang{:2.2f}.png".
                format(n_partitions, angle), dpi=DPI)
    plt.clf()
    ## b
    plt.errorbar(
        [x[2] for x in sorted(fit_params_dict)],
        [fit_params_dict[x][n_partitions][angle][2] 
            for x in sorted(fit_params_dict)],
        yerr=[fit_params_dict[x][n_partitions][angle][3] 
            for x in sorted(fit_params_dict)],
        linewidth=0.5,
        elinewidth=0.5,
        marker="x",
        markersize=1)
    plt.xlabel("$\epsilon$")
    plt.ylabel("$b$ value")
    plt.title("FIT1 $b$ parameter evolution over $\epsilon$\n"+
              "N partitions $= {}$, central angle $= {:.3f}$".
              format(n_partitions, angle))
    plt.savefig("img/fit/f1param_eps_b_N{}_ang{:2.2f}.png".
                format(n_partitions, angle), dpi=DPI)
    plt.clf()
    ## k
    plt.errorbar(
        [x[2] for x in sorted(fit_params_dict)],
        [fit_params_dict[x][n_partitions][angle][4] 
            for x in sorted(fit_params_dict)],
        yerr=[fit_params_dict[x][n_partitions][angle][5] 
            for x in sorted(fit_params_dict)],
        linewidth=0.5,
        elinewidth=0.5,
        marker="x",
        markersize=1)
    plt.xlabel("$\epsilon$")
    plt.ylabel("$k$ value")
    plt.title("FIT1 $k$ parameter evolution over $\epsilon$\n"+
              "N partitions $= {}$, central angle $= {:.3f}$".
              format(n_partitions, angle))
    plt.savefig("img/fit/f1param_eps_k_N{}_ang{:2.2f}.png".
                format(n_partitions, angle), dpi=DPI)
    plt.clf()


def fit_params_over_epsilon2(fit_params_dict, n_partitions=1, angle=np.pi / 4,
                             imgname="img/fit/f2param_eps", titlekind=""):
    ## k
    plt.errorbar(
        [x[2] for x in sorted(fit_params_dict)],
        [fit_params_dict[x][n_partitions][angle][0] 
            for x in sorted(fit_params_dict)],
        yerr=[fit_params_dict[x][n_partitions][angle][1] 
            for x in sorted(fit_params_dict)],
        linewidth=0.5,
        elinewidth=0.5,
        marker="x",
        markersize=1)
    plt.xlabel("$\epsilon$")
    plt.ylabel("$k$ value")
    plt.grid(True)
    plt.title("FIT2 $k$ parameter evolution over $\epsilon$\n"+
              "N partitions $= {}$, central angle $= {:.3f}$, ".
              format(n_partitions, angle) + titlekind) 
    plt.savefig(imgname + "_k_N{}_ang{:2.2f}.png".
                format(n_partitions, angle), dpi=DPI)
    plt.clf()
    ## B
    plt.errorbar(
        [x[2] for x in sorted(fit_params_dict)],
        [fit_params_dict[x][n_partitions][angle][2] 
            for x in sorted(fit_params_dict)],
        yerr=[fit_params_dict[x][n_partitions][angle][3] 
            for x in sorted(fit_params_dict)],
        linewidth=0.5,
        elinewidth=0.5,
        marker="x",
        markersize=1)
    plt.xlabel("$\epsilon$")
    plt.ylabel("$B$ value")
    plt.grid(True)
    plt.title("FIT2 $B$ parameter evolution over $\epsilon$\n"+
              "N partitions $= {}$, central angle $= {:.3f}$, ".
              format(n_partitions, angle) + titlekind)
    plt.savefig(imgname + "_B_N{}_ang{:2.2f}.png".
                format(n_partitions, angle), dpi=DPI)
    plt.clf()
    ## a
    plt.errorbar(
        [x[2] for x in sorted(fit_params_dict)],
        [fit_params_dict[x][n_partitions][angle][4] 
            for x in sorted(fit_params_dict)],
        yerr=[fit_params_dict[x][n_partitions][angle][5] 
            for x in sorted(fit_params_dict)],
        linewidth=0.5,
        elinewidth=0.5,
        marker="x",
        markersize=1)
    plt.xlabel("$\epsilon$")
    plt.yscale("log")
    plt.ylabel("$a$ value")
    plt.ylim(top=1.2e7)
    plt.grid(True)
    plt.title("FIT2 $a$ parameter evolution over $\epsilon$\n"+
              "N partitions $= {}$, central angle $= {:.3f}$, ".
              format(n_partitions, angle) +  titlekind)
    plt.savefig(imgname + "_a_N{}_ang{:2.2f}.png".
                format(n_partitions, angle), dpi=DPI)
    plt.clf()


def plot_B_over_k(fit_params_dict, n_partitions=1, angle=np.pi / 4,
                  imgname="img/fit/f2param_B_k", titlekind=""):
    # B
    x = np.asarray([fit_params_dict[i][n_partitions][angle][2] 
                    for i in sorted(fit_params_dict)])
    # k
    y = np.asarray([fit_params_dict[i][n_partitions][angle][0] 
                    for i in sorted(fit_params_dict)])
    plt.errorbar(
        x, y,
        xerr=[fit_params_dict[x][n_partitions][angle][3] 
            for x in sorted(fit_params_dict)],
        yerr=[fit_params_dict[x][n_partitions][angle][1] 
            for x in sorted(fit_params_dict)],
        linewidth=0.0,
        elinewidth=1.,
        marker="x",
        markersize=1,
        label="Data")
    par, cov = np.polyfit(x, y, deg=1, cov=True)
    # par_plus = np.array([par[0] + np.sqrt(cov[0][0]),
    #                      par[1] + np.sqrt(cov[1][1])])
    # par_minus = np.array([par[0] - np.sqrt(cov[0][0]),
    #                       par[1] - np.sqrt(cov[1][1])]) 
    p = np.poly1d(par)
    # p_plus = np.poly1d(par_plus)
    # p_minus = np.poly1d(par_minus)
    plt.plot(x, p(x), "g--", linewidth=0.5,
             label="p[0]$={:.3}\pm{:.1}$, p[1]$={:.3}\pm{:.1}$".
             format(par[0], np.sqrt(cov[0][0]), par[1], np.sqrt(cov[1][1])))
    # plt.plot(x, p_plus(x), "g--", linewidth=0.5)
    # plt.plot(x, p_minus(x), "g--", linewidth=0.5)
    plt.xlabel("B parameter")
    plt.ylabel("k parameter")
    plt.grid(True)
    plt.title("FIT2 $k$ parameter over $B$ parameter\n"+
              "N partitions $= {}$, central angle $= {:.3f}$, ".
              format(n_partitions, angle) +  titlekind)
    plt.legend()
    plt.tight_layout()
    plt.savefig(imgname + "_N{}_ang{:2.2f}.png".
                format(n_partitions, angle), dpi=DPI)
    plt.clf()


################################################################################
################################################################################
################################################################################
###  LOSS COMPUTATION FUNCTIONS  ###############################################
################################################################################
################################################################################
################################################################################

# Here stuff becomes more heavy...

# Sigmas for gaussian distribution to explore
sigmas = [0.2, 0.25, 0.5, 0.75, 1]

# Functions

def intensity_zero_gaussian(x, y, sigma_x, sigma_y):
    """
    I_0 distribution as a 2D gaussian
    """
    return (1 / (2 * np.pi * sigma_x * sigma_y)) * np.exp(-(
        (x**2 / (2 * sigma_x**2)) + (y**2 / (2 * sigma_y**2))))


def relative_intensity_D_law(D, sigma=1):  # Assuming equal sigma for 2Ds
    return 1 - np.exp(-D**2 / (2 * sigma * sigma))


def D_from_loss(loss, sigma=1):  # INVERSE FORMULA
    return np.sqrt(-(np.log(-(loss - 1))) * (2 * sigma * sigma))


def loss_from_anglescan(contour, time, sigma=1):
    # Inelegant, but works
    dtheta = sorted(contour)[1] - sorted(contour)[0]
    return (2 / np.pi) * integrate.trapz(
        [
            relative_intensity_D_law(contour[sorted(contour)[i]][time], sigma)
            for i in range(len(sorted(contour)))
        ],
        dx=dtheta)


def radscan_intensity(grid, dx=dx):
    # int_theta(int_radius(element * x * dx) * dtheta)
    processed = {}
    for angle in grid:
        temp = [dx * i * grid[angle][i] for i in range(len(grid[angle]))]
        processed[angle] = temp
    lines = {}
    for angle in grid:
        #print(processed[angle])
        lines[angle] = integrate.simps(processed[angle], dx=dx)
    return integrate.simps([lines[angle] for angle in list(sorted(processed))],
            x=[angle for angle in list(sorted(lines))])


def single_partition_intensity(best_fit_params, pass_par_func, time, sigma):
    current_dynamic_aperture = pass_par_func(time, best_fit_params)
    #print(best_fit_params)
    #print(current_dynamic_aperture)
    return relative_intensity_D_law(current_dynamic_aperture, sigma)


def multiple_partition_intensity(best_fit_params, fit_func, n_parts, time,
                                 sigma):
    # Let's treat it as a basic summatory
    intensity = 0.
    for angle in best_fit_params:
        current_dynamic_aperture = fit_func(time, best_fit_params[angle])
        intensity += relative_intensity_D_law(current_dynamic_aperture,
                                              sigma) / n_parts
    return intensity


def error_loss_estimation(best_fit_params, fit_func, contour_data, n_parts,
                          time, sigma):
    error = 0.
    for angle in best_fit_params:
        current_dynamic_aperture = fit_func(time, best_fit_params[angle])
        error_list = []
        angle_list = []
        for theta in contour_data:
            if angle - (np.pi / (n_parts * 2)) <= theta <= angle + (np.pi / 
                                                                (n_parts * 2)):
                error_list.append(np.absolute(current_dynamic_aperture -
                                                    contour_data[theta][time]))
                angle_list.append(theta)
        error_list = np.asarray(error_list)
        angle_list = np.asarray(angle_list)
        error += ((2 / np.pi) * 
                 np.exp(-(current_dynamic_aperture**2) / (2 * sigma**2)) * 
                 current_dynamic_aperture * 
                 integrate.trapz(error_list, x=angle_list))
    return error


def error_loss_estimation_single_partition(best_fit_params, fit_func,
                                           contour_data, time, sigma):
    current_dynamic_aperture = fit_func(time, best_fit_params)
    error_list = []
    angle_list = []
    for theta in contour_data:
        error_list.append(np.absolute(current_dynamic_aperture -
                                            contour_data[theta][time]))
        angle_list.append(theta)
    error_list = np.asarray(error_list)
    angle_list = np.asarray(angle_list)
    return ((2 / np.pi) * 
             np.exp(-(current_dynamic_aperture**2) / (2 * sigma**2)) * 
             current_dynamic_aperture * 
             integrate.trapz(error_list, x=angle_list))

################################################################################
################################################################################
################################################################################
###  LOSS PLOTTING FUNCTIONS  ##################################################
################################################################################
################################################################################
################################################################################

def plot_fit_loss1(fit_params, sigma, epsilon, n_turns, dynamic_aperture,
                   loss_kind, imgpath="img/loss/fit1"):
    plt.errorbar(
        n_turns, dynamic_aperture[sigma][epsilon],
        yerr=dynamic_aperture[sigma][epsilon] * 0.01,
        linewidth=0,
        elinewidth=2,
        label='Data ' + loss_kind)
    plt.plot(
        n_turns,
        pass_params_fit1(n_turns, fit_params),
        'g--',
        linewidth=0.5,
        label='fit: $D_\infty={:6.3f}, B={:6.3f}, k={:6.3f}$'.format(
            fit_params[0], fit_params[2], fit_params[4]))
    plt.plot(
        n_turns,
        [pass_params_fit1_min(x, fit_params) for x in n_turns],
        'g--',
        linewidth=0.5)
    plt.plot(
        n_turns,
        [pass_params_fit1_max(x, fit_params) for x in n_turns],
        'g--',
        linewidth=0.5)
    plt.axhline(
        y=fit_params[0],
        color='r',
        linestyle='-',
        label='$y=D_\infty={:6.3f}$'.format(fit_params[0]))
    plt.xlabel("$N$ turns")
    plt.xscale("log")
    plt.ylabel("$D (A.U.)$")
    plt.ylim(0., 1.)
    plt.title(
        "FIT1, " + loss_kind + "\n$\sigma = {:.2f}, \epsilon = {:.2f}$".
        format(sigma, epsilon[2]))
    # Tweak for legend.
    plt.plot(
        [], [],
        '',
        linewidth=0,
        label="$D_\infty = {:.2} \pm {:.2}$".format(fit_params[0],
                                                    fit_params[1]))
    plt.plot(
        [], [],
        '',
        linewidth=0,
        label="$B = {:.2} \pm {:.2}$".format(fit_params[2], fit_params[3]))
    plt.plot(
        [], [],
        '',
        linewidth=0,
        label="$k = {:.2} \pm {:.2}$".format(fit_params[4], fit_params[5]))
    # And then the legend.
    plt.legend(prop={"size": 7})
    plt.tight_layout()
    plt.savefig(
        imgpath + "sig_{:2.2f}_eps{:2.0f}.png".
        format(sigma, epsilon[2]),
        dpi=DPI)
    plt.clf()


def plot_fit_loss2(fit_params, sigma, epsilon, n_turns, dynamic_aperture,
                    loss_kind, imgpath="img/fit/fit2"):
    plt.errorbar(
        n_turns, dynamic_aperture[sigma][epsilon],
        yerr=dynamic_aperture[sigma][epsilon] * 0.01,
        linewidth=0,
        elinewidth=2,
        label='Data')
    plt.plot(
        n_turns,
        pass_params_fit2(n_turns, fit_params),
        'g--',
        linewidth=0.5,
        label='fit: $a={:.2}, b={:.2}, k={:.2}$'.format(
            fit_params[4], np.exp(fit_params[2]), fit_params[0]))
    plt.plot(
        n_turns,
        [pass_params_fit2_min(x, fit_params) for x in n_turns],
        'g--',
        linewidth=0.5)
    plt.plot(
        n_turns,
        [pass_params_fit2_max(x, fit_params) for x in n_turns],
        'g--',
        linewidth=0.5)
    plt.xlabel("$N$ turns")
    plt.xscale("log")
    plt.ylabel("$D (A.U.)$")
    #plt.ylim(0., 1.)
    plt.title(
        "FIT2, " + loss_kind + "\n$\sigma = {:.2f}, \epsilon = {:.2f}$".
        format(sigma, epsilon[2]))
    # Tweak for legend.
    plt.plot(
        [], [],
        '',
        linewidth=0,
        label="$a = {:.2} \pm {:.2}$".format(fit_params[4], fit_params[5]))
    plt.plot(
        [], [],
        '',
        linewidth=0,
        label="$b = {:.2} \pm {:.2}$".format(
            np.exp(fit_params[2]),
            np.exp(fit_params[2]) * fit_params[3]))
    plt.plot(
        [], [],
        '',
        linewidth=0,
        label="$k = {:.2} \pm {:.2}$".format(fit_params[0], fit_params[1]))
    # And then the legend.
    plt.legend(prop={"size": 7})
    plt.tight_layout()
    plt.savefig(
        imgpath + "sig_{:2.2f}_eps{:2.0f}.png".
        format(sigma, epsilon[2]),
        dpi=DPI)
    plt.clf()


def plot_losses(title, filename,
                n_turns, data_list=[], data_label_list=[], data_color_list=[],
                param_list=[], param_list_min=[], param_list_max=[],
                param_error_list=[], param_label_list=[], param_color_list=[],
                scan_error=True, fit_error=True):
    for i in range(len(data_list)):
        plt.plot(
            n_turns,
            data_list[i][1:],
            linewidth=0.5,
            label=data_label_list[i],
            color=data_color_list[i])    
    for i in range(len(param_list)):
        if scan_error:
            plt.errorbar(
                n_turns,
                param_list[i][1:],
                yerr=param_error_list[i][1:],
                linewidth=0.5,
                label=param_label_list[i],
                color=param_color_list[i])
        else:
            plt.plot(
                n_turns,
                param_list[i][1:],
                linewidth=0.5,
                label=param_label_list[i],
                color=param_color_list[i])
        if fit_error:
            plt.fill_between(
                n_turns,
                param_list_min[i][1:],
                param_list_max[i][1:],
                interpolate=True, alpha=0.5,
                color=param_color_list[i])
    #plt.title(title)
    plt.xlabel("N turns")
    plt.xscale("log")
    plt.xlim(1e3, 1e7)
    plt.ylabel("Relative Intensity")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(filename, dpi=DPI)  
    plt.clf()


def compute_l2(n_turns, data, param_list):
    error = np.array([(data[i] - param_list[i])**2 for i in range(len(data))])
    return integrate.simps(error, n_turns)


def plot_l2(title, filename, n_turns, data, data_all, param_list):
    l2 = []
    l2_all = []
    for i in range(len(param_list)):
        l2.append(compute_l2(n_turns, data[1:], param_list[i][1:]))
        l2_all.append(compute_l2(n_turns, data_all[1:], param_list[i][1:]))
    plt.plot(
        list(range(1, len(param_list) + 1)),
        l2,
        marker="x",
        linewidth=0.,
        markersize=4.,
        label="$L^2 norm with main region loss$")
    # plt.plot(
    #     list(range(1, len(param_list) + 1)),
    #     l2_all,
    #     marker="x",
    #     linewidth=0.,
    #     markersize=2.,
    #     label="$L^2 norm with all losses$")
    # plt.legend()
    plt.xlabel("Number of partitions")
    plt.ylabel("$L^2$-norm")
    plt.ylim(bottom=0.)
    #plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(filename, dpi=DPI)
    plt.clf()

################################################################################
################################################################################
################################################################################
###  LHC COMPUTATION AND PLOTTING FUNCTIONS   ##################################
################################################################################
################################################################################
################################################################################

from matplotlib.lines import Line2D

def remove_first_times_lhc(data, lower_bound):
    for folder in data:
        for kind in data[folder]:
            for seed in data[folder][kind]:
                for time in list(seed.keys()):
                    if time < lower_bound:
                        del seed[time]
    return data


def sigma_filler(data_dict, perc):
    """
    We need the error estimation on the data for performing the fits,
    this thing allows us to put in there an estimation for the
    error percentage and use the resulting error data in the fit
    """
    sigma_dict = {}
    for element in data_dict:
        sigma_dict[element] = data_dict[element] * perc
    return sigma_dict

# I was using this when I wanted to distinguish particular combination of cases
# now it's basically deactivated, as soon as I can I will work to remove it.
def lambda_color(fit1_selected, fit2_decent):
    if not (fit1_selected ^ fit2_decent):
        return "g--"
    elif (fit1_selected and not fit2_decent):
        return "g--"
    elif (fit2_decent and not fit1_selected):
        return "g--"


def plot_lhc_fit(best_fit, data, func, label):
    j = 0
    for i in range(len(data)):
        plt.plot(
            sorted(data[i]), [data[i][x] for x in sorted(data[i])],
            label="data {}".format(i),
            color="b",
            markersize=0.5,
            marker="x",
            linewidth=0)
        plt.plot(
            sorted(data[i]),
            func(sorted(data[i]), best_fit[i]),
            lambda_color(True, True),
            linewidth=0.5,
            label='fit {}'.format(i))
        j += 1
    plt.xlabel("N turns")
    plt.xscale("log")
    plt.ylabel("D (A.U.)")
    plt.title("All LHC fits, " + label)
    plt.tight_layout()
    plt.savefig("img/lhc/lhc_" + label + "_all.png", dpi=DPI)
    plt.clf()


def best_fit_seed_distrib1(params, label="plot"):
    plt.errorbar(
        list(range(len(params))), [x[0] for x in params],
        yerr=[x[1] for x in params],
        linewidth=0,
        elinewidth=1,
        marker="o",
        markersize=2)
    plt.xlabel("Seed number")
    plt.ylabel("$D_\infty$" + " parameter")
    plt.title(label + ", " + "$D_\infty$" + " parameter")
    plt.axhline(y=0, color='r', linestyle='-', linewidth=0.5)
    plt.tight_layout()
    plt.savefig("img/lhc/lhc_" + label + "_Dinf.png", dpi=DPI)
    plt.clf()

    plt.errorbar(
        list(range(len(params))), [x[2] for x in params],
        yerr=[x[3] for x in params],
        linewidth=0,
        elinewidth=1,
        marker="o",
        markersize=2)
    plt.xlabel("Seed number")
    plt.ylabel("b parameter")
    plt.yscale("symlog")
    plt.title(label + ", b parameter")
    plt.axhline(y=0, color='r', linestyle='-', linewidth=0.5)
    plt.tight_layout()
    plt.savefig("img/lhc/lhc_" + label + "_B.png", dpi=DPI)
    plt.clf()

    plt.errorbar(
        list(range(len(params))), [x[4] for x in params],
        yerr=[x[5] for x in params],
        linewidth=0,
        elinewidth=1,
        marker="o",
        markersize=2)
    plt.xlabel("Seed number")
    plt.ylabel("k parameter")
    plt.title(label + ", k parameter")
    plt.axhline(y=0, color='r', linestyle='-', linewidth=0.5)
    plt.tight_layout()
    plt.savefig("img/lhc/lhc_" + label + "_k.png", dpi=DPI)
    plt.clf()


def best_fit_seed_distrib2(params, label="plot"):
    plt.errorbar(
        list(range(len(params))), [x[4] for x in params],
        yerr=[x[5] for x in params],
        linewidth=0,
        elinewidth=1,
        marker="o",
        markersize=2)
    plt.xlabel("Seed number")
    plt.ylabel("$a$" + " parameter")
    plt.yscale("symlog")
    plt.title(label + ", " + "$a$" + " parameter")
    plt.axhline(y=0, color='r', linestyle='-', linewidth=0.5)
    plt.tight_layout()
    plt.savefig("img/lhc/lhc_" + label + "_a.png", dpi=DPI)
    plt.clf()

    plt.errorbar(
        list(range(len(params))), [np.exp(x[2]) for x in params],
        yerr=[np.exp(x[2]) * x[3] for x in params],
        linewidth=0,
        elinewidth=1,
        marker="o",
        markersize=2)
    plt.xlabel("Seed number")
    plt.ylabel("b parameter")
    plt.title(label + ", b parameter")
    plt.axhline(y=0, color='r', linestyle='-', linewidth=0.5)
    plt.tight_layout()
    plt.savefig("img/lhc/lhc_" + label + "_B.png", dpi=DPI)
    plt.clf()

    plt.errorbar(
        list(range(len(params))), [x[0] for x in params],
        yerr=[x[1] for x in params],
        linewidth=0,
        elinewidth=1,
        marker="o",
        markersize=2)
    plt.xlabel("Seed number")
    plt.ylabel("k parameter")
    plt.title(label + ", k parameter")
    plt.axhline(y=0, color='r', linestyle='-', linewidth=0.5)
    plt.tight_layout()
    plt.savefig("img/lhc/lhc_" + label + "_k.png", dpi=DPI)
    plt.clf()


def lhc_2param_comparison1(params, label="plot"):
    plt.plot(
        [x[0] for x in params], [x[2] for x in params],
        linewidth=0,
        marker="o",
        markersize=2)
    plt.xlabel("$D_\infty$")
    plt.ylabel("$b$")
    plt.title("Correlation plot between $D_\infty$ and $b$")
    plt.tight_layout()
    plt.savefig("img/lhc/lhc_" + label + "_DB.png", dpi=DPI)
    plt.clf()

    plt.plot(
        [x[0] for x in params], [x[4] for x in params],
        linewidth=0,
        marker="o",
        markersize=2)
    plt.xlabel("$D_\infty$")
    plt.ylabel("$k$")
    plt.title("Correlation plot between $D_\infty$ and $k$")
    plt.tight_layout()
    plt.savefig("img/lhc/lhc_" + label + "_Dk.png", dpi=DPI)
    plt.clf()

    plt.plot(
        [x[2] for x in params], [x[4] for x in params],
        linewidth=0,
        marker="o",
        markersize=2)
    plt.xlabel("$b$")
    plt.ylabel("$k$")
    plt.title("Correlation plot between $b$ and $k$")
    plt.tight_layout()
    plt.savefig("img/lhc/lhc_" + label + "_Bk.png", dpi=DPI)
    plt.clf()


def lhc_2param_comparison2(params, label="plot"):
    plt.plot(
        [x[4] for x in params], [np.exp(x[2]) for x in params],
        linewidth=0,
        marker="o",
        markersize=2)
    plt.xlabel("$a$")
    plt.xscale("log")
    plt.xlim(left=0.00001)
    plt.ylabel("$b$")
    plt.title("Correlation plot between $a$ and $b$")
    plt.tight_layout()
    plt.savefig("img/lhc/lhc_" + label + "_aB.png", dpi=DPI)
    plt.clf()

    plt.plot(
        [x[4] for x in params], [x[0] for x in params],
        linewidth=0,
        marker="o",
        markersize=2)
    plt.xlabel("$a$")
    plt.xscale("log")
    plt.xlim(left=0.00001)
    plt.ylabel("$k$")
    plt.title("Correlation plot between $a$ and $k$")
    plt.tight_layout()
    plt.savefig("img/lhc/lhc_" + label + "_ak.png", dpi=DPI)
    plt.clf()

    plt.plot(
        [np.exp(x[2]) for x in params], [x[0] for x in params],
        linewidth=0,
        marker="o",
        markersize=2)
    plt.xlabel("$b$")
    plt.ylabel("$k$")
    plt.title("Correlation plot between $b$ and $k$")
    plt.tight_layout()
    plt.savefig("img/lhc/lhc_" + label + "_Bk.png", dpi=DPI)
    plt.clf()


def lhc_plot_chi_squared1(data, folder, kind, fit1_p, fit2_b):
    j = 0
    for seed in data:
        plt.plot(sorted(seed), 
                 [seed[x][2] for x in sorted(seed)],
                 lambda_color(fit1_p[j], fit2_b[j]),
                 linewidth=0.3,
                 marker='o',
                 markersize=0.0)
        j += 1
    plt.xlabel("k value")
    plt.ylabel("Chi-Squared value")
    plt.title("Behaviour of Chi-Squared function in non linear fit part")
    plt.tight_layout()
    plt.savefig("img/lhc/lhc_" + folder + kind + "f1" + "_chisquared.png",
                dpi=DPI)
    plt.clf()

def lhc_plot_chi_squared2(data, folder, kind, fit1_p, fit2_b):
    j = 0
    for seed in data:
        plt.plot(sorted(seed), 
                 [seed[x][2] for x in sorted(seed)],
                 lambda_color(fit1_p[j], fit2_b[j]),
                 linewidth=0.3,
                 marker='o',
                 markersize=0.0)
        j += 1
    plt.xlabel("a value")
    plt.xscale("log")
    plt.ylabel("Chi-Squared value")
    plt.title("Behaviour of Chi-Squared function in non linear fit part")
    plt.tight_layout()
    plt.savefig("img/lhc/lhc_" + folder + kind + "f2" + "_chisquared.png",
                dpi=DPI)
    plt.clf()


def lhc_plot_chi_squared2_multiple(data, free_data, folder, kind, k_values,
                                   imgname="img/lhc/lhc_multiple"):
    custom_lines = []
    custom_labels = []
    colors = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9']
    
    for seed in free_data[folder][kind]:
        plt.plot(sorted(seed),
                 [seed[x][2] for x in sorted(seed)],
                 color=colors[0],
                 linewidth=0.3,
                 marker='o',
                 markersize=0.0)
    custom_lines.append(Line2D([0], [0], color=colors[0], lw=4))
    custom_labels.append("Free $k$")
    j = 1
    for k in k_values:
        for seed in data[k][folder][kind]:
            plt.plot(sorted(seed),
                     [seed[x][2] for x in sorted(seed)],
                     color=colors[j % len(colors)],
                     linewidth=0.3,
                     marker='o',
                     markersize=0.0)
        custom_lines.append(Line2D([0], [0], color=colors[j % len(colors)], lw=4))
        custom_labels.append("$k = {:.2f}$".format(k))
        j += 1
    plt.xlabel("a value")
    plt.xscale("log")
    plt.ylabel("Chi-Squared value")
    plt.yscale("log")
    plt.title("Behaviour of Chi-Squared function for different $k$ values\n"
              + folder + " " + kind)
    plt.legend(custom_lines, custom_labels)
    plt.tight_layout()
    plt.savefig(imgname + "_" + folder + kind + "f2" + "_chisquared.png",
                dpi=DPI)
    plt.clf()


def combine_plots_lhc1(folder, kind):
    img2 = cv2.imread("img/lhc/lhc_" + folder + kind + "f1" + "_Dinf.png")
    img3 = cv2.imread("img/lhc/lhc_" + folder + kind + "f1" + "_B.png")
    img4 = cv2.imread("img/lhc/lhc_" + folder + kind + "f1" + "_k.png")
    img5 = cv2.imread("img/lhc/lhc_" + folder + kind + "f1" + "_DB.png")
    img6 = cv2.imread("img/lhc/lhc_" + folder + kind + "f1" + "_Dk.png")
    img7 = cv2.imread("img/lhc/lhc_" + folder + kind + "f1" + "_Bk.png")
    row2 = np.concatenate((img2, img3, img4), axis=1)
    row3 = np.concatenate((img5, img6, img7), axis=1)
    image = np.concatenate((row2, row3), axis=0)
    cv2.imwrite("img/lhc/lhc_bigpicture_" + folder + kind + "f1" + ".png",
                image)


def combine_plots_lhc2(folder, kind):
    img2 = cv2.imread("img/lhc/lhc_" + folder + kind + "f2" + "_a.png")
    img3 = cv2.imread("img/lhc/lhc_" + folder + kind + "f2" + "_B.png")
    img4 = cv2.imread("img/lhc/lhc_" + folder + kind + "f2" + "_k.png")
    img5 = cv2.imread("img/lhc/lhc_" + folder + kind + "f2" + "_aB.png")
    img6 = cv2.imread("img/lhc/lhc_" + folder + kind + "f2" + "_ak.png")
    img7 = cv2.imread("img/lhc/lhc_" + folder + kind + "f2" + "_Bk.png")
    row2 = np.concatenate((img2, img3, img4), axis=1)
    row3 = np.concatenate((img5, img6, img7), axis=1)
    image = np.concatenate((row2, row3), axis=0)
    cv2.imwrite("img/lhc/lhc_bigpicture_" + "f2" + folder + kind + ".png", image)


def combine_plots_lhc3(folder, kind):
    img1 = cv2.imread("img/lhc/lhc_" + folder + kind + "f2" + "_all.png")
    img2 = cv2.imread("img/lhc/lhc_" + folder + kind + "f2" + "_a.png")
    img3 = cv2.imread("img/lhc/lhc_" + folder + kind + "f2" + "_B.png")
    img4 = cv2.imread("img/lhc/lhc_" + folder + kind + "f2" + "_k.png")
    img5 = cv2.imread("img/lhc/lhc_" + folder + kind + "f1" + "_Dinf.png")
    img6 = cv2.imread("img/lhc/lhc_" + folder + kind + "f1" + "_B.png")
    img7 = cv2.imread("img/lhc/lhc_" + folder + kind + "f1" + "_k.png")
    img8 = cv2.imread("img/lhc/lhc_" + folder + kind + "f2" + "_chisquared.png")
    img9 = cv2.imread("img/lhc/lhc_" + folder + kind + "f1" + "_chisquared.png")
    filler = np.zeros(img1.shape)
    row1 = np.concatenate((filler, img1, filler), axis=1)
    row2 = np.concatenate((img2, img3, img4), axis=1)
    row3 = np.concatenate((img5, img6, img7), axis=1)
    image = np.concatenate((row1, row2, row3), axis=0)
    cv2.imwrite("img/lhc/lhc_bigpicture_" + "both" + folder + kind + ".png",
                image)


################################################################################
################################################################################
################################################################################
###  GENERAL AND RANDOM FUNCTIONS  #############################################
################################################################################
################################################################################
################################################################################

def plot_fit_nek1(fit_params, label, n_turns, dynamic_aperture,
                  dynamic_aperture_err, imgpath="img/nek/fit1", logscale=False):
    plt.errorbar(
        n_turns, [dynamic_aperture[i] for i in n_turns],
        yerr=[dynamic_aperture_err[i] for i in n_turns],
        linewidth=0,
        elinewidth=2,
        label='Data')
    plt.plot(
        n_turns,
        pass_params_fit1(n_turns, fit_params),
        'g--',
        linewidth=0.5,
        label='fit: $D_\infty={:.2}, B={:.2}, k={:.2}$'.format(
            fit_params[0], fit_params[2], fit_params[4]))
    plt.axhline(
        y=fit_params[0],
        color='r',
        linestyle='-',
        label='$y=D_\infty={:.2}$'.format(fit_params[0]))
    plt.xlabel("$N$ turns")
    if logscale:
        plt.xscale("log")
    plt.ylabel("$D (Sigma Units)$")
    #plt.ylim(0., 1.)
    plt.title(
        "FIT1, label $= {}$".format(label))
    # Tweak for legend.
    plt.plot(
        [], [],
        '',
        linewidth=0,
        label="$D_\infty = {:.2} \pm {:.2}$".format(fit_params[0],
                                                    fit_params[1]))
    plt.plot(
        [], [],
        '',
        linewidth=0,
        label="$B = {:.2} \pm {:.2}$".format(fit_params[2], fit_params[3]))
    plt.plot(
        [], [],
        '',
        linewidth=0,
        label="$k = {:.2} \pm {:.2}$".format(fit_params[4], fit_params[5]))
    # And then the legend.
    plt.legend(prop={"size": 7})
    plt.tight_layout()
    plt.savefig(
        imgpath + "_label{}.png".
        format(label),
        dpi=DPI)
    plt.clf()


def plot_fit_nek2(fit_params, label, n_turns, dynamic_aperture,
                  dynamic_aperture_err, imgpath="img/nek/fit2", logscale=False):
    plt.errorbar(
        n_turns, [dynamic_aperture[i] for i in n_turns],
        yerr=[dynamic_aperture_err[i] for i in n_turns],
        linewidth=0,
        elinewidth=2,
        label='Data')
    plt.plot(
        n_turns,
        pass_params_fit2(n_turns, fit_params),
        'g--',
        linewidth=0.5,
        label='fit: $a={:.2}, b={:.2}, k={:.2}$'.format(
            fit_params[4], np.exp(fit_params[2]), fit_params[0]))
    plt.xlabel("$N$ turns")
    if logscale:
        plt.xscale("log")
    plt.ylabel("$D (Sigma Units)$")
    #plt.ylim(0., 1.)
    plt.title(
        "FIT2, label $ = {}$".format(label))
    # Tweak for legend.
    plt.plot(
        [], [],
        '',
        linewidth=0,
        label="$a = {:.2} \pm {:.2}$".format(fit_params[4], fit_params[5]))
    plt.plot(
        [], [],
        '',
        linewidth=0,
        label="$b = {:.2} \pm {:.2}$".format(
            np.exp(fit_params[2]),
            np.exp(fit_params[2]) * fit_params[3]))
    plt.plot(
        [], [],
        '',
        linewidth=0,
        label="$k = {:.2} \pm {:.2}$".format(fit_params[0], fit_params[1]))
    # And then the legend.
    plt.legend(prop={"size": 7})
    plt.tight_layout()
    plt.savefig(
        imgpath + "_label{}.png".format(label),
        dpi=DPI)
    plt.clf()


def combine_image_3x1(imgname, path1, path2="none", path3="none"):
    img1 = cv2.imread(path1)
    filler = np.zeros(img1.shape)
    img2 = cv2.imread(path2) if path2 is not "none" else filler
    img3 = cv2.imread(path3) if path3 is not "none" else filler
    row1 = np.concatenate((img1, img2, img3), axis=1)
    cv2.imwrite(imgname, row1)


def combine_image_3x2(imgname, path1, path2="none", path3="none", path4="none",
                      path5="none", path6="none"):
    img1 = cv2.imread(path1)
    filler = np.zeros(img1.shape)
    img2 = cv2.imread(path2) if path2 is not "none" else filler
    img3 = cv2.imread(path3) if path3 is not "none" else filler
    img4 = cv2.imread(path4) if path4 is not "none" else filler
    img5 = cv2.imread(path5) if path5 is not "none" else filler
    img6 = cv2.imread(path6) if path6 is not "none" else filler
    row1 = np.concatenate((img1, img2, img3), axis=1)
    row2 = np.concatenate((img4, img5, img6), axis=1)
    image = np.concatenate((row1, row2), axis=0)
    cv2.imwrite(imgname, image)


def combine_image_3x3(imgname, path1, path2="none", path3="none", path4="none",
                      path5="none", path6="none", path7="none", path8="none",
                      path9="none"):
    img1 = cv2.imread(path1)
    filler = np.zeros(img1.shape)
    img2 = cv2.imread(path2) if path2 is not "none" else filler
    img3 = cv2.imread(path3) if path3 is not "none" else filler
    img4 = cv2.imread(path4) if path4 is not "none" else filler
    img5 = cv2.imread(path5) if path5 is not "none" else filler
    img6 = cv2.imread(path6) if path6 is not "none" else filler
    img7 = cv2.imread(path7) if path7 is not "none" else filler
    img8 = cv2.imread(path8) if path8 is not "none" else filler
    img9 = cv2.imread(path9) if path9 is not "none" else filler
    row1 = np.concatenate((img1, img2, img3), axis=1)
    row2 = np.concatenate((img4, img5, img6), axis=1)
    row3 = np.concatenate((img7, img8, img9), axis=1)
    image = np.concatenate((row1, row2, row3), axis=0)
    cv2.imwrite(imgname, image)


def combine_image_6x2(imgname, path1, path2="none", path3="none", path4="none",
                    path5="none", path6="none", path7="none", path8="none",
                    path9="none", path10="none", path11="none", path12="none"):
    img1 = cv2.imread(path1)
    filler = np.zeros(img1.shape)
    img2 = cv2.imread(path2) if path2 is not "none" else filler
    img3 = cv2.imread(path3) if path3 is not "none" else filler
    img4 = cv2.imread(path4) if path4 is not "none" else filler
    img5 = cv2.imread(path5) if path5 is not "none" else filler
    img6 = cv2.imread(path6) if path6 is not "none" else filler
    img7 = cv2.imread(path7) if path7 is not "none" else filler
    img8 = cv2.imread(path8) if path8 is not "none" else filler
    img9 = cv2.imread(path9) if path9 is not "none" else filler
    img10 = cv2.imread(path10) if path9 is not "none" else filler
    img11 = cv2.imread(path11) if path9 is not "none" else filler
    img12 = cv2.imread(path12) if path9 is not "none" else filler
    row1 = np.concatenate((img1, img2, img3, img4, img5, img6), axis=1)
    row2 = np.concatenate((img7, img8, img9, img10, img11, img12), axis=1)
    image = np.concatenate((row1, row2), axis=0)
    cv2.imwrite(imgname, image)