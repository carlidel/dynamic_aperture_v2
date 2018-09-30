import pickle
import numpy as np
import matplotlib.pyplot as plt

from fit_library import *

################################################################################
################################################################################
################################################################################
###  FIRST PART - BASIC FITS ON HENNON MAPS  ###################################
################################################################################
################################################################################
################################################################################

print("load data")

data = pickle.load(open("revised_version_hennon.pkl", "rb"))

# temporary removal of high epsilons for performance:
#i = 0
#for epsilon in sorted(data.keys()):
#    if i > 9:
#        del data[epsilon]
#        del lin_data[epsilon]
#    i += 1
# end temporary

dx = 0.01

#%%
contour_data = {}
for epsilon in data:
    contour_data[epsilon] = make_countour_data(data[epsilon], n_turns, dx)

#%%
dynamic_aperture = {}

for epsilon in sorted(data):
    dyn_temp = {}
    for partition_list in partition_lists:
        dyn_temp[len(partition_list) - 1] = divide_and_compute(
            data[epsilon], n_turns, partition_list)
    dynamic_aperture[epsilon] = dyn_temp

#%%
print("Fit1 Iterated")

# Search parameters
k_min = -20.
k_max = 7.
dk = 0.1
n_iterations = 2

fit_parameters1 = {}
best_fit_parameters1 = {}

for epsilon in dynamic_aperture:
    print(epsilon)
    # fit1
    fit_parameters_epsilon = {}
    best_fit_parameters_epsilon = {}

    for partition_list in partition_lists:
        fit = {}
        best = {}
        for angle in dynamic_aperture[epsilon][len(partition_list) - 1]:
            fit[angle], best[angle] = non_linear_fit1_iterated(
                dynamic_aperture[epsilon][len(partition_list) - 1][angle][0],
                dynamic_aperture[epsilon][len(partition_list) - 1][angle][1],
                n_turns, k_min, k_max, dk, n_iterations)
        best_fit_parameters_epsilon[len(partition_list) - 1] = best
        fit_parameters_epsilon[len(partition_list) - 1] = fit
    best_fit_parameters1[epsilon] = best_fit_parameters_epsilon
    fit_parameters1[epsilon] = fit_parameters_epsilon

#%%
print("Fit2 DOUBLESCAN on k and a!!!")

k_min = 0.05
dk = 0.01

da = 0.0005
a_max = 0.01
a_min = (1 / n_turns[0]) + da ### under this value it doesn't converge
a_bound = n_turns[-1] / a_max
a_default = n_turns[-1]

best_fit_parameters2_doublescan = {}

for epsilon in dynamic_aperture:
    print(epsilon)
    best_fit_parameters_epsilon = {}
    for partition_list in partition_lists:
        best = {}
        for angle in dynamic_aperture[epsilon][len(partition_list) - 1]:
            _, best[angle] = non_linear_fit2_doublescan(
                dynamic_aperture[epsilon][len(partition_list) - 1][angle][0],
                    dynamic_aperture[epsilon][len(partition_list) - 1][angle][1],
                    n_turns,
                    k_min, dk,
                    a_min, a_max, da, a_bound, a_default)
        best_fit_parameters_epsilon[len(partition_list) - 1] = best
    best_fit_parameters2_doublescan[epsilon] = best_fit_parameters_epsilon


#%%
print("Plot fits from simulation 1.")
for epsilon in best_fit_parameters1:
    print(epsilon)
    #for n_angles in best_fit_parameters1[epsilon]:
    for N in best_fit_parameters1[epsilon]:
        for angle in best_fit_parameters1[epsilon][N]:
            plot_fit_basic1(best_fit_parameters1[epsilon][N][angle],
                            N, epsilon, angle, n_turns,
                            dynamic_aperture)

#%%
print("Plot fits from simulation 2 doublescan")
for epsilon in best_fit_parameters2_doublescan:
    for angle in best_fit_parameters2_doublescan[epsilon][1]:
        plot_fit_basic2(
                best_fit_parameters2_doublescan[epsilon][1][angle],
                1, epsilon, angle, n_turns, dynamic_aperture,
                "img/fit/fit2_doublescan_")

#%%
print("Compare chi squared fits1.")
for epsilon in fit_parameters1:
    for N in fit_parameters1[epsilon]:
        for angle in fit_parameters1[epsilon][N]:
            plot_chi_squared1(fit_parameters1[epsilon][N][angle],
                              epsilon[2], N, angle)

###############################################################################
#%%
print("Fit1 param evolution over epsilon.")
temp = list(best_fit_parameters1.keys())[0]
for N in best_fit_parameters1[temp]:
    for angle in (best_fit_parameters1[temp][N]):
        fit_params_over_epsilon1(best_fit_parameters1, N, angle)

#%%
print("Fit2 param evolution over epsilon doublescan")
temp = list(best_fit_parameters2_doublescan.keys())[0]
for N in best_fit_parameters2_doublescan[temp]:
    for angle in (best_fit_parameters2_doublescan[temp][N]):
        fit_params_over_epsilon2(best_fit_parameters2_doublescan, N, angle,
                                 "img/fit/f2param_eps")

#%%
print("compose fit over epsilon.")
for N in best_fit_parameters1[temp]:
    for angle in (best_fit_parameters1[temp][N]):
        combine_image_3x2("img/fit/params_over_epsilon_N{}_ang{:2.2f}.png".
            format(N, angle),
                  "img/fit/f1param_eps_D_N{}_ang{:2.2f}.png".format(N, angle),
                  "img/fit/f1param_eps_b_N{}_ang{:2.2f}.png".format(N, angle),
                  "img/fit/f1param_eps_k_N{}_ang{:2.2f}.png".format(N, angle),
                  "img/fit/f2param_eps_A_N{}_ang{:2.2f}.png".format(N, angle),
                  "img/fit/f2param_eps_B_N{}_ang{:2.2f}.png".format(N, angle),
                  "img/fit/f2param_eps_k_N{}_ang{:2.2f}.png".format(N, angle))

#%%
print("compose fit over epsilon.")
for N in best_fit_parameters2_doublescan[temp]:
    for angle in (best_fit_parameters2_doublescan[temp][N]):
        combine_image_3x2("img/fit/paramsFIT2_over_epsilon_N{}_ang{:2.2f}.png".
            format(N, angle),
          "img/fit/f2param_eps_doublescan_a_N{}_ang{:2.2f}.png".format(N, angle),
          "img/fit/f2param_eps_doublescan_B_N{}_ang{:2.2f}.png".format(N, angle),
          "img/fit/f2param_eps_doublescan_k_N{}_ang{:2.2f}.png".format(N, angle),
          "img/fit/f2param_eps_a_N{}_ang{:2.2f}.png".format(N, angle),
          "img/fit/f2param_eps_B_N{}_ang{:2.2f}.png".format(N, angle),
          "img/fit/f2param_eps_k_N{}_ang{:2.2f}.png".format(N, angle))

#%%
print("Parameters over partitions")
for epsilon in best_fit_parameters1:
    print(epsilon)
    label = "partitioneps{:2.2f}".format(epsilon[2])
    fit_parameters_evolution1(best_fit_parameters1[epsilon],
        label)
    combine_image_3x1("img/fit/partitions1_eps{:2.2f}.png".format(epsilon[2]),
        "img/fit/fit1" + label + "_Dinf.png",
        "img/fit/fit1" + label + "_B.png",
        "img/fit/fit1" + label + "_k.png")

for epsilon in best_fit_parameters2_doublescan:
    print(epsilon)
    label = "partitioneps{:2.2f}".format(epsilon[2])
    fit_parameters_evolution2(best_fit_parameters2_doublescan[epsilon],
        label)
    combine_image_3x1("img/fit/partitions2_eps{:2.2f}.png".format(epsilon[2]),
        "img/fit/fit2" + label + "_a.png",
        "img/fit/fit2" + label + "_B.png",
        "img/fit/fit2" + label + "_k.png")

#%%
print("B and k!")
temp = list(best_fit_parameters2_doublescan.keys())[0]
for N in best_fit_parameters2_doublescan[temp]:
    for angle in (best_fit_parameters2_doublescan[temp][N]):
        plot_B_over_k(best_fit_parameters2_doublescan, N, angle,
                      "img/fit/f2param_B_k_doublescan", "doublescan")

################################################################################
################################################################################
################################################################################
###  SECOND PART - LOSS COMPUTATION AND FITS  ##################################
################################################################################
################################################################################
################################################################################
#%%
print("Is this loss?")

# Sigmas to explore
sigmas = [0.2, 0.25, 0.5, 0.75, 1]

I0 = 0.25
weights = {}

print("Computing weights")
for sigma in sigmas:
    print(sigma)
    weights_sigma = {}
    temp = list(data.keys())[0]
    for angle in data[temp]:
        weights_sigma[angle] = [intensity_zero_gaussian(i * dx * np.cos(angle),
            i * dx * np.sin(angle), sigma, sigma) for i in range(100)]
    weights[sigma] = weights_sigma

#%%
print("loss considering all")
loss_all = {}
for sigma in sigmas:
    print(sigma)
    loss_all_sigma = {}
    for epsilon in data:
        print(epsilon)
        intensity_evolution = [I0]
        for time in n_turns:
            mask = {}
            for angle in data[epsilon]:
                mask[angle] = [x >= time for x in data[epsilon][angle]]
            masked_weights = {}
            for angle in data[epsilon]:
                masked_weights[angle] = [mask[angle][i] * weights[sigma][angle][i]
                                         for i in range(len(mask[angle]))]
            intensity_evolution.append(radscan_intensity(masked_weights))
        loss_all_sigma[epsilon] = np.asarray(intensity_evolution) / I0
    loss_all[sigma] = loss_all_sigma

#%%
print("loss considering only main part")
loss_mainregion = {}
for sigma in sigmas:
    print(sigma)
    loss_mainregion_sigma = {}
    for epsilon in data:
        print(epsilon)
        intensity_evolution = [I0]
        for time in n_turns:
            mask = {}
            for angle in data[epsilon]:
                mask[angle] = [i * dx <= contour_data[epsilon][angle][time]
                                for i in range(len(data[epsilon][angle]))]
            masked_weights = {}
            for angle in data[epsilon]:
                masked_weights[angle] = [mask[angle][i] * weights[sigma][angle][i]
                                         for i in range(len(mask[angle]))]
            intensity_evolution.append(radscan_intensity(masked_weights))
        loss_mainregion_sigma[epsilon] = np.asarray(intensity_evolution) / I0
    loss_mainregion[sigma] = loss_mainregion_sigma

#%%
print("loss considering FIT1")
loss_D_fit1 = {}
loss_D_fit1_min = {}
loss_D_fit1_max = {}
loss_D_fit1_err = {}
for sigma in sigmas:
    print(sigma)
    loss_D_fit1_sigma = {}
    loss_D_fit1_min_sigma = {}
    loss_D_fit1_max_sigma = {}
    loss_D_fit1_err_sigma = {}
    for epsilon in best_fit_parameters1:
        print(epsilon)
        loss_D_fit1_sigma_epsilon = {}
        loss_D_fit1_min_sigma_epsilon = {}
        loss_D_fit1_max_sigma_epsilon = {}
        loss_D_fit1_err_sigma_epsilon = {}
        for N in best_fit_parameters1[epsilon]:
            intensity_evolution = [1.]
            intensity_evolution_min = [1.]
            intensity_evolution_max = [1.]
            intensity_evolution_err = [0.]
            for time in n_turns:
                intensity_evolution.append(
                    multiple_partition_intensity(
                        best_fit_parameters1[epsilon][N],
                        pass_params_fit1,
                        N, time, sigma))
                intensity_evolution_min.append(
                    multiple_partition_intensity(
                        best_fit_parameters1[epsilon][N],
                        pass_params_fit1_min,
                        N, time, sigma))
                intensity_evolution_max.append(
                    multiple_partition_intensity(
                        best_fit_parameters1[epsilon][N],
                        pass_params_fit1_max,
                        N, time, sigma))
                intensity_evolution_err.append(
                    error_loss_estimation(
                        best_fit_parameters1[epsilon][N],
                        pass_params_fit1, contour_data[epsilon],
                        N, time, sigma))
            loss_D_fit1_sigma_epsilon[N] = np.asarray(intensity_evolution)
            loss_D_fit1_min_sigma_epsilon[N] = np.asarray(intensity_evolution_min)
            loss_D_fit1_max_sigma_epsilon[N] = np.asarray(intensity_evolution_max)
            loss_D_fit1_err_sigma_epsilon[N] = np.asarray(intensity_evolution_err)
        loss_D_fit1_sigma[epsilon] = loss_D_fit1_sigma_epsilon
        loss_D_fit1_min_sigma[epsilon] = loss_D_fit1_min_sigma_epsilon
        loss_D_fit1_max_sigma[epsilon] = loss_D_fit1_max_sigma_epsilon
        loss_D_fit1_err_sigma[epsilon] = loss_D_fit1_err_sigma_epsilon
    loss_D_fit1[sigma] = loss_D_fit1_sigma
    loss_D_fit1_min[sigma] = loss_D_fit1_min_sigma
    loss_D_fit1_max[sigma] = loss_D_fit1_max_sigma
    loss_D_fit1_err[sigma] = loss_D_fit1_err_sigma

#%%
print("loss considering FIT2")
loss_D_fit2 = {}
loss_D_fit2_min = {}
loss_D_fit2_max = {}
loss_D_fit2_err = {}
for sigma in sigmas:
    print(sigma)
    loss_D_fit2_sigma = {}
    loss_D_fit2_min_sigma = {}
    loss_D_fit2_max_sigma = {}
    loss_D_fit2_err_sigma = {}
    for epsilon in best_fit_parameters1:
        print(epsilon)
        loss_D_fit2_sigma_epsilon = {}
        loss_D_fit2_min_sigma_epsilon = {}
        loss_D_fit2_max_sigma_epsilon = {}
        loss_D_fit2_err_sigma_epsilon = {}
        for N in best_fit_parameters1[epsilon]:
            intensity_evolution = [1.]
            intensity_evolution_min = [1.]
            intensity_evolution_max = [1.]
            intensity_evolution_err = [0.]
            for time in n_turns:
                intensity_evolution.append(
                    multiple_partition_intensity(
                        best_fit_parameters2_doublescan[epsilon][N],
                        pass_params_fit2,
                        N, time, sigma))
                intensity_evolution_min.append(
                    multiple_partition_intensity(
                        best_fit_parameters2_doublescan[epsilon][N],
                        pass_params_fit2_min,
                        N, time, sigma))
                intensity_evolution_max.append(
                    multiple_partition_intensity(
                        best_fit_parameters2_doublescan[epsilon][N],
                        pass_params_fit2_max,
                        N, time, sigma))
                intensity_evolution_err.append(
                    error_loss_estimation(
                        best_fit_parameters2_doublescan[epsilon][N],
                        pass_params_fit2, contour_data[epsilon],
                        N, time, sigma))
            loss_D_fit2_sigma_epsilon[N] = np.asarray(intensity_evolution)
            loss_D_fit2_min_sigma_epsilon[N] = np.asarray(intensity_evolution_min)
            loss_D_fit2_max_sigma_epsilon[N] = np.asarray(intensity_evolution_max)
            loss_D_fit2_err_sigma_epsilon[N] = np.asarray(intensity_evolution_err)
        loss_D_fit2_sigma[epsilon] = loss_D_fit2_sigma_epsilon
        loss_D_fit2_min_sigma[epsilon] = loss_D_fit2_min_sigma_epsilon
        loss_D_fit2_max_sigma[epsilon] = loss_D_fit2_max_sigma_epsilon
        loss_D_fit2_err_sigma[epsilon] = loss_D_fit2_err_sigma_epsilon
    loss_D_fit2[sigma] = loss_D_fit2_sigma
    loss_D_fit2_min[sigma] = loss_D_fit2_min_sigma
    loss_D_fit2_max[sigma] = loss_D_fit2_max_sigma
    loss_D_fit2_err[sigma] = loss_D_fit2_err_sigma

#%%
print("Reverse loss for dynamic aperture")

D_from_loss_all = {}
D_from_loss_mainregion = {}

for sigma in loss_all:
    D_from_loss_all_sigma = {}
    D_from_loss_mainregion_sigma = {}
    for epsilon in loss_all[sigma]:
        D_from_loss_all_sigma[epsilon] = D_from_loss(
            np.copy(loss_all[sigma][epsilon][1:]),
            sigma)
        D_from_loss_mainregion_sigma[epsilon] = D_from_loss(
            np.copy(loss_mainregion[sigma][epsilon][1:]),
            sigma)
    D_from_loss_all[sigma] = D_from_loss_all_sigma
    D_from_loss_mainregion[sigma] = D_from_loss_mainregion_sigma

#%%
print("Fit1 on loss all")

k_min = -20.
k_max = 7.
dk = 0.1
n_iterations = 7

fit1_loss_all = {}
for sigma in loss_all:
    print(sigma)
    fit1_loss_all_sigma = {}
    for epsilon in loss_all[sigma]:
        _, fit1_loss_all_sigma[epsilon] = non_linear_fit1_iterated(
            dict(zip(n_turns,
                D_from_loss_all[sigma][epsilon])),
            dict(zip(n_turns,
                D_from_loss_all[sigma][epsilon] * 0.001)),
            n_turns, k_min, k_max, dk, n_iterations)
    fit1_loss_all[sigma] = fit1_loss_all_sigma

#%%
print("plot the fit result")
for sigma in loss_all:
    for epsilon in loss_all[sigma]:
        print(sigma, epsilon)
        plot_fit_loss1(fit1_loss_all[sigma][epsilon],
            sigma, epsilon, n_turns, D_from_loss_all,
            "all", "img/loss/fit1_all")

#%%
print("and now loss")
loss_fit1_loss_all = {}
loss_fit1_loss_all_min = {}
loss_fit1_loss_all_max = {}
loss_fit1_loss_all_err = {}
for sigma in sigmas:
    print(sigma)
    loss_fit1_loss_all_sigma = {}
    loss_fit1_loss_all_min_sigma = {}
    loss_fit1_loss_all_max_sigma = {}
    loss_fit1_loss_all_err_sigma = {}
    for epsilon in fit1_loss_all[sigma]:
        print(epsilon)
        intensity_evolution = [1.]
        intensity_evolution_min = [1.]
        intensity_evolution_max = [1.]
        intensity_evolution_err = [0.]
        for time in n_turns:
            intensity_evolution.append(
                single_partition_intensity(fit1_loss_all[sigma][epsilon],
                                           pass_params_fit1, time, sigma))
            intensity_evolution_min.append(
                single_partition_intensity(fit1_loss_all[sigma][epsilon],
                                           pass_params_fit1_min, time, sigma))
            intensity_evolution_max.append(
                single_partition_intensity(fit1_loss_all[sigma][epsilon],
                                           pass_params_fit1_max, time, sigma))
            intensity_evolution_err.append(
                error_loss_estimation_single_partition(
                                            fit1_loss_all[sigma][epsilon],
                                            pass_params_fit1,
                                            contour_data[epsilon],
                                            time, sigma))
        loss_fit1_loss_all_sigma[epsilon] = np.asarray(intensity_evolution)
        loss_fit1_loss_all_min_sigma[epsilon] = np.asarray(intensity_evolution_min)
        loss_fit1_loss_all_max_sigma[epsilon] = np.asarray(intensity_evolution_max)
        loss_fit1_loss_all_err_sigma[epsilon] = np.asarray(intensity_evolution_err)
    loss_fit1_loss_all[sigma] = loss_fit1_loss_all_sigma
    loss_fit1_loss_all_min[sigma] = loss_fit1_loss_all_min_sigma
    loss_fit1_loss_all_max[sigma] = loss_fit1_loss_all_max_sigma
    loss_fit1_loss_all_err[sigma] = loss_fit1_loss_all_err_sigma

#%%
print("Fit2 on loss all")

da = 0.0001
a_max = 0.01
a_min = (1 / n_turns[0]) + da ### under this value it doesn't converge
a_bound = 1e10
a_default = n_turns[-1]

fit2_loss_all = {}
for sigma in loss_all:
    print(sigma)
    fit2_loss_all_sigma = {}
    for epsilon in loss_all[sigma]:
        _, fit2_loss_all_sigma[epsilon] = non_linear_fit2_doublescan(
            dict(zip(n_turns,
                D_from_loss_all[sigma][epsilon])),
            dict(zip(n_turns,
                D_from_loss_all[sigma][epsilon] * 0.001)),
            n_turns, 0.05, 0.01, a_min, a_max, da, a_bound, a_default)
    fit2_loss_all[sigma] = fit2_loss_all_sigma

#%%
print("plot the fit result")
for sigma in loss_all:
    for epsilon in loss_all[sigma]:
        print(sigma, epsilon)
        plot_fit_loss2(fit2_loss_all[sigma][epsilon],
            sigma, epsilon, n_turns, D_from_loss_all,
            "all", "img/loss/fit2_all")

#%%
print("and now loss")
loss_fit2_loss_all = {}
loss_fit2_loss_all_min = {}
loss_fit2_loss_all_max = {}
loss_fit2_loss_all_err = {}
for sigma in sigmas:
    print(sigma)
    loss_fit2_loss_all_sigma = {}
    loss_fit2_loss_all_min_sigma = {}
    loss_fit2_loss_all_max_sigma = {}
    loss_fit2_loss_all_err_sigma = {}
    for epsilon in fit2_loss_all[sigma]:
        print(epsilon)
        intensity_evolution = [1.]
        intensity_evolution_min = [1.]
        intensity_evolution_max = [1.]
        intensity_evolution_err = [0.]
        for time in n_turns:
            intensity_evolution.append(
                single_partition_intensity(fit2_loss_all[sigma][epsilon],
                                           pass_params_fit2, time, sigma))
            intensity_evolution_min.append(
                single_partition_intensity(fit2_loss_all[sigma][epsilon],
                                           pass_params_fit2_min, time, sigma))
            intensity_evolution_max.append(
                single_partition_intensity(fit2_loss_all[sigma][epsilon],
                                           pass_params_fit2_max, time, sigma))
            intensity_evolution_err.append(
                error_loss_estimation_single_partition(
                                            fit2_loss_all[sigma][epsilon],
                                            pass_params_fit2,
                                            contour_data[epsilon],
                                            time, sigma))
        loss_fit2_loss_all_sigma[epsilon] = np.asarray(intensity_evolution)
        loss_fit2_loss_all_min_sigma[epsilon] = np.asarray(intensity_evolution_min)
        loss_fit2_loss_all_max_sigma[epsilon] = np.asarray(intensity_evolution_max)
        loss_fit2_loss_all_err_sigma[epsilon] = np.asarray(intensity_evolution_err)
    loss_fit2_loss_all[sigma] = loss_fit2_loss_all_sigma
    loss_fit2_loss_all_min[sigma] = loss_fit2_loss_all_min_sigma
    loss_fit2_loss_all_max[sigma] = loss_fit2_loss_all_max_sigma
    loss_fit2_loss_all_err[sigma] = loss_fit2_loss_all_err_sigma

#%%
print("Fit1 on loss mainregion")

k_min = -20.
k_max = 7.
dk = 0.1
n_iterations = 7

fit1_loss_mainregion = {}
for sigma in loss_mainregion:
    print(sigma)
    fit1_loss_mainregion_sigma = {}
    for epsilon in loss_mainregion[sigma]:
        _, fit1_loss_mainregion_sigma[epsilon] = non_linear_fit1_iterated(
            dict(zip(n_turns,
                D_from_loss_mainregion[sigma][epsilon])),
            dict(zip(n_turns,
                D_from_loss_mainregion[sigma][epsilon] * 0.001)),
            n_turns, k_min, k_max, dk, n_iterations)
    fit1_loss_mainregion[sigma] = fit1_loss_mainregion_sigma

#%%
print("plot the fit result")
for sigma in loss_mainregion:
    for epsilon in loss_mainregion[sigma]:
        print(sigma, epsilon)
        plot_fit_loss1(fit1_loss_mainregion[sigma][epsilon],
            sigma, epsilon, n_turns, D_from_loss_mainregion,
            "mainregion", "img/loss/fit1_mainregion")

#%%
print("and now loss")
loss_fit1_loss_mainregion = {}
loss_fit1_loss_mainregion_min = {}
loss_fit1_loss_mainregion_max = {}
loss_fit1_loss_mainregion_err = {}
for sigma in sigmas:
    print(sigma)
    loss_fit1_loss_mainregion_sigma = {}
    loss_fit1_loss_mainregion_min_sigma = {}
    loss_fit1_loss_mainregion_max_sigma = {}
    loss_fit1_loss_mainregion_err_sigma = {}
    for epsilon in fit1_loss_mainregion[sigma]:
        print(epsilon)
        intensity_evolution = [1.]
        intensity_evolution_min = [1.]
        intensity_evolution_max = [1.]
        intensity_evolution_err = [0.]
        for time in n_turns:
            intensity_evolution.append(
                single_partition_intensity(fit1_loss_mainregion[sigma][epsilon],
                                           pass_params_fit1, time, sigma))
            intensity_evolution_min.append(
                single_partition_intensity(fit1_loss_mainregion[sigma][epsilon],
                                           pass_params_fit1_min, time, sigma))
            intensity_evolution_max.append(
                single_partition_intensity(fit1_loss_mainregion[sigma][epsilon],
                                           pass_params_fit1_max, time, sigma))
            intensity_evolution_err.append(
                error_loss_estimation_single_partition(
                                            fit1_loss_mainregion[sigma][epsilon],
                                            pass_params_fit1,
                                            contour_data[epsilon],
                                            time, sigma))
        loss_fit1_loss_mainregion_sigma[epsilon] = np.asarray(intensity_evolution)
        loss_fit1_loss_mainregion_min_sigma[epsilon] = np.asarray(intensity_evolution_min)
        loss_fit1_loss_mainregion_max_sigma[epsilon] = np.asarray(intensity_evolution_max)
        loss_fit1_loss_mainregion_err_sigma[epsilon] = np.asarray(intensity_evolution_err)
    loss_fit1_loss_mainregion[sigma] = loss_fit1_loss_mainregion_sigma
    loss_fit1_loss_mainregion_min[sigma] = loss_fit1_loss_mainregion_min_sigma
    loss_fit1_loss_mainregion_max[sigma] = loss_fit1_loss_mainregion_max_sigma
    loss_fit1_loss_mainregion_err[sigma] = loss_fit1_loss_mainregion_err_sigma

#%%
print("Fit2 on loss mainregion")

da = 0.0001
a_max = 0.01
a_min = (1 / n_turns[0]) + da ### under this value it doesn't converge
a_bound = 1e10
a_default = n_turns[-1]

fit2_loss_mainregion = {}
for sigma in loss_mainregion:
    print(sigma)
    fit2_loss_mainregion_sigma = {}
    for epsilon in loss_mainregion[sigma]:
        _, fit2_loss_mainregion_sigma[epsilon] = non_linear_fit2_doublescan(
            dict(zip(n_turns,
                D_from_loss_mainregion[sigma][epsilon])),
            dict(zip(n_turns,
                D_from_loss_mainregion[sigma][epsilon] * 0.001)),
            n_turns, 0.05, 0.01, a_min, a_max, da, a_bound, a_default)
    fit2_loss_mainregion[sigma] = fit2_loss_mainregion_sigma

#%%
print("plot the fit result")
for sigma in loss_mainregion:
    for epsilon in loss_mainregion[sigma]:
        print(sigma, epsilon)
        plot_fit_loss2(fit2_loss_mainregion[sigma][epsilon],
            sigma, epsilon, n_turns, D_from_loss_mainregion,
            "mainregion", "img/loss/fit2_mainregion")

#%%
print("and now loss")
loss_fit2_loss_mainregion = {}
loss_fit2_loss_mainregion_min = {}
loss_fit2_loss_mainregion_max = {}
loss_fit2_loss_mainregion_err = {}
for sigma in sigmas:
    print(sigma)
    loss_fit2_loss_mainregion_sigma = {}
    loss_fit2_loss_mainregion_min_sigma = {}
    loss_fit2_loss_mainregion_max_sigma = {}
    loss_fit2_loss_mainregion_err_sigma = {}
    for epsilon in fit2_loss_mainregion[sigma]:
        print(epsilon)
        intensity_evolution = [1.]
        intensity_evolution_min = [1.]
        intensity_evolution_max = [1.]
        intensity_evolution_err = [0.]
        for time in n_turns:
            intensity_evolution.append(
                single_partition_intensity(fit2_loss_mainregion[sigma][epsilon],
                                           pass_params_fit2, time, sigma))
            intensity_evolution_min.append(
                single_partition_intensity(fit2_loss_mainregion[sigma][epsilon],
                                           pass_params_fit2_min, time, sigma))
            intensity_evolution_max.append(
                single_partition_intensity(fit2_loss_mainregion[sigma][epsilon],
                                           pass_params_fit2_max, time, sigma))
            intensity_evolution_err.append(
                error_loss_estimation_single_partition(
                                            fit2_loss_mainregion[sigma][epsilon],
                                            pass_params_fit2,
                                            contour_data[epsilon],
                                            time, sigma))
        loss_fit2_loss_mainregion_sigma[epsilon] = np.asarray(intensity_evolution)
        loss_fit2_loss_mainregion_min_sigma[epsilon] = np.asarray(intensity_evolution_min)
        loss_fit2_loss_mainregion_max_sigma[epsilon] = np.asarray(intensity_evolution_max)
        loss_fit2_loss_mainregion_err_sigma[epsilon] = np.asarray(intensity_evolution_err)
    loss_fit2_loss_mainregion[sigma] = loss_fit2_loss_mainregion_sigma
    loss_fit2_loss_mainregion_min[sigma] = loss_fit2_loss_mainregion_min_sigma
    loss_fit2_loss_mainregion_max[sigma] = loss_fit2_loss_mainregion_max_sigma
    loss_fit2_loss_mainregion_err[sigma] = loss_fit2_loss_mainregion_err_sigma

#%%

print("Plot the losses!")

for sigma in sigmas:
    print(sigma)
    for epsilon in loss_all[sigma]:
        print(epsilon)
        # ### Just the losses (no fits)
        # plot_losses(
        #     ("Comparison of loss measures (All and Main Region),\n" +
        #     "$\sigma = {:2.2f}$, $\epsilon = {:2.0f}$".
        #     format(sigma, epsilon[2])),
        #     ("img/loss/loss_all_mainregion_sig{:2.2f}_eps{:2.0f}.png".
        #     format(sigma, epsilon[2])),
        #     n_turns,
        #     [loss_all[sigma][epsilon], loss_mainregion[sigma][epsilon]],
        #     ["All loss", "Mainregion loss"])

        # ### Precise and Precise Fit
        # plot_losses(
        #     ("Comparison of loss measures (all with all FITS),\n" +
        #         "$\sigma = {:2.2f}$, $\epsilon = {:2.0f}$".
        #         format(sigma, epsilon[2])),
        #     ("img/loss/loss_all_and_fits_sig{:2.2f}_eps{:2.0f}.png".
        #         format(sigma, epsilon[2])),
        #     n_turns,
        #     [loss_all[sigma][epsilon]],
        #     ["All loss"],
        #     [loss_fit1_loss_all[sigma][epsilon],
        #         loss_fit2_loss_all[sigma][epsilon]],
        #     [loss_fit1_loss_all_min[sigma][epsilon],
        #         loss_fit2_loss_all_min[sigma][epsilon]],
        #     [loss_fit1_loss_all_max[sigma][epsilon],
        #         loss_fit2_loss_all_max[sigma][epsilon]],
        #     [loss_fit1_loss_all_err[sigma][epsilon],
        #         loss_fit2_loss_all_err[sigma][epsilon]],
        #     ["D loss all FIT1", "D loss all FIT2"],
        #     fit_error=False,
        #     scan_error=False)

        # ### Anglescan and Anglescan Fit
        # plot_losses(
        #     ("Comparison of loss measures (mainregion with mainregion FITS),\n" +
        #                     "$\sigma = {:2.2f}$, $\epsilon = {:2.0f}$".
        #                     format(sigma, epsilon[2])),
        #     ("img/loss/loss_mainregion_and_fits_sig{:2.2f}_eps{:2.0f}.png".
        #                     format(sigma, epsilon[2])),
        #     n_turns,
        #     [loss_mainregion[sigma][epsilon]],
        #     ["mainregion loss"],
        #     [loss_fit1_loss_mainregion[sigma][epsilon],
        #         loss_fit2_loss_mainregion[sigma][epsilon]],
        #     [loss_fit1_loss_mainregion_min[sigma][epsilon],
        #         loss_fit2_loss_mainregion_min[sigma][epsilon]],
        #     [loss_fit1_loss_mainregion_max[sigma][epsilon],
        #         loss_fit2_loss_mainregion_max[sigma][epsilon]],
        #     [loss_fit1_loss_mainregion_err[sigma][epsilon],
        #         loss_fit2_loss_mainregion_err[sigma][epsilon]],
        #     ["D loss mainregion FIT1", "D loss mainregion FIT2"],
        #     fit_error=False,
        #     scan_error=False)

        # plot_losses(
        #     ("Comparison of loss measures (mainregion with D FITS),\n" +
        #                 "$\sigma = {:2.2f}$, $\epsilon = {:2.0f}$".
        #                 format(sigma, epsilon[2])),
        #     ("img/loss/loss_mainregion_and_Dfits_sig{:2.2f}_eps{:2.0f}.png".
        #                 format(sigma, epsilon[2])),
        #     n_turns,
        #     [loss_mainregion[sigma][epsilon]],
        #     ["Mainregion loss"],
        #     [loss_D_fit1[sigma][epsilon][N]
        #         for N in loss_D_fit1[sigma][epsilon]] +
        #         [loss_D_fit2[sigma][epsilon][N]
        #         for N in loss_D_fit2[sigma][epsilon]],
        #     [loss_D_fit1_min[sigma][epsilon][N]
        #         for N in loss_D_fit1_min[sigma][epsilon]] +
        #         [loss_D_fit2_min[sigma][epsilon][N]
        #         for N in loss_D_fit2_min[sigma][epsilon]],
        #     [loss_D_fit1_max[sigma][epsilon][N]
        #         for N in loss_D_fit1_max[sigma][epsilon]] +
        #         [loss_D_fit2_max[sigma][epsilon][N]
        #         for N in loss_D_fit2_max[sigma][epsilon]],
        #     [loss_D_fit1_err[sigma][epsilon][N]
        #         for N in loss_D_fit1_err[sigma][epsilon]] +
        #         [loss_D_fit2_err[sigma][epsilon][N]
        #         for N in loss_D_fit2_err[sigma][epsilon]],
        #     ["D loss FIT1, N $= {}$".format(N)
        #         for N in loss_D_fit1[sigma][epsilon]] +
        #         ["D loss FIT2, N $= {}$".format(N)
        #         for N in loss_D_fit2[sigma][epsilon]],
        #     fit_error=False,
        #     scan_error=False)

        plot_losses(
            ("Comparison of loss measures (loss mainregion, D FIT1),\n" +
                "$\sigma = {:2.2f}$, $\epsilon = {:2.0f}$".
                        format(sigma, epsilon[2])),
            ("img/loss/loss_mainregion_and_dfit1_sig{:2.2f}_eps{:2.0f}.png".
                        format(sigma, epsilon[2])),
            n_turns,
            [loss_mainregion[sigma][epsilon]],
            ["Mainregion loss"],
            ["C0"],
            [loss_D_fit1[sigma][epsilon][N] for N in range(1, 5)],
            [loss_D_fit1_min[sigma][epsilon][N] for N in range(1, 5)],
            [loss_D_fit1_max[sigma][epsilon][N] for N in range(1, 5)],
            [loss_D_fit1_err[sigma][epsilon][N] for N in range(1, 5)],
            ["D loss FIT1, N $= {}$".format(N) for N in range(1, 5)],
            ["C1", "C2", "C3", "C4"],
            fit_error=False,
            scan_error=False)

        plot_losses(
            ("Comparison of loss measures (loss mainregion, D FIT2),\n" +
                "$\sigma = {:2.2f}$, $\epsilon = {:2.0f}$".
                        format(sigma, epsilon[2])),
            ("img/loss/loss_mainregion_and_dfit2_sig{:2.2f}_eps{:2.0f}.png".
                        format(sigma, epsilon[2])),
            n_turns,
            [loss_mainregion[sigma][epsilon]],
            ["Mainregion loss"],
            ["C0"],
            [loss_D_fit2[sigma][epsilon][N] for N in range(1, 5)],
            [loss_D_fit2_min[sigma][epsilon][N] for N in range(1, 5)],
            [loss_D_fit2_max[sigma][epsilon][N] for N in range(1, 5)],
            [loss_D_fit2_err[sigma][epsilon][N] for N in range(1, 5)],
            ["D loss FIT2, N $= {}$".format(N) for N in range(1, 5)],
            ["C1", "C2", "C3", "C4"],
            fit_error=False,
            scan_error=False)

        plot_losses(
            ("Comparison of loss measures (loss mainregion, D FIT1 and FIT2),\n" +
                "$\sigma = {:2.2f}$, $\epsilon = {:2.0f}$".
                        format(sigma, epsilon[2])),
            ("img/loss/loss_mainregion_and_dfit1and2_sig{:2.2f}_eps{:2.0f}.png".
                        format(sigma, epsilon[2])),
            n_turns,
            [loss_mainregion[sigma][epsilon]],
            ["Mainregion loss"],
            ["C0"],
            [loss_D_fit1[sigma][epsilon][N] for N in range(1, 5)] +
                [loss_D_fit2[sigma][epsilon][N] for N in range(1, 5)],
            [loss_D_fit1_min[sigma][epsilon][N] for N in range(1, 5)] +
                [loss_D_fit2_min[sigma][epsilon][N] for N in range(1, 5)],
            [loss_D_fit1_max[sigma][epsilon][N] for N in range(1, 5)] +
                [loss_D_fit2_max[sigma][epsilon][N] for N in range(1, 5)],
            [loss_D_fit1_err[sigma][epsilon][N] for N in range(1, 5)] +
                [loss_D_fit2_err[sigma][epsilon][N] for N in range(1, 5)],
            ["D loss FIT1, N $= {}$".format(N) for N in range(1, 5)] +
                ["D loss FIT2, N $= {}$".format(N) for N in range(1, 5)],
            [(0.25, 0., 0.), (0.50, 0., 0.), (0.75, 0., 0.), (1., 0., 0.)] +
                [(0., 0.25, 0.), (0., 0.50, 0.), (0., 0.75, 0.), (0., 1., 0.)],
            fit_error=False,
            scan_error=False)

#%%
print("plot L2 norm")
for sigma in sigmas:
    print(sigma)
    for epsilon in loss_all[sigma]:
        print(epsilon)
        plot_l2(
            ("$L^2$-norm of difference between mainregion loss\n" +
             "and FIT1 over D loss, $\sigma = {:2.2f}$, $\epsilon = {:2.0f}$".
                        format(sigma, epsilon[2])),
             "img/loss/l2_mainregion_fit1_sig{:2.2f}_eps{:2.0f}.png".
                        format(sigma, epsilon[2]),
             n_turns,
             loss_mainregion[sigma][epsilon],
             loss_all[sigma][epsilon],
             [loss_D_fit1[sigma][epsilon][N]
                for N in loss_D_fit1[sigma][epsilon]])
        plot_l2(
            ("$L^2$-norm of difference between mainregion loss\n" +
             "and FIT2 over D loss, $\sigma = {:2.2f}$, $\epsilon = {:2.0f}$".
                        format(sigma, epsilon[2])),
             "img/loss/l2_mainregion_fit2_sig{:2.2f}_eps{:2.0f}.png".
                        format(sigma, epsilon[2]),
             n_turns,
             loss_mainregion[sigma][epsilon],
             loss_all[sigma][epsilon],
             [loss_D_fit2[sigma][epsilon][N]
                for N in loss_D_fit2[sigma][epsilon]])


################################################################################
################################################################################
################################################################################
###  PART THREE - LHC FITS AND ANALYSIS  #######################################
################################################################################
################################################################################
################################################################################
#%%
import pickle
import numpy as np
import matplotlib.pyplot as plt

from fit_library import *

#%%
print("LHC data.")
lhc_data = pickle.load(open("LHC_DATA.pkl", "rb"))
lhc_data = remove_first_times_lhc(lhc_data, 1000)

#%%
print("Compute FIT1 Final Version")

# Search parameters
k_min = -20.
k_max = 7.
dk = 0.1
n_iterations = 2

fit_lhc1 = {}
best_fit_lhc1 = {}

for label in lhc_data:
    fit_lhc1_label = {}
    best_fit_lhc1_label = {}
    for i in lhc_data[label]:
        j = 0
        print(label, i)
        fit_lhc1_correction = []
        best_fit_lhc1_correction = []
        for seed in lhc_data[label][i]:
            print(j)
            j += 1
            # FIT1
            fit, best = non_linear_fit1_iterated(seed,
                                                sigma_filler(seed, 0.05),
                                                np.asarray(sorted(seed.keys())),
                                                k_min, k_max,
                                                dk, n_iterations)
            fit_lhc1_correction.append(fit)
            best_fit_lhc1_correction.append(best)
        fit_lhc1_label[i] = fit_lhc1_correction
        best_fit_lhc1_label[i] = best_fit_lhc1_correction
    fit_lhc1[label] = fit_lhc1_label
    best_fit_lhc1[label] = best_fit_lhc1_label

#%%
print("Compute FIT2 doublescan")

k_min = 0.01
dk = 0.01

da = 0.0005
a_max = 0.01

fit_lhc2_doublescan = {}
best_fit_lhc2_doublescan = {}

for label in lhc_data:
    fit_lhc2_doublescan_label = {}
    best_fit_lhc2_doublescan_label = {}
    for i in lhc_data[label]:
        j = 0
        print(label, i)
        fit_lhc2_doublescan_correction = []
        best_fit_lhc2_doublescan_correction = []
        for seed in lhc_data[label][i]:
            print(j)
            j += 1
            a_default = np.asarray(sorted(seed.keys()))[-1]
            a_bound = np.asarray(sorted(seed.keys()))[-1] / a_max
            a_min = (1/np.asarray(sorted(seed.keys()))[0]) + da
            fit, best = non_linear_fit2_doublescan(
                            seed,
                            sigma_filler(seed, 0.05),
                            np.asarray(sorted(seed.keys())),
                            k_min, dk,
                            a_min, a_max, da, a_bound, a_default)
            fit_lhc2_doublescan_correction.append(fit)
            best_fit_lhc2_doublescan_correction.append(best)
        fit_lhc2_doublescan_label[i] = fit_lhc2_doublescan_correction
        best_fit_lhc2_doublescan_label[i] = best_fit_lhc2_doublescan_correction
    fit_lhc2_doublescan[label] = fit_lhc2_doublescan_label
    best_fit_lhc2_doublescan[label] = best_fit_lhc2_doublescan_label

#%%
print("Save fit")

print("I mean, you probably should.")
print("It depends on how much time it will take on your machine")
print("At a certain point, I was mainly using the ram saving feature on Spyder")

#%%
print("Is fit1 positive? Is fit2 bounded in a?")
fit1_lhc_pos = {}
fit2_lhc_bound = {}
for folder in best_fit_lhc1:
    fit1_pos_folder = {}
    fit2_bound_folder = {}
    for kind in best_fit_lhc1[folder]:
        fit1_pos_kind = []
        fit2_bound_kind = []
        for seed in best_fit_lhc1[folder][kind]:
            fit1_pos_kind.append(seed[4] > 0 and seed[0] > 0 and seed[2] > 0)
        for seed in best_fit_lhc2[folder][kind]:
            fit2_bound_kind.append(seed[4] < 1e+10)
        fit1_pos_folder[kind] = fit1_pos_kind
        fit2_bound_folder[kind] = fit2_bound_kind
    fit1_lhc_pos[folder] = fit1_pos_folder
    fit2_lhc_bound[folder] = fit2_bound_folder

## Is fit2 equal or better fit1?
flag = True
for folder in fit1_lhc_pos:
    for kind in fit1_lhc_pos[folder]:
        for i in range(len(fit1_lhc_pos[folder][kind])):
            if (fit1_lhc_pos[folder][kind][i] and
                    not fit2_lhc_bound[folder][kind][i]):
                print("DID NOT WORK FOR {}-{}".format(folder, kind))
                flag = False
print(flag)

#%%

print("general lhc plots.")

for folder in lhc_data:
    for kind in lhc_data[folder]:
        print(folder, kind)
        plot_lhc_fit(best_fit_lhc1[folder][kind],
                     lhc_data[folder][kind],
                     pass_params_fit1, folder + kind + "f1")
        plot_lhc_fit(best_fit_lhc2_doublescan[folder][kind],
                     lhc_data[folder][kind],
                     pass_params_fit2, folder + kind + "f2")

#%%
print("All chisquared!")

for folder in lhc_data:
    for kind in lhc_data[folder]:
        print(folder, kind)
        lhc_plot_chi_squared2_multiple(fit_lhc2_fixedk, fit_lhc2,
                                       folder, kind, k_values)

#%%
print("lhc best fit distribution1")

for label in best_fit_lhc1:
    for kind in best_fit_lhc1[label]:
        print(label, kind)
        best_fit_seed_distrib1(best_fit_lhc1[label][kind], label + kind + "f1")
        lhc_2param_comparison1(best_fit_lhc1[label][kind], label + kind + "f1")
        #lhc_plot_chi_squared1(fit_lhc1[label][kind], label, kind,
        #                      fit1_lhc_pos[label][kind],
        #                      fit2_lhc_bound[label][kind])
#%%
print("lhc best fit distribution2")

for label in best_fit_lhc2:
    for kind in best_fit_lhc2[label]:
        print(label, kind)
        best_fit_seed_distrib2(best_fit_lhc2_doublescan[label][kind],
                               label + kind + "f2")
        lhc_2param_comparison2(best_fit_lhc2_doublescan[label][kind],
                               label + kind + "f2")
        #lhc_plot_chi_squared2(fit_lhc2[label][kind], label, kind,
        #                      fit1_lhc_pos[label][kind],
        #                      fit2_lhc_bound[label][kind])

#%%
for label in best_fit_lhc2:
    for kind in best_fit_lhc2[label]:
        combine_plots_lhc1(label, kind)
        combine_plots_lhc2(label, kind)
        #combine_plots_lhc3(label, kind)

#%%
print("Nekoroshev data")

import pickle
import numpy as np
import matplotlib.pyplot as plt

from fit_library import *

################################################################################
################################################################################
################################################################################
###  FOURTH PART - BASIC FITS ON NEKOROSHEV SIMULATION DATA  ###################
################################################################################
################################################################################
################################################################################

print("load data")

nek_data = pickle.load(open("data_nek_dictionary.pkl", "rb"))

#%%
print("reverse engeneering D from intensity")

nek_D = {}
for label in nek_data:
    nek_D[label] = (nek_data[label][0],
                    D_from_loss(nek_data[label][1], 1))

#%%
print("fit1")

# Search parameters
k_min = -20.
k_max = 7.
dk = 0.1
n_iterations = 7

nek_fit1 = {}
for label in nek_D:
    print(label)
    _, nek_fit1[label] = non_linear_fit1_iterated(
                            dict(zip(nek_D[label][0], nek_D[label][1])),
                            dict(zip(nek_D[label][0], nek_D[label][1] * 0.001)),
                            nek_D[label][0],
                            k_min, k_max, dk, n_iterations)

#%%
print("plot the things1")

for label in nek_fit1:
    plot_fit_nek1(nek_fit1[label], label,
                  nek_D[label][0],
                  dict(zip(nek_D[label][0], nek_D[label][1])),
                  dict(zip(nek_D[label][0], nek_D[label][1] * 0.001)))


#%%
print("fit2")

da = 0.0001
a_max = 0.01
a_min = 0.001 + da ### under this value it doesn't converge
a_bound = 1e20

nek_fit2 = {}
for label in nek_D:
    print(label)
    a_default = nek_D[label][0][-1]
    _, nek_fit2[label] = non_linear_fit2_doublescan(
            dict(zip(nek_D[label][0], nek_D[label][1])),
            dict(zip(nek_D[label][0], nek_D[label][1] * 0.001)),
            nek_D[label][0],
            0.05, 0.01,
            a_min, a_max, da, a_bound, a_default)

#%%
print("plot the things2")

for label in nek_fit2:
    plot_fit_nek2(nek_fit2[label], label,
                  nek_D[label][0],
                  dict(zip(nek_D[label][0], nek_D[label][1])),
                  dict(zip(nek_D[label][0], nek_D[label][1] * 0.001)),
                  imgpath="img/nek/fit2_standard_")
    plot_fit_nek2(nek_fit2[label], label,
                  nek_D[label][0],
                  dict(zip(nek_D[label][0], nek_D[label][1])),
                  dict(zip(nek_D[label][0], nek_D[label][1] * 0.001)),
                  "img/nek/fit2_standard_log_", True)

#%%
print("combine!")
combine_image_6x2("img/nek/combine_linear.png",
    "img/nek/fit2_fixedk__label6a.png", "img/nek/fit2_fixedk__label6b.png",
    "img/nek/fit2_fixedk__label7a.png", "img/nek/fit2_fixedk__label7b.png",
    "img/nek/fit2_fixedk__label7c.png", "img/nek/fit2_fixedk__label7d.png",
    "img/nek/fit2_standard__label6a.png", "img/nek/fit2_standard__label6b.png",
    "img/nek/fit2_standard__label7a.png", "img/nek/fit2_standard__label7b.png",
    "img/nek/fit2_standard__label7c.png", "img/nek/fit2_standard__label7d.png")

combine_image_6x2("img/nek/combine_log.png",
    "img/nek/fit2_fixedk_log__label6a.png", "img/nek/fit2_fixedk_log__label6b.png",
    "img/nek/fit2_fixedk_log__label7a.png", "img/nek/fit2_fixedk_log__label7b.png",
    "img/nek/fit2_fixedk_log__label7c.png", "img/nek/fit2_fixedk_log__label7d.png",
    "img/nek/fit2_standard_log__label6a.png", "img/nek/fit2_standard_log__label6b.png",
    "img/nek/fit2_standard_log__label7a.png", "img/nek/fit2_standard_log__label7b.png",
    "img/nek/fit2_standard_log__label7c.png", "img/nek/fit2_standard_log__label7d.png")

#%%
print("Draw 2D stability maps")

stability_levels = np.array([1000, 10000, 100000, 1000000, 10000000])

for epsilon in data:
    fig, ax = plt.subplots()
    for level in stability_levels:
        x = []
        y = []
        x.append(0.)
        y.append(0.)
        for line in sorted(data[epsilon]):
            j = 0
            while data[epsilon][line][j] >= level:
                j += 1
            x.append((j - 1) * dx * np.cos(line))
            y.append((j - 1) * dx * np.sin(line))
        ax.fill(x, y, label="N turns $= 10^{}$".format(int(np.log10(level))))
    ax.legend()
    ax.set_xlabel("X coordinate (A.U.)")
    ax.set_ylabel("Y coordinate (A.U.)")
    ax.set_xlim(0,0.8)
    ax.set_ylim(0,0.8)
    ax.set_aspect("equal", "box")
    ax.grid(True)
    ax.set_title("Stability map after N turns, $\epsilon = {:2.0f}$".format(epsilon[2], level))
    fig.savefig("img/stabmap_eps{:2.0f}_N{}.png".format(epsilon[2], level), dpi = DPI)
    plt.close()
#%%
from png_to_jpg import png_to_jpg

png_to_jpg("img/")
