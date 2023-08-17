import pandas as pd
import matplotlib.pylab as plt
import numpy as np
import emcee
import matplotlib.backends.backend_pdf
from tqdm import tqdm
import corner


def main():
	# path_df_transient_lightcurves = 'data/raw/transient_lightcurves_selected_fluxneg_3sigma.csv'
	
	path_df_transient_lightcurves = 'data/raw/transient_lightcurves_triggers.csv'
	
	transient_ids = {
		'TranID906221090914122502': 'CV 10',
		'TranID1204191180784142832': 'Flare 11',
		'TranID1607081520484130475': 'Blazar 21',
		'TranID1504291690384116269': 'HPM 27',
		'TranID1305131120734140944': 'AGN/SN 39',
		'TranID1101140070504144585': 'CV 96',
		'TranID909160120034110694': 'SN 101',
		# '': '',
		'TranID1001121180684103058': 'SN 350',
		# 'TranID1109231260224137668': 'YSO 337'
		# '': '',
	}
	"""
	path_df_transient_lightcurves = 'data/raw/TranIDTestEarlyLightcurve.csv'

	transient_ids = {
		'TranIDTest': 'CV 10',
		# '': '',
	}
	"""
	
	df_transient_lightcurves = pd.read_csv(path_df_transient_lightcurves, error_bad_lines=False)
	
	"""
	for transient_id in transient_ids.keys():
		if (df_transient_lightcurves['ID'] == transient_id).value_counts()[True]>0:
			print(transient_id)
	"""
	# print(transient_ids.keys())
	
	# TODO: Seleccionar random del dataset
	# df_random_selected = get_random_lightcurves(df_transient_lightcurves)
	run_transients(df_transient_lightcurves, transient_ids)


def run_transients(df_transient_lightcurves, transient_ids):
	pdf = matplotlib.backends.backend_pdf.PdfPages("output.pdf")
	
	for transient_id in tqdm(transient_ids.keys()):
		plot_figures(df_transient_lightcurves, pdf, transient_id)
	
	pdf.close()


def plot_figures(df_transient_lightcurves, pdf, transient_id):
	fig1, axes = plt.subplots(1, 2, figsize=(20, 4))
	i = 0
	df_transient, flux, fluxerr, mean_flux, median_flux, t, t_min, t_min_integer, t_peak, t_peak_integer, t_trigger, max_flux, min_flux = get_transient_and_parameters(
		df_transient_lightcurves, transient_id)
	
	sampler, bestpars, t0_estimate = fit_early_lightcurve(flux, fluxerr, mean_flux, median_flux, t, t_trigger, max_flux,
														  min_flux, t_peak, t_min)
	
	a_estimate, axis_x, c_estimate, early_lightcurve = early_lightcurve_tabular(bestpars, t0_estimate,
																				t_min_integer, t_peak_integer)
	
	plot_scatter(axes, df_transient, i, transient_id)
	
	plot_early_lightcurve(a_estimate, axes, axis_x, c_estimate, df_transient, early_lightcurve, fig1, i,
						  t0_estimate, t_peak_integer, t_trigger)
	
	pdf.savefig(fig1)
	
	fig2 = plot_corner(sampler)
	pdf.savefig(fig2)


def early_lightcurve_tabular(bestpars, t0_estimate, t_min_integer, t_peak_integer):
	# el start si lo dejamos desde el t_min o lo recortamos?
	axis_x = np.linspace(start=t_min_integer, stop=t_peak_integer, num=1000)
	a_estimate, c_estimate = bestpars
	early_lightcurve = np.heaviside((axis_x - t0_estimate), 1) * (a_estimate * (axis_x - t0_estimate) ** 2) + c_estimate
	return a_estimate, axis_x, c_estimate, early_lightcurve


def fit_early_lightcurve(flux, fluxerr, mean_flux, median_flux, t, t_trigger, max_flux, min_flux, t_peak, t_min):
	sampler, ndim, bestpars = markov_chain_montecarlo(t, flux, fluxerr, t_trigger, median_flux, mean_flux, max_flux,
													  min_flux, t_peak, t_min)
	burn = 200
	samples = sampler.chain[:, burn:, :].reshape((-1, ndim))
	samples[:, 2] = np.exp(samples[:, 2])
	bestpars2 = list(map(lambda v: (v[0]), zip(*np.percentile(samples, [50], axis=0))))
	print(bestpars)
	print('b2', bestpars2)
	print(bestpars - bestpars2)
	t0_estimate = bestpars[0]
	bestpars = bestpars[1:]
	return sampler, bestpars, t0_estimate


def get_transient_and_parameters(df_transient_lightcurves, transient_id):
	df_transient = df_transient_lightcurves[df_transient_lightcurves.ID == transient_id]
	
	flux_peak = df_transient['FluxNeg'].max(skipna=True)
	t_peak = df_transient[df_transient['FluxNeg'] == flux_peak]['MJD'].iloc[0]
	t_peak_integer = int(t_peak)
	t_min = df_transient['MJD'].min(skipna=True)
	t_min_integer = int(t_min)
	
	t_trigger = df_transient[df_transient['TriggerObsFluxNeg5'].notnull()].iloc[0]['MJD']
	t_trigger_integer = int(t_trigger)
	median_flux = df_transient['FluxNeg'].median(skipna=True)
	mean_flux = df_transient['FluxNeg'].mean(skipna=True)
	min_flux = df_transient['FluxNeg'].min(skipna=True)
	max_flux = df_transient['FluxNeg'].max(skipna=True)
	
	t = df_transient[0:df_transient[df_transient['MJD'] == t_peak].index[0] + 1]['MJD']
	flux = df_transient['FluxNeg']
	fluxerr = df_transient['FluxNegErr']
	return df_transient, flux, fluxerr, mean_flux, median_flux, t, t_min, t_min_integer, t_peak, t_peak_integer, t_trigger, max_flux, min_flux


def plot_early_lightcurve(a_estimate, axes, axis_x, c_estimate, df_transient, early_lightcurve, fig1, i, t0_estimate,
						  t_peak_integer, t_trigger):
	df_transient.plot.scatter(x='MJD', y='FluxNeg', c='DarkBlue', ax=axes[1])
	axes[1].errorbar(df_transient['MJD'], df_transient['FluxNeg'], yerr=df_transient['FluxNegErr'], fmt='o')
	fig1.suptitle(r'$a(t-t_{0})^2 \cdot H(t-t_{0})+c=[' + str(round(a_estimate, 10)) + '(t - ' + str(
		round(t0_estimate, 2)) + ')^2] \cdot H(t - ' + str(round(t0_estimate, 2)) + ')+' + str(
		round(c_estimate, 10)) + '$')
	axes[1].plot(axis_x, early_lightcurve)
	axes[1].axvline(x=t_trigger, color='r', linestyle='dotted')
	axes[1].axvline(x=t0_estimate, color='g', linestyle='dotted')
	last_point = np.heaviside((t_peak_integer - t0_estimate), 1) * (
	  a_estimate * (t_peak_integer - t0_estimate) ** 2) + c_estimate
	axes[1].axhline(y=last_point, color='grey', linestyle='dashed')


def plot_scatter(axes, df_transient, i, transient_id):
	df_transient.plot.scatter(x='MJD', y='FluxNeg', c='DarkBlue', ax=axes[0])
	axes[0].errorbar(df_transient['MJD'], df_transient['FluxNeg'], yerr=df_transient['FluxNegErr'], fmt='o')
	axes[0].set_title(transient_id)


def plot_corner(sampler):
	flat_samples = sampler.get_chain(discard=100, thin=15, flat=True)
	# https://emcee.readthedocs.io/en/stable/user/sampler/?highlight=get_chain#emcee.EnsembleSampler.get_chain
	print(len(flat_samples))
	labels = ["t_0", "a", "c"]
	return corner.corner(flat_samples, labels=labels)


def markov_chain_montecarlo(t, flux, fluxerr, t_trigger, median_flux, mean_flux, max_flux, min_flux, t_peak, t_min):
	nsteps = 1000
	
	t_sub = t_min if t_trigger - t_min > 35 else 35
	divisor = (t_peak - t_trigger) ** 2 if t_peak != t_trigger else 1
	
	pos = np.array([[np.random.uniform(t_trigger - t_sub, t_trigger),
					 np.random.uniform(-((max_flux - min_flux) / divisor),
									   ((max_flux - min_flux) / divisor)),
					 np.random.uniform(0, max_flux)] for i in range(200)])
	
	nwalkers, ndim = pos.shape
	
	sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability,
									args=(t, flux, fluxerr, t_trigger, min_flux, max_flux, t_peak))
	# sampler.run_mcmc(pos, 500, progress=True);
	
	# FInd parameters of lowest chi2
	pos, prob, state = sampler.run_mcmc(pos, 500)
	
	opos, oprob, orstate = [], [], []
	for pos, prob, rstate in sampler.sample(pos, prob, state, iterations=nsteps):
		opos.append(pos.copy())
		oprob.append(prob.copy())
	pos = np.array(opos)
	prob = np.array(oprob)
	nstep, nwalk = np.unravel_index(prob.argmax(), prob.shape)
	bestpars = pos[nstep, nwalk]
	posterior = prob
	
	return sampler, ndim, bestpars


def log_likelihood(theta, times, fluxes, fluxerrs):
	t0 = theta[0]
	a = theta[1]
	c = theta[2]
	
	chi2 = 0
	for i, value in times.items():
		model = np.heaviside((times[i] - t0), 1) * (a * (times[i] - t0) ** 2) + c
		chi2 += (fluxes[i] - model) ** 2 / fluxerrs[i] ** 2
	
	return -0.5 * chi2


def log_prior(theta, t_trigger, min_flux, max_flux, t_peak):
	t0 = theta[0]
	# pars = theta[1:]
	a = theta[1]
	c = theta[2]
	
	# for par in pars:
	# 	if par > 1 or par < 0:
	# 		return -np.inf
	# # else:
	# #	return 0.0
	
	# 	pos = np.array([[np.random.uniform(t_trigger - 500, t_trigger), np.random.uniform(0, ((max_flux - min_flux) / ((t_peak - t_trigger) ** 2))), np.random.uniform(0, max_flux)] for i in range(1000)])
	
	divisor = (t_peak - t_trigger) ** 2 if t_peak != t_trigger else 1
	
	if t_trigger - 35 < t0 < t_trigger and -((max_flux - min_flux) / divisor) < a < (
	  (max_flux - min_flux) / divisor) and 0 < c < max_flux:
		return 0.0
	
	"""
	if t_trigger - 500 < t0 < t_trigger:
		return 0.0

	if a > ((max_flux - min_flux) / ((t_peak - t_trigger) ** 2)) or a < -((max_flux - min_flux) / ((t_peak - t_trigger) ** 2)):
		return -np.inf

	if c > max_flux or c < 0:
		return -np.inf
	"""
	
	return -np.inf


def log_probability(theta, t, flux, fluxerr, t_trigger, min_flux, max_flux, t_peak):
	lp = log_prior(theta, t_trigger, min_flux, max_flux, t_peak)
	if not np.isfinite(lp):
		return -np.inf
	return lp + log_likelihood(theta, t, flux, fluxerr)


def get_random_lightcurves(df_transient_lightcurves):
	path_df_transient_classification = 'data/raw/classification_transient_lightcurves_with_10_or_more_observations.csv'
	df_transient_classification = pd.read_csv(path_df_transient_classification, error_bad_lines=False)


# return df_random_selected

main()
