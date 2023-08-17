import pandas as pd
import numpy as np
import emcee
from tqdm import tqdm


def main():
	path_df_transient_lightcurves = 'data/raw/transient_lightcurves_selected_fluxneg_3sigma.csv'
	df_transient_lightcurves = pd.read_csv(path_df_transient_lightcurves, error_bad_lines=False)
	
	path_df_classification_transient_lightcurves = 'data/raw/classification_transient_lightcurves_triggered_selected.csv'
	df_classification_transient_lightcurves = pd.read_csv(path_df_classification_transient_lightcurves,
														  error_bad_lines=False)
	
	run_early_lightcurve(df_transient_lightcurves, df_classification_transient_lightcurves)


def run_early_lightcurve(df_transient_lightcurves, transients):
	
	# only model events with no t_0 parameter calculated, to not repeat any calculation
	df_transients_non_early_lightcurve_modeled = transients[transients['t0'].isna()]
	
	for index, row in tqdm(df_transients_non_early_lightcurve_modeled.iterrows()):
		df_transient, flux, fluxerr, mean_flux, median_flux, t, t_min, t_min_integer, t_peak, t_peak_integer, t_trigger, max_flux, min_flux = get_transient_and_parameters(
			df_transient_lightcurves, row['TranID'])
		
		sampler, bestpars, t0_estimate = fit_early_lightcurve(flux, fluxerr, mean_flux, median_flux, t, t_trigger,
															  max_flux, min_flux, t_peak, t_min)
		
		a_estimate, c_estimate = bestpars
		
		transients.loc[index, 'a'] = a_estimate
		transients.loc[index, 'c'] = c_estimate
		transients.loc[index, 't0'] = t0_estimate
		
		# save results every 5 iterations and when there are less than 5 events to finish, to maintain the file updated
		if index % 5 == 0 or len(transients) - index < 5:
			transients.to_csv('data/raw/classification_transient_lightcurves_triggered_selected.csv')


def fit_early_lightcurve(flux, fluxerr, mean_flux, median_flux, t, t_trigger, max_flux, min_flux, t_peak, t_min):
	sampler, ndim, bestpars = markov_chain_montecarlo(t, flux, fluxerr, t_trigger, median_flux, mean_flux, max_flux,
													  min_flux, t_peak, t_min)
	burn = 200
	samples = sampler.chain[:, burn:, :].reshape((-1, ndim))
	samples[:, 2] = np.exp(samples[:, 2])
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
	t_trigger = df_transient[df_transient['TriggerObsFluxNeg3'].notnull()].iloc[0]['MJD']
	t_trigger_integer = int(t_trigger)
	median_flux = df_transient['FluxNeg'].median(skipna=True)
	mean_flux = df_transient['FluxNeg'].mean(skipna=True)
	min_flux = df_transient['FluxNeg'].min(skipna=True)
	max_flux = df_transient['FluxNeg'].max(skipna=True)
	
	t = df_transient[0:df_transient[df_transient['MJD'] == t_peak].index[0] + 1]['MJD']
	flux = df_transient['FluxNeg']
	fluxerr = df_transient['FluxNegErr']
	return df_transient, flux, fluxerr, mean_flux, median_flux, t, t_min, t_min_integer, t_peak, t_peak_integer, t_trigger, max_flux, min_flux


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
	
	a = theta[1]
	c = theta[2]
	
	divisor = (t_peak - t_trigger) ** 2 if t_peak != t_trigger else 1
	
	if t_trigger - 35 < t0 < t_trigger and -((max_flux - min_flux) / divisor) < a < (
	  (max_flux - min_flux) / divisor) and 0 < c < max_flux:
		return 0.0
	
	return -np.inf


def log_probability(theta, t, flux, fluxerr, t_trigger, min_flux, max_flux, t_peak):
	lp = log_prior(theta, t_trigger, min_flux, max_flux, t_peak)
	if not np.isfinite(lp):
		return -np.inf
	return lp + log_likelihood(theta, t, flux, fluxerr)


main()
