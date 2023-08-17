import statistics
from tqdm import tqdm
import pandas as pd


def get_cumulative_mean_and_stdev(df, calculated_with):
	events = df.ID.unique()
	position_of_value = 4
	if calculated_with == 'FluxPos':
		position_of_value = 7
	if calculated_with == 'FluxNeg':
		position_of_value = 8
	for event in tqdm(events):
		bright_values = []
		for observation in df[df.ID == event].itertuples(name=None):
			bright_values.append(observation[position_of_value])
			if len(bright_values) > 1:
				mean = statistics.mean(bright_values)
				st_dev = statistics.stdev(bright_values)

				df.loc[observation[1], 'CumMean' + calculated_with] = mean
				df.loc[observation[1], 'CumStd' + calculated_with] = st_dev


def calculate_limits(df, calculated_with, sigma):
	df['LowerLimit' + calculated_with + str(sigma)] = df['CumMean' + calculated_with] - (
		df['CumStd' + calculated_with] * sigma)
	df['UpperLimit' + calculated_with + str(sigma)] = df['CumMean' + calculated_with] + (
		df['CumStd' + calculated_with] * sigma)


def find_triggers(df, calculated_with, sigma):
	events = pd.DataFrame(df.ID.unique())
	for event in tqdm(events):
		i = 3
		for observation in df[df.ID == event][2:].itertuples(name=None):
			if df.loc[observation[1], calculated_with] <= df.loc[
				observation[1] - 1, 'LowerLimit' + calculated_with + str(sigma)]:
				df.loc[observation[1], 'Trigger' + calculated_with + str(sigma)] = 'Lower'
				df.loc[observation[1], 'TriggerObs' + calculated_with + str(sigma)] = i
				break
			if df.loc[observation[1], calculated_with] >= df.loc[
				observation[1] - 1, 'UpperLimit' + calculated_with + str(sigma)]:
				df.loc[observation[1], 'Trigger' + calculated_with + str(sigma)] = 'Upper'
				df.loc[observation[1], 'TriggerObs' + calculated_with + str(sigma)] = i
				break
			i += 1

