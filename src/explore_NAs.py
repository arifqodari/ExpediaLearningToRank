import numpy as np
import pandas as pd
import matplotlib.pyplot as pl
from setting import *
from main import read_training_data

trainReader = read_training_data()

# Function to do the statistics of NAs for each chunk
def stat_na(data_chunk):
	NaNs = list()
	Non_NaNs = list()
	for i in data_chunk.columns[2:53]:
		a = len(data_chunk[i][pd.Series.isnull(data_chunk[i])])
		b = len(data_chunk[i][pd.Series.isnull(data_chunk[i])== False])
		NaNs.append(a)
		Non_NaNs.append(b)
	NaN_statistics = {'NaN':NaNs,'Non_NaNs':Non_NaNs}
	NaN_statistics = pd.DataFrame(NaN_statistics,index=data_chunk.columns[2:53])
	return NaN_statistics

if __name__ == "__main__":
	trainReader = read_training_data()

	NaN_stat_total = list()
	for data_chunk in trainReader:
		NaN_stat = stat_na(data_chunk)
		NaN_stat_total.append(NaN_stat)
	NaN_stat_total = sum(NaN_stat_total)

	#Make the NA stat data in the form of percentage
	NaN_stat_total = NaN_stat_total.divide(NaN_stat_total.sum(axis=1),axis=0)
	#Make the NA stat data sorted by NA column descendingly
	NaN_stat_total_ordered = NaN_stat_total.sort(columns='NaN',ascending=False)

	print NaN_stat_total

	print NaN_stat_total_ordered

	positions = list()
	for i in range(19):
		for j in range(3):
			positions.append((i,j))
	fig, axes = pl.subplots(nrows=19, ncols=3)
	for (index,(i,j)) in zip(NaN_stat_total.index,positions):
		NaN_stat_total.loc[index].plot(ax=axes[i,j],kind='barh',title = index)
	pl.show()