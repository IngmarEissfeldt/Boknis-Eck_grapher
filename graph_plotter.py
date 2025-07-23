import streamlit as st
import pandas as pd
import numpy as np
import kaleido
import plotly.graph_objects as go
import io
from datetime import datetime, timedelta


#Plots selected data.
import pandas as pd
import plotly.graph_objects as go

import pandas as pd
import plotly.graph_objects as go

def plot_data(
		df_full: pd.DataFrame,
		vars: list[str],
		flags: list[str],
		depth: list[int],
		show_flags: bool = False,
		scatterplot: bool = False,
	) -> go.Figure:
	"""
	If scatterplot=True, makes a scatter of vars[0] vs vars[1].
	Otherwise plots each var over the index (lines+markers),
	and if show_flags=True overlays:
	- red dots where flag == 9 (missing)
	- orange dots where flag == 7 (inaccurate)
	"""
	fig = go.Figure()
	
	
	if scatterplot:
		# require exactly two variables
		if len(vars) != 2:
			raise ValueError("scatterplot=True requires exactly two variables in `vars`.")
		x_var, y_var = vars

		idx  = pd.IndexSlice
		depth =[int(x) for x in depth]
		df = df_full.loc[idx[:, depth], :]

		# Base scatter
		fig.add_trace(go.Scatter(
			x=df[x_var],
			y=df[y_var],
			mode='markers',
			name=f'{x_var} vs {y_var}'
		))

		# Optionally overlay flagged points on the scatter
		if show_flags:
			for var, flag_col in zip(vars, flags):
				# align flags to their variable: color by missing/inacc on the *other* axis
				idx_missing = df.index[df[flag_col] == 9]
				idx_inacc   = df.index[df[flag_col] == 7]

				# missing
				if not idx_missing.empty:
					fig.add_trace(go.Scatter(
						x=df.loc[idx_missing, x_var],
						y=df.loc[idx_missing, y_var],
						mode='markers',
						marker=dict(color='red', size=10, symbol='x'),
						name=f'{var} missing',
						showlegend=False
					))
				# inaccurate
				if not idx_inacc.empty:
					fig.add_trace(go.Scatter(
						x=df.loc[idx_inacc, x_var],
						y=df.loc[idx_inacc, y_var],
						mode='markers',
						marker=dict(color='orange', size=10, symbol='diamond'),
						name=f'{var} inaccurate',
						showlegend=False
					))

		fig.update_layout(
			xaxis_title=x_var,
			yaxis_title=y_var,
			height=600,
			margin=dict(t=30, b=30),
		)
		return fig

	
	df = df_full.xs(int(depth[0]), level='Depth water [m]')	
				
	# time-series by default
	for var in vars:
		fig.add_trace(go.Scatter(
		x=df.index,
		y=df[var],
		mode='lines+markers',
		name=var
	))

	if show_flags:
		# compute y-baseline for flags
		ymin = df[vars].min().min()
		y_offset = (df[vars].max().max() - ymin) * 0.05
		marker_y = ymin - y_offset

		for var_col, flag_col in zip(vars, flags):
			idx_missing = df.index[df[flag_col] == 9]
			idx_inacc   = df.index[df[flag_col] == 7]

			if not idx_missing.empty:
				fig.add_trace(go.Scatter(
					x=idx_missing,
					y=[marker_y] * len(idx_missing),
					mode='markers',
					marker=dict(color='red', size=8),
					name=f'{var_col} missing',
					showlegend=False
				))

			if not idx_inacc.empty:
				fig.add_trace(go.Scatter(
					x=idx_inacc,
					y=[marker_y] * len(idx_inacc),
					mode='markers',
					marker=dict(color='orange', size=8),
					name=f'{var_col} inaccurate',
					showlegend=False
				))

	fig.update_layout(
	height=600,
	margin=dict(t=30, b=30)
	)

	return fig



#Sets download button
def download_button(plot, var_list, depth, element, key):

	var_map = {
		"Nitrate": "Nia", 
		"Nitrite": "Nii",
		"Phosphate": "P",  
		"Oxygen": "O", 
		"Salinity": "Sa", 
		"Silicate": "Si", 
		"Temperature": "T"
	}

	file_name = "BE_" + depth
	
	for var in var_list:
		file_name += "_" + var_map[var]
	
	file_name += ".png"

	#buffer for Download Button
	buf = io.BytesIO()
	plot.write_image(buf)
	buf.seek(0)  # rewind to beginning of buffer

	#Download button
	element.download_button(
		label=f"📥 Download plot ↑",
		data=buf,
		file_name=file_name,
		mime="image/png",
		key=key
	)

def select_variable(plotnum, scatterplot):
	to_plot = []
	depth = []
	if  not scatterplot:
		to_plot = st.sidebar.multiselect("Choose vars for plot " + str(plotnum), ["Nitrate", "Nitrite", "Oxygen", "Phosphate", "Salinity", "Silicate", "Temperature"], key=plotnum)
		depth = st.sidebar.selectbox("Choose depth in meter for plot " + str(plotnum), ["1", "5", "10", "15", "20", "25"], key=plotnum)
	else:
		to_plot.append(st.sidebar.selectbox("Choose var for plot " + str(plotnum), ["Nitrate", "Nitrite", "Oxygen", "Phosphate", "Salinity", "Silicate", "Temperature"], key=plotnum))
		depth.append(st.sidebar.selectbox("Choose depth in meter for first variable in plot " + str(plotnum), ["1", "5", "10", "15", "20", "25"], key=plotnum + 1))
				
		to_plot.append(st.sidebar.selectbox("Choose other var for plot " + str(plotnum), ["Nitrate", "Nitrite", "Oxygen", "Phosphate", "Salinity", "Silicate", "Temperature"], key=plotnum + 2))
		depth.append(st.sidebar.selectbox("Choose depth in meter for second variable in plot " + str(plotnum), ["1", "5", "10", "15", "20", "25"], key=plotnum + 3))

		if to_plot[0] == to_plot[1]:
			st.write("Please select 2 different variables for the scatter plot!")
			to_plot = []
	return to_plot, depth
	

st.set_page_config(layout="wide")

st.write("Click the top left arrow to begin!")

#st. are streamlit UI elements
st.title("Boknis Eck Data")

#The .tab file contains a large description in front of the table, this splits the file into the description (which ends with */) and the table
with open("BoknisEck_2015-2023.tab", "r", encoding="utf-8") as file:
    content = file.read()
    description = ""
    df_str = ""
    for i in range(len(content)):
    	if content[i] == "*" and content[i+1] == "/":
    		description = content[:i+2]
    		df_str = content[i+2:]
    		break
#The string table is simulated as a file using IO and read as a csv by pandas
df_str_io = io.StringIO(df_str)
df = pd.read_csv(df_str_io, sep="\t")

def df_preprocessing(df, selected_range):
	#Remove unnecessary columns
	df.drop(["Latitude", "Longitude", "Sample label"], axis=1, inplace=True)

	problematic_columns = ["[NO3]- [µmol/l]", "[NO2]- [µmol/l]", "[PO4]3- [µmol/l]"]

	#Replace '<0.01' with 0, convert to float and change flag to 7
	for column in problematic_columns:
		for row_pos, i in enumerate(df[column]):
			if i == "<0.01":
				df.iloc[row_pos, df.columns.get_loc(column)] = 0.00
				df.iloc[row_pos, df.columns.get_loc(column) + 1] = 7
		df[column] = df[column].astype(float)
		
	return df

#Set Date/Time column to index
df['Date/Time'] = pd.to_datetime(df['Date/Time'])
df = df.set_index(['Date/Time', 'Depth water [m]']) 


#Maps UI names to df names
column_map = {
	"Nitrate": "[NO3]- [µmol/l]", 
	"Nitrite": "[NO2]- [µmol/l]",
	"Phosphate": "[PO4]3- [µmol/l]",  
	"Oxygen": "O2 [µmol/l]", 
	"Salinity": "Sal", 
	"Silicate": "Si(OH)4 [µmol/l]", 
	"Temperature": "Temp [°C]"
	}
	
flag_map = {
	"Nitrate": "Flag ((NO3))", 
	"Nitrite": "Flag ((NO2))",
	"Phosphate": "Flag ((PO4))",  
	"Oxygen": "Flag ((Oxygen))", 
	"Salinity": "Flag ((Sal))", 
	"Silicate": "Flag ((SiO2))", 
	"Temperature": "Flag ((Temp))"
	}

st.sidebar.title("Options")

with st.sidebar.expander("See description"):
    st.write(description)

#UI to toggle stats
stats = st.sidebar.checkbox("Toggle for stats")

#UI to toggle inaccuracy colors
show_flags = st.sidebar.checkbox("Toggle for coloring inaccurate data")

#UI to display color legend
if show_flags:
	st.sidebar.write(pd.DataFrame({"Color": ["Yellow", "Red"], "Meaning": ["Inaccurate data", "Missing data"]}))

two_plots = st.sidebar.checkbox("Toggle to graph 2 seperate plots")

scatterplot = st.sidebar.checkbox("Toggle scatterplot instead of timeseries")



range_start = df.index[0][0].to_pydatetime()
range_end = df.index[-1][0].to_pydatetime()

# Range datetime slider
selected_range = st.sidebar.slider(
    "Select time range",
    min_value=range_start,     
    max_value=range_end,    
    value=(range_start, range_end)
)



df = df_preprocessing(df, selected_range)


#UI to display name legend
name_legend_placeholder = st.sidebar.empty()

#UI to choose depth and columns
to_plot1, depth1 = select_variable(1, scatterplot)

if to_plot1:
	download_1_placeholder = st.sidebar.empty()

#Map selection names to column names
variables1 = [column_map[x] for x in to_plot1]
flags1 = [flag_map[x] for x in to_plot1]
columns1 = variables1 + flags1

to_plot2 = []

if two_plots:
	to_plot2, depth2 = select_variable(2, scatterplot)
	
	if to_plot2:
		download_2_placeholder = st.sidebar.empty()
	
	variables2 = [column_map[x] for x in to_plot2]
	flags2 = [flag_map[x] for x in to_plot2]
	columns2 = variables2 + flags2
	


#Make custom name legend for selected columns
legend = pd.DataFrame({'Common': to_plot1, 'Scientific': variables1})

if two_plots:
	legend2 = pd.DataFrame({'Common': to_plot2, 'Scientific': variables2})
	legend = pd.concat([legend, legend2], ignore_index=True)
name_legend_placeholder.write(legend)


#Displays plots and download buttons
if to_plot1:
	plot1 = plot_data(df, variables1, flags1, depth1, show_flags, scatterplot)
	st.plotly_chart(plot1)
	#download_button(plot1, to_plot1, depth1, download_1_placeholder, 1)

if to_plot2 and two_plots:
	plot2 = plot_data(df, variables2, flags2, depth2, show_flags, scatterplot)
	st.plotly_chart(plot2)
	#download_button(plot2, to_plot2, depth2, download_2_placeholder, 2)
