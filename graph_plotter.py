import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io


#Plots selected data.
def plot_data(df: pd.DataFrame,
		vars: list[str],
		flags: list[str],
		show_flags: bool = False,
		pltaspect: float = 0.3,
		ax: plt.Axes | None = None) -> plt.Figure:
	"""
	Plot each variable in `vars` (with markers),
	and if show_flags=True overlay:
	- red dots where flag == 9 (missing)
	- yellow dots where flag == 7 (inaccurate)
	"""
	if ax is None:
		fig, ax = plt.subplots()
	else:
		fig = ax.figure

	# plot the variables
	df[vars].plot(ax=ax, marker='o', linestyle='-')

	if show_flags:
		ymin, ymax = ax.get_ylim()
		# for each var/flag pair
		for var_col, flag_col in zip(vars, flags):
			# find indices
			idx_missing = df.index[df[flag_col] == 9]
			idx_inacc   = df.index[df[flag_col] == 7]

			# scatter red for missing, yellow for inaccurate
			ax.scatter(idx_missing, [ymin]*len(idx_missing),
				color='red', label='_nolegend_', zorder=5)
			ax.scatter(idx_inacc, [ymin]*len(idx_inacc),
				color='orange', label='_nolegend_', zorder=5)

		ax.set_ylim(ymin, ymax)
	ax.set_box_aspect(pltaspect)
	return fig


#Sets download button
def download_button(plot, element, key):
	#buffer for Download Button
	buf = io.BytesIO()
	plot.savefig(buf, format="png", dpi=300, bbox_inches="tight")
	buf.seek(0)  # rewind to beginning of buffer

	#Download button
	element.download_button(
		label=f"📥 Download plot ↑",
		data=buf,
		file_name="Boknis-Eck.png",
		mime="image/png",
		key=key
	)


st.set_page_config(layout="wide")

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

#Remove unnecessary columns
df.drop(["Latitude", "Longitude", "Sample label"], axis=1, inplace=True)

#Set Date/Time column to index
df['Date/Time'] = pd.to_datetime(df['Date/Time'])
df = df.set_index('Date/Time')


problematic_columns = ["[NO3]- [µmol/l]", "[NO2]- [µmol/l]", "[PO4]3- [µmol/l]"]

#Replace '<0.01' with 0, convert to float and change flag to 7
for column in problematic_columns:
	for row_pos, i in enumerate(df[column]):
		if i == "<0.01":
			df.iloc[row_pos, df.columns.get_loc(column)] = 0.00
			df.iloc[row_pos, df.columns.get_loc(column) + 1] = 7
	df[column] = df[column].astype(float)
 


#The dataframe has 6 different depth measurements for every point in time. To have a df that can  plotted as a time series they have to be split
#Split df into depth specific dfs
df_1 = df[0::6]
df_5 = df[1::6]
df_10 = df[2::6]
df_15 = df[3::6]
df_20 = df[4::6]
df_25 = df[5::6]


#Maps UI names to code names
depth_map = {
	"1": df_1,
	"5": df_5,
	"10": df_10,
	"15": df_15,
	"20": df_20,
	"25": df_25
	}

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
inaccuracy_colors = st.sidebar.checkbox("Toggle for coloring inaccurate data")

#UI to display color legend
if inaccuracy_colors:
	st.sidebar.write(pd.DataFrame({"Color": ["Yellow", "Red"], "Meaning": ["Inaccurate data", "Missing data"]}))

two_plots = st.sidebar.checkbox("Toggle to graph 2 seperate plots")


pltaspect = st.sidebar.slider(
    label="Pick an aspect ratio",
    min_value=0.01,
    max_value=1.0,
    value=0.3,
    step=0.01,
    help="Default: 0.3"
)


#UI to display name legend
name_legend_placeholder = st.sidebar.empty()

#UI to choose depth and columns
depth1 = st.sidebar.selectbox("Choose depth in meter for plot 1", ["1", "5", "10", "15", "20", "25"])
to_plot1 = st.sidebar.multiselect("Choose vars for plot 1", ["Nitrate", "Nitrite", "Oxygen", "Phosphate", "Salinity", "Silicate", "Temperature"])

if to_plot1:
	download_1_placeholder = st.sidebar.empty()

#Map selection names to column names
columns1 = [column_map[x] for x in to_plot1]
flags1 = [flag_map[x] for x in to_plot1]

current_df1 = depth_map[depth1][columns1]

to_plot2 = []

if two_plots:
	depth2 = st.sidebar.selectbox("Choose depth in meter for plot 2", ["1", "5", "10", "15", "20", "25"])
	to_plot2 = st.sidebar.multiselect("Choose vars for plot 2", ["Nitrate", "Nitrite", "Oxygen", "Phosphate", "Salinity", "Silicate", "Temperature"])
	
	if to_plot2:
		download_2_placeholder = st.sidebar.empty()
	
	columns2 = [column_map[x] for x in to_plot2]
	flags2 = [flag_map[x] for x in to_plot2]
	current_df2 = depth_map[depth2][columns2]


#Make custom name legend for selected columns
legend = pd.DataFrame({'Common': to_plot1, 'Scientific': columns1})
if two_plots:
	legend2 = pd.DataFrame({'Common': to_plot2, 'Scientific': columns2})
	legend = pd.concat([legend, legend2], ignore_index=True)
name_legend_placeholder.write(legend)

#describe()
#st.write(depth_map[depth1][columns1].describe())



if to_plot1:
	plot1 = plot_data(current_df1, columns1, flags1, stats, pltaspect)
	st.pyplot(plot1)
	download_button(plot1, download_1_placeholder, 1)

if to_plot2 and two_plots:
	plot2 = plot_data(current_df2, columns2, flags2, stats, pltaspect)
	st.pyplot(plot2)
	download_button(plot2, download_2_placeholder, 2)
