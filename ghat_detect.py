#import MySQLdb
#from functools import partial
#from operator import mul
from datetime import datetime
from haversine import haversine
import googlemaps
from googlemaps import convert
import csv
import pandas as pd
import numpy as np
#import routeplot as rp
import matplotlib.pyplot as plt
from itertools import groupby
from matplotlib.pyplot import *
import wayplot as wp
import warnings
import math
warnings.simplefilter("ignore")
starttime = datetime.now()
wmap = wp.WayMap(edge_width=5)#868325025818719
#####################################################################################################
data = pd.read_csv("20_11_2017_Mon/result_20/mysur_ooty/mysur_ooty.csv")#will be use to find elevation#lonavala_route.csv
##################################################################################################################################
def bearingAngle(lat1, lng1, lat2, lng2):
	lat1, lng1, lat2, lng2 = map(np.radians, [lat1, lng1, lat2, lng2])
	dlng = lng2 - lng1

	X = np.sin(dlng)*np.cos(lat2)
	Y = np.cos(lat1)*np.sin(lat2) - np.sin(lat1)*np.cos(lat2)*np.cos(dlng)

	bearing_radian = np.arctan2(X,Y)
	bearing_degree = (bearing_radian *180.0)/np.pi

	return bearing_degree
##################################################################################################################################
data =pd.concat((data, data[["latitude", "longitude","elevation"]].shift(periods=1)\
	.rename(columns={"latitude":"lat_dup", "longitude":"long_dup","elevation":"elev_dup"})), axis=1)
data["distance"]  = data.apply(lambda x:haversine((x["latitude"],x["longitude"]),(x["lat_dup"],x["long_dup"])), axis=1)
data["abs_elev"]  = data.apply(lambda x:abs(x["elevation"] - x["elev_dup"]),axis=1)
data.loc[0,["distance","abs_elev"]] = 0
data["roll_sum"] = data["distance"].rolling(10).sum()# rolling distance
data["cum_dist"] = data["distance"].cumsum()# cumulative distance
##################################################################################################################################
data["Bearing"] = data.apply(lambda x: bearingAngle(x["lat_dup"],x["long_dup"],x["latitude"],x["longitude"]), axis=1)
data.loc[0,["Bearing"]] = 0# Nan at 0 index replaced by 0
data = pd.concat((data,data[["Bearing"]].shift(periods=-1).rename(columns={"Bearing":"Bear_ang"})), axis=1)#shifting column,which start from 2row
data["turn_angle"] = data.apply(lambda x: abs(x["Bearing"] - x["Bear_ang"]), axis=1)
data.turn_angle = data.turn_angle.round(3)
data = pd.concat((data,data[["turn_angle"]].shift(periods=-1).rename(columns={"turn_angle":"t_angle"})),axis =1)
data["t_angle"] = data["t_angle"].fillna(int(0))#turn signal for spotting the turn before some distance 
#################################################################################################################################  
data["low_turn"] =  pd.DataFrame(data[(data.t_angle>=150)&(data.t_angle<270)].t_angle)#turning angle threshould
data["C_turn"]   = pd.DataFrame(data[(data.t_angle>=270)&(data.t_angle<360)].t_angle)
ind_turn_1   = pd.DataFrame(data[(data.t_angle>=150)&(data.t_angle<270)].t_angle).index.tolist()
ind_turn_2   = pd.DataFrame(data[(data.t_angle>=270)&(data.t_angle<=360)].t_angle).index.tolist()#index of the turning place
data_turn_1 = data.ix[ind_turn_1].reset_index(drop=True)#dataframe for column low_turn
data_turn_2 = data.ix[ind_turn_2].reset_index(drop=True)#dataframe column for C_turn
#print(data[["Bearing","Bear_ang","t_angle"]])####################################################################################
print("numer of turn points :",len(ind_turn_2)-5) 
#################################################################################################################################
data[["latitude","longitude","elevation","elev_dup","abs_elev","distance","cum_dist","Bear_ang","t_angle","low_turn","C_turn"]]\
.to_csv("20_11_2017_Mon/result_20/mysur_ooty/lat_long_ele_my_oty.csv")#lat_long_ele_otych
##################################################################################################################################
#pre route alert detection
try:
	def ghat_alert(t_dist,abs_change_ele):
		#t_dist = 500/1000#target distance
		I_dist = t_dist#alert generate at this distance
		delta  = 0.4;ghat_ind=[]
		I_elev = data["elevation"][0]
		for i in range(len(data["cum_dist"])):
			if (data["cum_dist"][i] <=I_dist + delta)and(data["cum_dist"][i] >I_dist - delta):
				print(data["cum_dist"][i])
				change_elev = abs(I_elev - data["elevation"][i])
				if change_elev >=abs_change_ele:# threshold value for change in elevation
					alert ="ghat is there after reaching at : "+str(round(data["cum_dist"][i],2))+" "+"KM"
					I_elev = data["elevation"][i]#updated initial value by second condition
					ghat_ind.append(i)
					print(alert)
				I_dist = I_dist + t_dist
		#print(ghat_ind)
		return alert,ghat_ind# error term expected
	dist_thr  =  input("Enter the distacne threshold :")# user specifying value for distance
	elev_thr  = input("Enter the elevation threshold :");print("\n")# user specifying value for elevation
	ghat_info = ghat_alert(float(dist_thr),float(elev_thr))#calling ghat alert function(target_distance(5KM), elevation_change(50))
	#print(ghat_info[1])# list of index of alert when ghat condition satisfying
	data_ghat = data.ix[ghat_info[1]].reset_index(drop = True)#taking the index of ghat alert message
except UnboundLocalError:
	print("Wrong Input") 
####################################################################################################################################
df = pd.read_csv("20_11_2017_Mon/result_20/mysur_ooty/lat_long_ele_my_oty.csv")#will be use to find cum_dist,elev
print("\n");print(df["elevation"].describe())#summary of elevation column
print("range :",(df["elevation"].max() - df["elevation"].min()))
df["ele_min"] = pd.DataFrame(df[(df.elevation>=df["elevation"].describe()[3]) & (df.elevation< df["elevation"].describe()[4])].elevation)
df["ele_q1"] = pd.DataFrame(df[(df.elevation>=df["elevation"].describe()[4]) & (df.elevation< df["elevation"].describe()[5])].elevation)
df["ele_q2"] = pd.DataFrame(df[(df.elevation>=df["elevation"].describe()[5]) & (df.elevation< df["elevation"].describe()[6])].elevation)
df["ele_q3"] = pd.DataFrame(df[(df.elevation>=df["elevation"].describe()[6]) & (df.elevation<= df["elevation"].describe()[7])].elevation)
########################################################################################################################################
ind_min = pd.DataFrame(df[(df.elevation>=df["elevation"].describe()[3]) & (df.elevation< df["elevation"].describe()[4])].elevation).index.tolist()
ind_q1 =  pd.DataFrame(df[(df.elevation>=df["elevation"].describe()[4]) & (df.elevation< df["elevation"].describe()[5])].elevation).index.tolist()
ind_q2 = pd.DataFrame(df[(df.elevation>=df["elevation"].describe()[5]) & (df.elevation< df["elevation"].describe()[6])].elevation).index.tolist()
ind_q3 = pd.DataFrame(df[(df.elevation>=df["elevation"].describe()[6]) & (df.elevation<= df["elevation"].describe()[7])].elevation).index.tolist()
########################################################################################################################################
df["q3_dumy"] = df["ele_q3"].fillna(int(0))
df["q3_dumy"] = np.where(df["ele_q3"]>0,1, 0)# in dumpy column the value will get change by 1 if ele_q3>0 otherwise 0
df["binary_e"] = np.where(df["abs_elev"]>=1,1, 0)
df["flag"] = np.where(df["elev_dup"] - df["elevation"]>0,0,1)#ture condition->0, false condtion->1
df.loc[0,["flag"]] = 0
#grouping the repeated address
def max_ele_region(data_list):
	max_ele = [(len(list(j)), i) for i, j in groupby(data_list)]
	df_max_ele =pd.DataFrame({"count":[a[0] for a in max_ele],"add":[a[1] for a in max_ele]})
	df_max_ele["cum_sum"] = df_max_ele["count"].cumsum()
	df_max_ele["start"] = df_max_ele["cum_sum"] - df_max_ele["count"]
	df_max_ele["end"]   = df_max_ele["cum_sum"] - 1
	start_ifix = df_max_ele["start"].tolist()#list of start index
	df_max_ele = df_max_ele.ix[(df_max_ele["end"] - df_max_ele["start"])>0].reset_index(drop = True)
	start_i = df_max_ele["start"].tolist()#list of start of repeated term
	end_i = df_max_ele["end"].tolist()#list of end index of repeated term
	return df_max_ele
df_max_ele = max_ele_region(df["q3_dumy"].tolist())
df_max_ele = df_max_ele.loc[(df_max_ele["add"]==1)].nlargest(2,"count").reset_index(drop =True)#max elevated region dataframe
#print(df_max_ele)# will give the first max and second max element of cout column
########################################################################################################################################
df_ele_q3 = df.loc[(df.index>=df_max_ele["start"][0]) & (df.index<=df_max_ele["end"][0])]
df_ele_q2 = df.loc[(df.index>=df_max_ele["start"][1]) & (df.index<=df_max_ele["end"][1])]#this can not occur sometime in picture
#print(df_ele_q3[["latitude","longitude","elevation","Bearing"]],"\n")
########################################################################################################################################
df_ups  = df.loc[(df["flag"]==1)].reset_index(drop =True)#dataframe for all flag == 1
df_down = df.loc[(df["flag"]==0)].reset_index(drop =True)#dataframe for all flag == 0
########################################################################################################################################
df.plot(x= "cum_dist",y =["t_angle","low_turn","C_turn"],kind ="line",title='turn angle vs Distance traveled',\
							style =["-*y","-*r","-*b"])
plt.xlabel("Distance(KM)")
plt.ylabel("turns(abs(value))")
plt.grid(True)
plt.show()
########################################################################################################################################
df.plot(x= "cum_dist",y ="flag", kind ="line",title='ups_down vs Distance traveled',style ="-b*")
plt.xlabel("Distance(KM)")
plt.ylabel("Ups_Down")
plt.grid(True)
plt.show()
########################################################################################################################################
fig,ax = subplots()
df.plot(x="cum_dist",y =["elevation","ele_min","ele_q1","ele_q2","ele_q3"], kind ="line",title='Elevation vs Distance traveled',\
						 style =["-k","-r","-y","-g","-b"],ax = ax)#
box = ax.get_position()#to get the position for box of subplot
ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
ax.legend(np.round([df["elevation"].describe()[3],df["elevation"].describe()[4],df["elevation"].describe()[5],df["elevation"].describe()[6],\
		df["elevation"].describe()[7]],2),loc='center left',bbox_to_anchor=(1,0.7)) # for changing the legend values
ax.xaxis.grid(linestyle='-', linewidth=0.50,color ='k')
ax.yaxis.grid(linestyle='-', linewidth=0.50,color ='r')
#########################################################################################################################################
plt.xlabel("Distance(KM)")
plt.ylabel("Elevation(M)")
plt.show()
#########################################################################################################################################
wmap.plot_waypoints_route(df[["latitude","longitude","elevation","cum_dist","Bear_ang"]], plot_type="plot", color="#5F9EA0")#data route plot
wmap.plot_route(data_turn_1[["latitude","longitude","cum_dist","Bear_ang","t_angle"]],plot_type="scatter", color="y")#turn angle
wmap.plot_route(data_turn_2[["latitude","longitude","cum_dist","Bear_ang","t_angle"]],plot_type="scatter", color="g")
wmap.plot_route(data_ghat[["latitude","longitude","elevation","cum_dist"]],plot_type="scatter", color="k")##00FF7F
wmap.plot_route(df_down[["latitude","longitude","elevation","cum_dist","Bear_ang","flag"]],plot_type="scatter", color="r")#for down elev
#wmap.plot_route(df_ups[["latitude","longitude","elevation","cum_dist","Bear_ang","flag"]],plot_type="scatter", color="b")#for down elev
#########################################################################################################################################
# wmap.plot_route(df[["latitude","longitude","elevation","cum_dist","Bear_ang","abs_elev","binary_e"]].ix[ind_min].reset_index(drop=True),plot_type="scatter", color="r")
# wmap.plot_route(df[["latitude","longitude","elevation","cum_dist","Bear_ang","abs_elev","binary_e"]].ix[ind_q1].reset_index(drop=True),plot_type="scatter", color="g")
# wmap.plot_route(df[["latitude","longitude","elevation","cum_dist","Bear_ang","abs_elev","binary_e"]].ix[ind_q2].reset_index(drop=True),plot_type="scatter", color="g")
# wmap.plot_route(df[["latitude","longitude","elevation","cum_dist","Bear_ang","abs_elev","binary_e"]].ix[ind_q3].reset_index(drop=True),plot_type="scatter", color="b")
# wmap.plot_route(df_ele_q3[["latitude","longitude","elevation","cum_dist","Bear_ang","abs_elev","binary_e"]].reset_index(drop=True),plot_type="scatter", color="b")
#wmap.plot_route(df_ele_q2[["latitude","longitude","elevation","cum_dist","Bearing"]].reset_index(drop=True),plot_type="scatter", color="g")
wmap.draw("20_11_2017_Mon/ups_down_my_ooty.htl")
#########################################################################################################################################
#"20_11_2017_Mon/result_20/lonavala/lat_long_ele_lon.csv"
#"20_11_2017_Mon/result_20/ooty_ch/lat_long_ele_otych.csv"
#"20_11_2017_Mon/result_20/mysur_ooty/lat_long_ele_my_oty.csv"
#https://www.thedash.com/dashboard/S1L5OAkzf # mydashboard 