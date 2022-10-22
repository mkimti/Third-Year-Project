import pandas as pd
from pulp import *

#Load the data

PATH = 'Fantasy-Premier-League/data'

current_df = pd.read_csv(PATH + '/2021-22/gws/merged_gw.csv')
current_df.tail()

gw_5 = current_df[current_df.GW == 5]
gw_5.head()

data = gw_5[['name', 'team', 'position', 'total_points', 'value']]
data.head()

#Helper variables
POS = data.position.unique()
CLUBS = data.team.unique()
BUDGET = 1000
pos_available = {
    'DEF' : 5,
    'FWD' : 3,
    'MID' : 5,
    'GK' : 2,
}

#Initialise Variables
names = [data.name[i] for i in data.index]
teams = [data.team[i] for i in data.index]
positions = [data.position[i] for i in data.index]
prices = [data.value[i] for i in data.index]
points = [data.total_points[i] for i in data.index]
players = [LpVariable("player_" + str(i), cat="Binary") for i in data.index]

#Initialise the problem
prob = LpProblem("FPL Player Choices", LpMaximize)

#Define the objective 
prob += lpSum(players[i] * points[i] for i in range(len(data)))

#Build the constraints
#Budget Limit
prob += lpSum([players[i]] * data.value[data.index[i]] for i in range(len(data))) <= BUDGET

# Position Limit
for pos in POS:
  prob += lpSum(players[i] for i in range(len(data)) if positions[i] == pos) <= pos_available[pos]

# Club Limit
for club in CLUBS:
  prob += lpSum(players[i] for i in range(len(data)) if teams[i] == club) <= 3

# Solve the problem
prob.solve()

for v in prob.variables():
  if v.varValue != 0:
    name = data.name[int(v.name.split("_")[1])]
    club = data.team[int(v.name.split("_")[1])]
    position = data.position[int(v.name.split("_")[1])]
    point = data.total_points[int(v.name.split("_")[1])]
    price = data.value[int(v.name.split("_")[1])]
    print(name, position, club, point, price, sep=" | ")