"""
Model used for aggregation and forecasting
Qing Jun2022/12/21

"""
import pandas as pd
import biogeme.database as db
import biogeme.biogeme as bio
from biogeme.expressions import Beta, Variable, log, exp
import biogeme.segmentation as seg
from biogeme import models

df = pd.read_table('data/lpmc14.dat')
# database = db.Database('London_market_share', df)

# Specification ASC
ascWalk = Beta('ascWalk', 0, None, None, 0)
ascCycle = Beta('ascCycle', 0, None, None, 0)
ascPT = Beta('ascPT', 0, None, None, 0)
ascDrive = Beta('ascDrive', 0, None, None, 0)
# Generi Factor
gene_cost = Beta('cost', 0, None, None, 0)
gene_time = Beta('time', 0, None, None, 0)

# basic model
## 1 -> variable for time
dur_pt_rail = Variable('dur_pt_rail')
dur_pt_bus = Variable('dur_pt_bus')
pt_interchanges = Variable('pt_interchanges')
dur_pt_int = Variable('dur_pt_int')
dur_pt_access = Variable('dur_pt_access')
dur_pt = dur_pt_rail + dur_pt_bus + dur_pt_int + dur_pt_access
dur_driving = Variable('dur_driving')
dur_walking = Variable('dur_walking')
dur_cycling = Variable('dur_cycling')

## 2 -> variable for cost
cost_transit = Variable('cost_transit')
cost_driving_fuel = Variable('cost_driving_fuel')
cost_driving_ccharge = Variable('cost_driving_ccharge')
cost_driving = cost_driving_ccharge + cost_driving_fuel
## 3 -> mode choice
choice = Variable('travel_mode')
## For time
walk_time = Beta('walk_time', 0, None, None, 0)
cycle_time = Beta('cycle_time', 0, None, None, 0)
pt_time = Beta('pt_time', 0, None, None, 0)
drive_time = Beta('drive_time', 0, None, None, 0)

# borrowed from model 2
df['segment_age'] = pd.cut(x=df['age'], bins=[0, 20, 60,120],labels=[1,2,3]).astype(float)
segmented_age = Variable('segment_age')
pt_interchanges = Beta('pt_interchanges', 0, None, None, 0)
df.segment_age.astype(float)
segmented_age = seg.DiscreteSegmentationTuple(
    variable=segmented_age, mapping={1: 'young', 2:'worker' ,3: 'elderly'}
)
season_map = {1:0,2:0,3:1,4:1,5:1,6:2,7:2,8:2,9:1,10:1,11:1,12:0}
df['season'] = df.travel_month.map(season_map)
travel_month = Variable('season')
segmented_ascWalk_age = seg.segment_parameter(
    ascWalk, [segmented_age]
)

segmented_ascCycle_age = seg.segment_parameter(
    ascCycle, [segmented_age]
)

segmented_ascPT_age = seg.segment_parameter(
    ascPT, [segmented_age]
)

segmented_ascDrive_age = seg.segment_parameter(
    ascDrive, [segmented_age]
)
## For travel month
walk_month = Beta('walk_travel_month', 0, None, None, 0)
cycle_month = Beta('cycle_travel_month', 0, None, None, 0)
pt_month = Beta('pt_travel_month', 0, None, None, 0)
drive_month = Beta('drive_travel_month', 0, None, None, 0)

# from model 3
lambda_boxcox = Beta('lambda_boxcox', 1, None, None, 0)
boxcox_time_walk = models.boxcox(dur_walking, lambda_boxcox)
boxcox_time_cycle = models.boxcox(dur_cycling, lambda_boxcox)
boxcox_time_pt = models.boxcox(dur_pt, lambda_boxcox)
boxcox_time_drive = models.boxcox(dur_driving, lambda_boxcox)
# from model 4
# New utility functions
OptDrive =  (segmented_ascDrive_age + gene_cost * cost_driving + drive_time * boxcox_time_drive + drive_month * travel_month )
OptWalk =  (segmented_ascWalk_age+ walk_time * boxcox_time_walk + walk_month * travel_month)
OptCycle = (segmented_ascCycle_age + cycle_time * boxcox_time_cycle + cycle_month * travel_month)
OptPt =  (segmented_ascPT_age + cost_transit*0.85 * gene_cost + pt_time * boxcox_time_pt  + pt_month * travel_month)

Free = Beta('Free', 1, 1, None, 0) # the first result is less than one so I apply a lower bound 1 on it
Pay = Beta('Pay', 1, None, None, 0)
free_nest = Free, [1, 2]
cost_nest = Pay, [3, 4]
nests = free_nest, cost_nest

database = db.Database('LPMC', df)
V = {1: OptWalk, 2: OptCycle, 3: OptPt, 4: OptDrive}

# access to propability required for market share
logprob = models.lognested(V, None, nests, choice)
prob_walk = models.nested(V, None, nests, 1)
prob_cycle = models.nested(V, None, nests, 2)
prob_pt = models.nested(V, None, nests, 3)
prob_drive = models.nested(V, None, nests, 4)

# access to the results
# biogeme_nested_same = bio.BIOGEME(database, logprob)
# biogeme_nested_same.modelName = 'nested_free'
# nested_same_results4 = biogeme_nested_same.estimate()


