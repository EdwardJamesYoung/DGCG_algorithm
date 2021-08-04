print("Simulation script started running.")
import argparse
print("Imported argparse.")

parser = argparse.ArgumentParser()
parser.add_argument('index', type = int)

args = parser.parse_args()

index = args.index
print("Job index is: {}.".format(index))

#-----------------------------

import numpy as np
import itertools
from time import localtime, strftime
print("Imported numpy, itertools, and time")

"""
noise_levels = np.array([0,0.3,0.6,1.2,2.4])
trade_off = np.array([0.2,0.5,0.8])
weight = np.array([0.2,0.6,1.2])

algorithm_args = np.array([ [(1-l)*w, l*w, n] for l,w,n in itertools.product(trade_off,weight,noise_levels)] )[index]
"""

noise_levels = np.array([0,0.2])
trade_off = np.array([0.5])
weight = np.array([0.2,0.4])

algorithm_args = np.array([ [(1-l)*w, l*w, n] for l,w,n in itertools.product(trade_off,weight,noise_levels)] )[index]
print("alpha = {:.2f}, beta = {:.2f}, noise level = {:.1f}".format(*algorithm_args))

#-----------------------------

import sys
import os

sys.path.insert(0, os.path.abspath('..'))
from src import DGCG
from light_microscopy_simulations import Gaussian_kernel
print('Imported DGCG and Gaussian_kernel.')

Res = 101
sigma = 0.02
T = 51
TIMESAMPLES = np.linspace(0,1,T)

#-----------------------------

pos1 = np.array([[0.4,0.3],[0.41741160034183417,0.29848646949445945],[0.43110990777977454,0.33162058997828314],[0.44158986394747407,0.3485622874852122],[0.4639284536220064,0.353392214121624],[0.4853099354429353,0.35850862750530477],[0.49798040199802435,0.3647815893495791],[0.5112005084406679,0.3763629575109525],[0.5343657572665693,0.3814163562991913],[0.5233807610271364,0.40465509700254126],[0.5082333808326764,0.3889620774638997],[0.5102791895440385,0.4294830817999204],[0.5045437179297376,0.4396374690023451],[0.5275070732854397,0.4361263853504813],[0.5376402993736471,0.43139176455413886],[0.5398993512322452,0.4532009910683471],[0.547171635046419,0.4658765258557866],[0.5456522788604415,0.47799779580763746],[0.531444793819554,0.4661534025827192],[0.5569477924329503,0.463893095334489],[0.5802402120023171,0.43079004564666123],[0.5689983658229979,0.4396846433695065],[0.5752005463742196,0.43379232474635476],[0.5889903991128006,0.44703795769814636],[0.5803383503120997,0.45621309836535806],[0.5872925443721773,0.46925904327435086],[0.5786280357244945,0.48798753185909965],[0.6142583094307226,0.4965509208848476],[0.6013294279013521,0.5064256794449531],[0.617823085524414,0.5268373074557502],[0.6229813061825761,0.540479626564335],[0.6406750722605731,0.5391039904954871],[0.6446899702747049,0.554540221194774],[0.6261691757601402,0.5766856908224751],[0.6464441597475383,0.5811022076155913],[0.6835283787834258,0.6102865146086283],[0.6659377477879826,0.6518970618542473],[0.6736900738293797,0.6287439435856913],[0.6760934423470193,0.640561905523449],[0.6895061762502793,0.6622123831528475],[0.7155835935817914,0.6767540319829057],[0.715887353821277,0.7094772774976623],[0.7389116373918927,0.7154979850031608],[0.7878800652562237,0.7481107367697666],[0.8100142191095288,0.7802568209003379],[0.8280844364209453,0.8073866004746834],[0.8065594643713198,0.8210449047017374],[0.8052508222745938,0.8423899768195374],[0.8141335806192864,0.8505646757650601],[0.8173115910322919,0.8672789669247982],[0.8,0.9]])
pos2 = np.array([[0.3,0.4],[0.3068436860856618,0.4154760924811113],[0.33508627362383925,0.4373648662452438],[0.3415755054604636,0.4450724898700008],[0.3695470768503329,0.4389730994535724],[0.3756563006767929,0.4800447324521145],[0.3870639186493443,0.4884164957629165],[0.3954754307360413,0.5032821260744973],[0.3909549417755661,0.48959644550992043],[0.3933814188745281,0.48802348775710963],[0.42023707492144124,0.48494392940120623],[0.41385098017712074,0.4969351939596551],[0.4460909467948975,0.5087503336608842],[0.4665130263827096,0.49775119928802136],[0.4934122220413756,0.4882318121676652],[0.5034494298246477,0.47737670769511864],[0.5108670398606444,0.498890602811369],[0.5106481742372224,0.5044456464794111],[0.5146428193514594,0.5061832332622439],[0.526225074385658,0.5023123984229774],[0.5389728072858226,0.49385520165001306],[0.562606368996529,0.4892450830326623],[0.60005703086307,0.51288776408155],[0.5995690222270835,0.5104000794083491],[0.5927996365022894,0.5125414104356806],[0.5991335476359654,0.5591642417096154],[0.610724936937337,0.5648187165976746],[0.6255082256873115,0.5519338171542285],[0.6421255124721051,0.5536805783358065],[0.6661665251440951,0.5597342808948753],[0.6734699307889396,0.554699501376523],[0.6707966197591412,0.5573734221204837],[0.6952002056873179,0.584188260075541],[0.6867375139836097,0.6102359752782546],[0.7054873860990695,0.6277688612654653],[0.7385990628875014,0.6273606672136494],[0.7446737147233583,0.6339952496862239],[0.7628616474604077,0.6662955238154439],[0.764392410299791,0.671786082309432],[0.7665099541402475,0.6648822072487782],[0.7795032452628183,0.6900244575559825],[0.7908901716209044,0.7113637327344455],[0.7996548828513089,0.7387812552655796],[0.8217688303330457,0.7186638483568126],[0.8058428439746551,0.7223690068282418],[0.8441886718290544,0.7460921538736891],[0.8467577250843613,0.7541785803867553],[0.8634105637109087,0.7880789393532246],[0.8893067754211332,0.7739513764692902],[0.9182728950531024,0.7741015093552468],[0.9,0.8]])
pos3 = np.array([[0.35,0.35],[0.3525060711128092,0.351688694215052],[0.3271259462849856,0.32550214660380494],[0.3281235996719551,0.3525876473017342],[0.32280081260525356,0.35730455878335726],[0.35031562040625486,0.380813271537153],[0.376173713187226,0.38517288615286904],[0.3711003883886445,0.3902748341159832],[0.374618576664072,0.40510741433745323],[0.40982226036602803,0.4130719713201563],[0.4448828438933447,0.43365120153373865],[0.45188471115620293,0.42746689949710825],[0.46125824607153887,0.4314193953462363],[0.4743814283098698,0.44477252146721524],[0.483882962486881,0.44951232863424656],[0.48535801247241667,0.4480894298641842],[0.4997481201685451,0.45845448455933524],[0.536645780343017,0.48353533009902255],[0.5501945946697077,0.49953908640103933],[0.5498882321293629,0.51236694239908],[0.5582330788453654,0.5134632583829435],[0.5500487330478957,0.5429871340878604],[0.5515720604960197,0.5383156912197749],[0.5542226611048381,0.5367502839359306],[0.5461234821775276,0.5458578786758769],[0.5696819062557001,0.5421005705217035],[0.5732569886212437,0.5812036252828439],[0.5943120264786828,0.5819194088820625],[0.6257467762117711,0.6056121524954271],[0.6284038491168322,0.6073981794689488],[0.6517366905711489,0.6026615805240864],[0.6684747340510563,0.6016313014987761],[0.6692107114129772,0.5976499762003696],[0.6528471176548423,0.6223242775610002],[0.6625857295233464,0.6347607223056513],[0.6837300056732107,0.6478095521023147],[0.7066860242000079,0.6514219991822668],[0.7113832013917557,0.6651216653174619],[0.7142675020899928,0.6838961677171853],[0.7412147006876512,0.7055749501993588],[0.7397046523839096,0.7222421911145178],[0.7282917634168398,0.7502127271994083],[0.7415053912848002,0.769463147779616],[0.7438894947752699,0.7665108338705576],[0.7744413695687756,0.8063882096051078],[0.7863079628666171,0.8294474409046365],[0.7855134741569545,0.829497218181188],[0.8180819507613193,0.8386773032356418],[0.8241944851006666,0.8213518970869895],[0.827753864697687,0.8269110893082214],[0.85,0.85]])

c1 = DGCG.classes.curve(pos1)
c2 = DGCG.classes.curve(pos2)
c3 = DGCG.classes.curve(pos3)

measure = DGCG.classes.measure()
measure.add(c1, 1*c1.energy())
measure.add(c2, 1*c1.energy())
measure.add(c3, 2*c1.energy())

print("Measure initialised.")

#-----------------------------

if __name__ == "__main__":

    print("Main routine started.")
    kernel = Gaussian_kernel.GaussianKernel(Res,sigma)    
    print("Kernel initialised.")
    DGCG.set_model_parameters(*algorithm_args[:2],TIMESAMPLES, np.ones(T,dtype=int)*(Res**2), kernel.eval, kernel.grad)
    print("Model initilised.")

    #-----------------------------
    
    data = DGCG.operators.K_t_star_full(measure)

    noise = np.random.randn(*data.shape)
    noise = noise/np.sqrt(DGCG.operators.int_time_H_t_product(noise,noise))
    noise = noise*algorithm_args[2]*np.sqrt(DGCG.operators.int_time_H_t_product(data,data))

    noisy_data = data + noise
    print("Noisy data obtained.")

    #-----------------------------

    simulation_parameters = {
        "insertion_max_restarts": 30,
        "insertion_min_restarts": 10,
        "results_folder": "a={:.2f},b={:.2f},n={:.1f},date={}".format(*algorithm_args,strftime("%m%d%H%M",localtime())),
        "multistart_pooling_num": 5000,
        "insertion_min_segments": 15,
        "insertion_max_segments": 40,
        "TOL": 10**(-8)
    }
	
    DGCG.config.time_limit = True
    DGCG.config.multistart_proposition_max_iter = 100000
    DGCG.config.full_max_time = 72000

    print("Solve about to start.")
    solution_measure = DGCG.solve(noisy_data, **simulation_parameters)

    #------------------------------

    recovered_data = DGCG.operators.K_t_star_full(solution_measure)
    diff = recovered_data - data

    er = DGCG.operators.int_time_H_t_product(diff,diff)
    print("Data error of the recovered solution:")
    print(er)