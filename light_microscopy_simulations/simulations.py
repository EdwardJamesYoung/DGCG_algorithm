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

noise_levels = np.array([0])
trade_off = np.array([0.5])
weight = np.array([0.04,0.08,0.12,0.16,0.20,0.24,0.28,0.32,0.36,0.40])

algorithm_args = np.array([ [(1-l)*w, l*w, n] for l,w,n in itertools.product(trade_off,weight,noise_levels)] )[index-1]


noise_levels = np.array([0])
alp = np.array([0.2])
bet = np.array([0.02])
algorithm_args = np.array([ [a,b,n] for a,b,n in itertools.product(alp,bet,noise_levels)])[index - 1]

print("alpha = {:.2f}, beta = {:.2f}, noise level = {:.1f}".format(*algorithm_args))
"""

segments = np.array([10,15,20])
restarts = np.array([50,100,500,1000])
pooling = np.array([500,1000])
algorithm_args = np.array([ [s,r,p] for s,r,p in itertools.product(segments,restarts,pooling)])[index - 1]

print("Maximum number of segments = {}, maximum restarts = {}, pooling number = {}".format(*algorithm_args))

#-----------------------------

import sys
import os

sys.path.insert(0, os.path.abspath('..'))
from src import DGCG
from light_microscopy_simulations import Gaussian_kernel
print('Imported DGCG and Gaussian_kernel.')

Res = 121
sigma = 0.015
T = 51
TIMESAMPLES = np.linspace(0,1,T)

#-----------------------------

#pos1 = np.array([[0.4,0.3],[0.41741160034183417,0.29848646949445945],[0.43110990777977454,0.33162058997828314],[0.44158986394747407,0.3485622874852122],[0.4639284536220064,0.353392214121624],[0.4853099354429353,0.35850862750530477],[0.49798040199802435,0.3647815893495791],[0.5112005084406679,0.3763629575109525],[0.5343657572665693,0.3814163562991913],[0.5233807610271364,0.40465509700254126],[0.5082333808326764,0.3889620774638997],[0.5102791895440385,0.4294830817999204],[0.5045437179297376,0.4396374690023451],[0.5275070732854397,0.4361263853504813],[0.5376402993736471,0.43139176455413886],[0.5398993512322452,0.4532009910683471],[0.547171635046419,0.4658765258557866],[0.5456522788604415,0.47799779580763746],[0.531444793819554,0.4661534025827192],[0.5569477924329503,0.463893095334489],[0.5802402120023171,0.43079004564666123],[0.5689983658229979,0.4396846433695065],[0.5752005463742196,0.43379232474635476],[0.5889903991128006,0.44703795769814636],[0.5803383503120997,0.45621309836535806],[0.5872925443721773,0.46925904327435086],[0.5786280357244945,0.48798753185909965],[0.6142583094307226,0.4965509208848476],[0.6013294279013521,0.5064256794449531],[0.617823085524414,0.5268373074557502],[0.6229813061825761,0.540479626564335],[0.6406750722605731,0.5391039904954871],[0.6446899702747049,0.554540221194774],[0.6261691757601402,0.5766856908224751],[0.6464441597475383,0.5811022076155913],[0.6835283787834258,0.6102865146086283],[0.6659377477879826,0.6518970618542473],[0.6736900738293797,0.6287439435856913],[0.6760934423470193,0.640561905523449],[0.6895061762502793,0.6622123831528475],[0.7155835935817914,0.6767540319829057],[0.715887353821277,0.7094772774976623],[0.7389116373918927,0.7154979850031608],[0.7878800652562237,0.7481107367697666],[0.8100142191095288,0.7802568209003379],[0.8280844364209453,0.8073866004746834],[0.8065594643713198,0.8210449047017374],[0.8052508222745938,0.8423899768195374],[0.8141335806192864,0.8505646757650601],[0.8173115910322919,0.8672789669247982],[0.8,0.9]])
#pos2 = np.array([[0.3,0.4],[0.3068436860856618,0.4154760924811113],[0.33508627362383925,0.4373648662452438],[0.3415755054604636,0.4450724898700008],[0.3695470768503329,0.4389730994535724],[0.3756563006767929,0.4800447324521145],[0.3870639186493443,0.4884164957629165],[0.3954754307360413,0.5032821260744973],[0.3909549417755661,0.48959644550992043],[0.3933814188745281,0.48802348775710963],[0.42023707492144124,0.48494392940120623],[0.41385098017712074,0.4969351939596551],[0.4460909467948975,0.5087503336608842],[0.4665130263827096,0.49775119928802136],[0.4934122220413756,0.4882318121676652],[0.5034494298246477,0.47737670769511864],[0.5108670398606444,0.498890602811369],[0.5106481742372224,0.5044456464794111],[0.5146428193514594,0.5061832332622439],[0.526225074385658,0.5023123984229774],[0.5389728072858226,0.49385520165001306],[0.562606368996529,0.4892450830326623],[0.60005703086307,0.51288776408155],[0.5995690222270835,0.5104000794083491],[0.5927996365022894,0.5125414104356806],[0.5991335476359654,0.5591642417096154],[0.610724936937337,0.5648187165976746],[0.6255082256873115,0.5519338171542285],[0.6421255124721051,0.5536805783358065],[0.6661665251440951,0.5597342808948753],[0.6734699307889396,0.554699501376523],[0.6707966197591412,0.5573734221204837],[0.6952002056873179,0.584188260075541],[0.6867375139836097,0.6102359752782546],[0.7054873860990695,0.6277688612654653],[0.7385990628875014,0.6273606672136494],[0.7446737147233583,0.6339952496862239],[0.7628616474604077,0.6662955238154439],[0.764392410299791,0.671786082309432],[0.7665099541402475,0.6648822072487782],[0.7795032452628183,0.6900244575559825],[0.7908901716209044,0.7113637327344455],[0.7996548828513089,0.7387812552655796],[0.8217688303330457,0.7186638483568126],[0.8058428439746551,0.7223690068282418],[0.8441886718290544,0.7460921538736891],[0.8467577250843613,0.7541785803867553],[0.8634105637109087,0.7880789393532246],[0.8893067754211332,0.7739513764692902],[0.9182728950531024,0.7741015093552468],[0.9,0.8]])
#pos3 = np.array([[0.35,0.35],[0.3525060711128092,0.351688694215052],[0.3271259462849856,0.32550214660380494],[0.3281235996719551,0.3525876473017342],[0.32280081260525356,0.35730455878335726],[0.35031562040625486,0.380813271537153],[0.376173713187226,0.38517288615286904],[0.3711003883886445,0.3902748341159832],[0.374618576664072,0.40510741433745323],[0.40982226036602803,0.4130719713201563],[0.4448828438933447,0.43365120153373865],[0.45188471115620293,0.42746689949710825],[0.46125824607153887,0.4314193953462363],[0.4743814283098698,0.44477252146721524],[0.483882962486881,0.44951232863424656],[0.48535801247241667,0.4480894298641842],[0.4997481201685451,0.45845448455933524],[0.536645780343017,0.48353533009902255],[0.5501945946697077,0.49953908640103933],[0.5498882321293629,0.51236694239908],[0.5582330788453654,0.5134632583829435],[0.5500487330478957,0.5429871340878604],[0.5515720604960197,0.5383156912197749],[0.5542226611048381,0.5367502839359306],[0.5461234821775276,0.5458578786758769],[0.5696819062557001,0.5421005705217035],[0.5732569886212437,0.5812036252828439],[0.5943120264786828,0.5819194088820625],[0.6257467762117711,0.6056121524954271],[0.6284038491168322,0.6073981794689488],[0.6517366905711489,0.6026615805240864],[0.6684747340510563,0.6016313014987761],[0.6692107114129772,0.5976499762003696],[0.6528471176548423,0.6223242775610002],[0.6625857295233464,0.6347607223056513],[0.6837300056732107,0.6478095521023147],[0.7066860242000079,0.6514219991822668],[0.7113832013917557,0.6651216653174619],[0.7142675020899928,0.6838961677171853],[0.7412147006876512,0.7055749501993588],[0.7397046523839096,0.7222421911145178],[0.7282917634168398,0.7502127271994083],[0.7415053912848002,0.769463147779616],[0.7438894947752699,0.7665108338705576],[0.7744413695687756,0.8063882096051078],[0.7863079628666171,0.8294474409046365],[0.7855134741569545,0.829497218181188],[0.8180819507613193,0.8386773032356418],[0.8241944851006666,0.8213518970869895],[0.827753864697687,0.8269110893082214],[0.85,0.85]])
#pos1 = np.array([[0.1,0.3],[0.19498806371654082,0.36165788082702555],[0.27027863466485375,0.4079867321176298],[0.3308591016349939,0.44494664929948435],[0.3901640690597412,0.45684829140911154],[0.4251964169579203,0.46827786405006977],[0.4512561140045491,0.48691334762302635],[0.4626505100919798,0.49944334761312553],[0.49195377486009406,0.5330195948454206],[0.519954751667586,0.5450006164776255],[0.5278446272652907,0.5684737958159929],[0.5667514900308419,0.5826982399935895],[0.6016259004550352,0.5848865142231479],[0.6164738454221692,0.5811211310481915],[0.6300335766777887,0.5714744596869383],[0.6494974251069627,0.5655715918695474],[0.6561816073834654,0.5591283009849564],[0.6476257190630612,0.5631476361834364],[0.6395802227097087,0.5584029907620067],[0.6291139121038787,0.5787634962303208],[0.6237256106722551,0.5981226199186319],[0.6158674859532978,0.597775099392776],[0.650216372533812,0.5925955734502217],[0.6701825137192814,0.5927222420063567],[0.6885449497817792,0.5884551681389614],[0.7164210505893558,0.582342966327355],[0.7349848231577094,0.5776198442983045],[0.7377999681616995,0.5766257635113226],[0.7375231933657247,0.5969013200209168],[0.7173313689310294,0.6217839236690624],[0.6929447818231127,0.6357314985787594],[0.7042495610639848,0.6354828428994248],[0.7044378887040956,0.6200745587379274],[0.6846999728301517,0.6216437633530494],[0.6833318128082887,0.6057137899845847],[0.6812255611218094,0.6162727171568626],[0.6844991906728038,0.6311348689493735],[0.6841657158405834,0.644015276972946],[0.6529381423305397,0.6403981583516407],[0.6339394801342415,0.6416374185806757],[0.6163029442259887,0.6434608869018019],[0.6141844277374149,0.6612252525703222],[0.6423357556567489,0.6792557308538307],[0.665954018385173,0.7049128613042421],[0.6914201394045567,0.7459006953162738],[0.702751608984513,0.7663502475114559],[0.6993974395497163,0.7989934382024538],[0.6916884781918363,0.8096629766659399],[0.6973409967454117,0.8327378462743082],[0.6922667318033184,0.8710653479424852],[0.7,0.9]])
#pos2 = np.array([[0.3,0.1],[0.28245934717396604,0.13853076075762144],[0.27789833570306227,0.17930096990583289],[0.2756445609614493,0.20923017095827645],[0.2744573935487779,0.22130893905121576],[0.2856117615147099,0.22633990219432704],[0.304583105137406,0.2409713303563611],[0.3210336326599693,0.25396743543538614],[0.3403857195378724,0.28038318851260624],[0.3577497350206555,0.30442765801622534],[0.3802956683793981,0.3251275856121849],[0.4097161865675381,0.34221157033359995],[0.4283151601598099,0.35944504566055985],[0.4375456726640948,0.38958563624987264],[0.4448348292596909,0.3934316033731666],[0.44742065131131886,0.4001079726041382],[0.4664948934522115,0.39463531817675995],[0.4766477891639584,0.3997698255065175],[0.4861745454973577,0.4129775560496086],[0.4869450167406758,0.44652871844029357],[0.49856658593763675,0.4764131048872452],[0.5174156696241776,0.5121240465024303],[0.5257729363571135,0.5327364101220696],[0.5376415207686339,0.5577655397547989],[0.5530730271538269,0.5920726475889094],[0.5603597977640847,0.6268091526153702],[0.5469186523490329,0.6632953573009152],[0.5389570979181226,0.6798706187494488],[0.5417541097760519,0.7153506863435302],[0.5648659269960381,0.7425644674601353],[0.5776150061734329,0.7638982798332824],[0.598925512197389,0.7741391751429189],[0.6075607375775121,0.7874354518938967],[0.6251573100608885,0.8119380859246697],[0.6526995580141609,0.8380531307895972],[0.6846921196331635,0.8608072649381626],[0.7087187129730738,0.8839318709321192],[0.7308517606923374,0.9024740094160648],[0.7463028142689532,0.9138398098956466],[0.7686727352281593,0.914037051751005],[0.7827302498928477,0.9177122652939036],[0.7897110749759045,0.9140464586348789],[0.7933370869729712,0.9040912407758933],[0.7967856769763215,0.9090023545033636],[0.8085386920154998,0.9146670238569046],[0.8227698502410123,0.903614639135882],[0.8401156476655328,0.8928760700927906],[0.8543358210954705,0.8839478884182622],[0.8706665450166846,0.872589584835467],[0.8876188691006783,0.8853205415896107],[0.9,0.9]])
#pos3 = np.array([[0.1,0.1],[0.10453484835635556,0.12714668483209365],[0.11238339336097637,0.15043135873310842],[0.11686269864877454,0.1523763541147924],[0.12233934719767228,0.15747570303925087],[0.12009603903561376,0.16384674901492025],[0.12203384677604374,0.17672414391547575],[0.12706592180877807,0.20430243322515831],[0.13500889296681856,0.21749557218321136],[0.13690836212461793,0.23052270156165394],[0.14494406297636558,0.24081869074157863],[0.15491850997614487,0.25958955791519617],[0.1719321444708357,0.2664658649636272],[0.17652497095345276,0.2722059332165359],[0.18775455136733088,0.27278470917023534],[0.2144404442647657,0.27667963603773227],[0.23591725264000166,0.29245859424319764],[0.2521185128746367,0.31337864786605546],[0.2619811235645668,0.33155195437849944],[0.28567790021348777,0.345454010799646],[0.30224780698548825,0.34944209622702527],[0.32110357126107564,0.3752362475828426],[0.3334582082005542,0.38803289135960306],[0.33413216986232513,0.41109274108314375],[0.34801737982996683,0.43700277733548354],[0.37533290356083715,0.47175255375680153],[0.4024741073163718,0.5118750915246848],[0.4229822665708081,0.5484049792486456],[0.4479902130853207,0.5759114096081],[0.47454128832378717,0.6017160944710677],[0.48839638786177836,0.6167605043968979],[0.5060799146622363,0.6146253759034198],[0.5217359199575229,0.6181809964825664],[0.5423194465265033,0.6161103978100588],[0.5584897132449372,0.598516044677494],[0.5724108705365861,0.5858649766459969],[0.5945536930130477,0.5775564446078163],[0.6198382282861581,0.5786337097066241],[0.6476718745643932,0.580759310458449],[0.6747480693539163,0.5735966269225287],[0.6884840861791921,0.5752349037376481],[0.7099707222046693,0.5829198086447139],[0.7308168765841763,0.5984063022905581],[0.7543049765644568,0.6273847215120802],[0.770594596706954,0.6549212808448839],[0.7916337418852122,0.663230016724536],[0.8158040182195146,0.6790821584765582],[0.8429829058338096,0.6912189746344286],[0.8630733570810515,0.6911501856670759],[0.8778769001094803,0.6956795366104686],[0.9,0.7]])
pos1 = np.array([[0.1,0.3],[0.13169450271296926,0.32318941186962846],[0.15059176427488472,0.3530938374047187],[0.17295308995829867,0.3804757098318936],[0.19948073190079527,0.4185200848579437],[0.20891114586437254,0.43786984165421416],[0.20673484742533774,0.4439673979826771],[0.20906713787864478,0.47879687056525483],[0.18306554911973374,0.4917083741070392],[0.1745181313799723,0.5171082606296313],[0.16548840626749345,0.5324791136276762],[0.1614571781044452,0.5530832150649296],[0.16719761241374456,0.5778162016896454],[0.18897191521053547,0.5926754398163359],[0.20878162250504181,0.6040749297432879],[0.2346588407449217,0.604024164838888],[0.2441597475436523,0.6040711290407034],[0.2803627246319281,0.5931792165280098],[0.3225398032756708,0.5684731695063712],[0.32408095438092904,0.5627145899727077],[0.33618828819974467,0.5478703374051157],[0.3490818791906254,0.5279053756626687],[0.3896936970053528,0.5211751303045014],[0.4092417176117545,0.5070067840588051],[0.4317966289146298,0.5007893800742477],[0.4527814474549683,0.48470447126499194],[0.47751917388518295,0.479278720839447],[0.5130324333902296,0.5001516855730354],[0.5433378919959182,0.5203010245457065],[0.5628208133862622,0.5396061391019865],[0.5674339646644095,0.5877869440593282],[0.582311658693373,0.6294720145298899],[0.6256528820596492,0.6519930648224691],[0.650418750071212,0.6782406577251098],[0.6526730942919738,0.6902376881372798],[0.6432072322175839,0.7141793575076045],[0.6296255459045561,0.7380158964174893],[0.6223143476164774,0.769804378900576],[0.6107843394851954,0.7961121952502594],[0.6004043413291097,0.8284181413328471],[0.6100912046495275,0.8572208886332843],[0.6401506644997219,0.8621146856244578],[0.6726361845319025,0.8686259987756806],[0.7108486352813884,0.8775463311588663],[0.7266604986015615,0.8882129334663378],[0.7415112499996528,0.8835789665015218],[0.7569941162699695,0.8740324062273446],[0.7698951651273948,0.8847474403988987],[0.7427973921612858,0.8859864062083588],[0.7238414886724116,0.8941769161419608],[0.7,0.9]])
pos2 = np.array([[0.3,0.1],[0.3098218705325589,0.11613361781523115],[0.31319002726288375,0.1256052048533204],[0.3286687787622328,0.12397724159627586],[0.35625118693989616,0.12523879758996026],[0.3929527322426045,0.1212028761447945],[0.4123988031005576,0.12011912377018434],[0.42195758554188034,0.12240362227424971],[0.4341941712865556,0.13395472060572994],[0.44257548422937026,0.14958589931471364],[0.44381814662357655,0.16618400867194905],[0.45647352974619665,0.18717088112431468],[0.4562703434823017,0.19474635493410114],[0.4711964311510205,0.19878763706451222],[0.4971583711779627,0.19753510063166557],[0.5067960258088029,0.21055077017748225],[0.5315872758552652,0.23242930734327663],[0.548095752143234,0.24975577052138717],[0.5652296185236343,0.2692390629380962],[0.5747152954174635,0.29010931571184373],[0.5758463929499773,0.32069490056890043],[0.5867170347587719,0.3306444207666325],[0.6009000338983355,0.33420255281889427],[0.6113647004887579,0.33353036830616195],[0.6102698981424417,0.3492577691675293],[0.609809094101377,0.3604290983628403],[0.6196392744354461,0.37156767974147126],[0.6360680519723712,0.37303346442225227],[0.6614864669071258,0.37558703429747586],[0.6663947729058727,0.38856652570922856],[0.6758686618216541,0.3940353000751676],[0.6887265798114134,0.4102607893328536],[0.6971082424060732,0.42487915894191575],[0.7205827022479797,0.44842086446859303],[0.7439582119612869,0.4618603762068713],[0.7689019073254747,0.4689151533260303],[0.7893156250417281,0.47524205569490363],[0.8039462276600264,0.4818844589173125],[0.8108140672471456,0.5051454471728497],[0.8008017420016802,0.5216601865019069],[0.8046208019993539,0.5297374028066661],[0.8187894108749262,0.5407766080659058],[0.8242875521950196,0.5645096268353779],[0.818818915872195,0.5754667764615793],[0.8297691897699404,0.5809631142367926],[0.8478282053206447,0.5912377072185978],[0.8658128646503765,0.6136864048947105],[0.8867632230674947,0.6316877722757096],[0.8913354470144133,0.6611913423199387],[0.8882986819014499,0.6851637621698597],[0.9,0.7]])
pos3 = np.array([[0.1,0.1],[0.1342026592286053,0.09718127323158796],[0.17330841613112208,0.11008077560379018],[0.19004730696641248,0.10930698036267403],[0.2111302081627555,0.11983352297646842],[0.22970065474087967,0.13098022739082688],[0.2410222466967294,0.13425341387257894],[0.2693409889567594,0.14154023485920364],[0.2862576734566089,0.16004889327575605],[0.3113992738902258,0.17438379030400034],[0.3361622362034326,0.1950279852439415],[0.34849548500446015,0.1967847979156378],[0.36255336789709414,0.21167716937515585],[0.3765880774780467,0.22677339720713424],[0.39123679822637414,0.2446040332062423],[0.3910779993588574,0.25035884074361475],[0.3942761425407971,0.24697923483672207],[0.40614129345611244,0.2403316601309278],[0.42982979938586674,0.2405595369623857],[0.4474166359046933,0.2575005557397369],[0.4606352528322446,0.27697688428803446],[0.47469348879749734,0.2909686812353893],[0.5049292553785821,0.3186410206126816],[0.527233164209978,0.3414703228432756],[0.5312184681059136,0.36130661861206054],[0.5499755253080587,0.3869548047478102],[0.5496527878493961,0.40702107251320996],[0.5707316812951585,0.4313280918901069],[0.5743068788811152,0.448711286392241],[0.5923767924399318,0.4679552382805492],[0.6056546712309209,0.4755294211267615],[0.6192463669800807,0.4833269632027002],[0.6391106819810063,0.4944724589031516],[0.6396523133577537,0.5016400249846212],[0.6497283951184418,0.5173789592311102],[0.674567040290959,0.5442138679275249],[0.6930634449005596,0.571670588580685],[0.7071431311938031,0.5951194479171644],[0.7155136627899747,0.6248962498193802],[0.7323180820265388,0.6600604930791227],[0.7520700015396808,0.6766281607352748],[0.7661809806520323,0.7115898953552343],[0.7727909183104765,0.7402620721121127],[0.7838902913652461,0.7619006869370581],[0.8004043487433713,0.7826955138619649],[0.8124957905002622,0.7971281405835476],[0.816463282057055,0.8151722985978279],[0.8359376797403851,0.8290753720281564],[0.8548682414954293,0.8490446210452273],[0.8788360521134478,0.8720824055500978],[0.9,0.9]])

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
        "insertion_max_restarts": algorithm_args[1],
        "insertion_min_restarts": 10,
        #"results_folder": "a={:.2f},b={:.2f},n={:.1f},date={}".format(*algorithm_args,strftime("%m%d%H%M",localtime())),
        "results_folder": "seg={},restarts={},pooling={},date={}".format(*algorithm_args,strftime("%m%d%H%M",localtime())),
        "multistart_pooling_num": algorithm_args[2],
        "insertion_min_segments": 1,
        "insertion_max_segments": algorithm_args[0],
        "TOL": 10**(-8)
    }
	
    DGCG.config.time_limit = True
    #DGCG.config.multistart_proposition_max_iter = 100000
    #DGCG.config.full_max_time = 720000
    DGCG.config.interpolation_sampling = False

    print("Solve about to start.")
    solution_measure = DGCG.solve(noisy_data, **simulation_parameters)

    #------------------------------

    recovered_data = DGCG.operators.K_t_star_full(solution_measure)
    diff = recovered_data - data

    er = DGCG.operators.int_time_H_t_product(diff,diff)
    print("Data error of the recovered solution:")
    print(er)