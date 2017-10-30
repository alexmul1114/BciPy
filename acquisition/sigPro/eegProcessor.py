import numpy as np

def sigPro(inputSeq, filt = None, fs = 256, k = 2, mode = '', channels = 16):
    '''
    :param inputSeq: Input sequence to be filtered. Expected dimensions are 16xT
    :param filt: Input for using a specific filter. If left empty, according to fs a pre-designed filter is going to be used. Filters are pre-designed for fs = 256,300 or 1024 Hz.
    :param fs: Sampling frequency of the hardware.
    :param k: downsampling order
    :param mode: ??? I am going to work on mode later on.
    :param channels: Number of channels
    :return: output sequence that is filtered and downsampled input. Filter delay is compensated. Dimensions are 16xT/k

    256Hz
        - 1.75Hz to 45Hz
        - 60Hz -64dB Gain

    300Hz
        - 1.84Hz to 45Hz
        - 60Hz -84dB Gain

    1024Hz
        - 1.75Hz to 45Hz
        - 60Hz -64dB Gain

    '''

    if fs == 256 and filt is None:
        filt = [0.000529149323385401, -0.000569534006625737, -0.00230632150128694, -0.00419950021683356, -0.00484633946050219, -0.00361374723259779, -0.00144774566972321, -0.000215773028930033, -0.000981137476096742, -0.00289052329436235, -0.00395720722514359, -0.00310951535865294, -0.00138770248941459, -0.000885730225734469, -0.00243487184084065, -0.00455360703925706, -0.0049748745356555, -0.00321848102902774, -0.00131840661857347, -0.00167560101416167, -0.00427256386274698, -0.0064582584554769, -0.00580480160482341, -0.00298811557914936, -0.00130402542092165, -0.00304126893086636, -0.00668970519253021, -0.0083375510428129, -0.00604763273618894, -0.0023324538076735, -0.00163181289339621, -0.00526767141541532, -0.009582361055121, -0.00976366488073788, -0.00540320875054485, -0.00140419738374147, -0.00271697105637597, -0.00850964323647693, -0.012590566466764, -0.0101878817363249, -0.00371067885204887, -0.00060443378889687, -0.00506079730013307, -0.0127568270457618, -0.0151194835737664, -0.00901990441512876, -0.00100745906298992, -0.000549040120863626, -0.00914431064641963, -0.0177376198981833, -0.0163221408175357, -0.00567400837658128, 0.00240933608093388, -0.00213040624869348, -0.0154994066639012, -0.0229648640944487, -0.0150529901213322, 0.000561211648513562, 0.00604086133150342, -0.00669097176742883, -0.025012758977698, -0.0277721811869047, -0.00932198117814122, 0.0113244416188075, 0.00927356122123326, -0.0173518624824942, -0.0408653051170222, -0.031472312971325, 0.00718668394996153, 0.0343684220916795, 0.011499879756672, -0.0507138649771079, -0.087782115740869, -0.0334887885825584, 0.114088755699163, 0.275651662622892, 0.345627046869997, 0.275651662622892, 0.114088755699163, -0.0334887885825584, -0.087782115740869, -0.0507138649771079, 0.011499879756672, 0.0343684220916795, 0.00718668394996153, -0.031472312971325, -0.0408653051170222, -0.0173518624824942, 0.00927356122123326, 0.0113244416188075, -0.00932198117814122, -0.0277721811869047, -0.025012758977698, -0.00669097176742883, 0.00604086133150342, 0.000561211648513562, -0.0150529901213322, -0.0229648640944487, -0.0154994066639012, -0.00213040624869348, 0.00240933608093388, -0.00567400837658128, -0.0163221408175357, -0.0177376198981833, -0.00914431064641963, -0.000549040120863626, -0.00100745906298992, -0.00901990441512876, -0.0151194835737664, -0.0127568270457618, -0.00506079730013307, -0.00060443378889687, -0.00371067885204887, -0.0101878817363249, -0.012590566466764, -0.00850964323647693, -0.00271697105637597, -0.00140419738374147, -0.00540320875054485, -0.00976366488073788, -0.009582361055121, -0.00526767141541532, -0.00163181289339621, -0.0023324538076735, -0.00604763273618894, -0.0083375510428129, -0.00668970519253021, -0.00304126893086636, -0.00130402542092165, -0.00298811557914936, -0.00580480160482341, -0.0064582584554769, -0.00427256386274698, -0.00167560101416167, -0.00131840661857347, -0.00321848102902774, -0.0049748745356555, -0.00455360703925706, -0.00243487184084065, -0.000885730225734469, -0.00138770248941459, -0.00310951535865294, -0.00395720722514359, -0.00289052329436235, -0.000981137476096742, -0.000215773028930033, -0.00144774566972321, -0.00361374723259779, -0.00484633946050219, -0.00419950021683356, -0.00230632150128694, -0.000569534006625737, 0.000529149323385401]

    elif fs == 300:

        filt = [0.000292625983003407, -0.000524638045719409, -0.00165326328780024, -0.00318757285900911, -0.00437471994482875, -0.00449623750435235, -0.00340464690559018, -0.00173748134484343, -0.000562822235770538, -0.00064229922083066, -0.00184709728001998, -0.00321704620961398, -0.00367924970622776, -0.00289947407001221, -0.00157874055593102, -0.000911589836448399, -0.00159718551084628, -0.00320823467801509, -0.00448460277858453, -0.00436984357286943, -0.00294520862819609, -0.00144670388204673, -0.00127588537784868, -0.00278389272623663, -0.00487604738138688, -0.00585810324183138, -0.0048681448905749, -0.00270968592701302, -0.00128012807338391, -0.00199520121368384, -0.00450407404251353, -0.00684945392971228, -0.00705703089261176, -0.00486922435940026, -0.00211491930250085, -0.00127101851378544, -0.00333779377273398, -0.00683725243363459, -0.00891152532193953, -0.00772365544954445, -0.00415772277693311, -0.00126117785067666, -0.0017397208245845, -0.00554854492809924, -0.00970827319985953, -0.0106875722123308, -0.00746686951461989, -0.00263929419203351, -0.000424034055685841, -0.00308565884057437, -0.00877898553462981, -0.0128349540329063, -0.0116375064256179, -0.00589603569364903, -0.000385146562062118, -8.11673161295026e-05, -0.00579462370928907, -0.0130938306873729, -0.015755343343821, -0.0110936233180732, -0.00264990820268525, 0.0023661484197682, -0.000898054435713402, -0.0104266463466768, -0.0184550676854263, -0.01776855494517, -0.00814265851468789, 0.00273343983990358, 0.0052246221910321, -0.00392116486768719, -0.0179629599767725, -0.024973356101223, -0.0178026039822977, -0.00104631534123702, 0.0114742380556717, 0.00772778732783745, -0.0115375849697881, -0.0314515924335526, -0.0339239244462167, -0.0132052121482301, 0.0165881957721921, 0.0298354157361063, 0.00943981990370132, -0.0359561870297561, -0.0720377195930181, -0.0591299402329566, 0.0194678963443084, 0.140913405071435, 0.251305172824872, 0.295764927420209, 0.251305172824872, 0.140913405071435, 0.0194678963443084, -0.0591299402329566, -0.0720377195930181, -0.0359561870297561, 0.00943981990370132, 0.0298354157361063, 0.0165881957721921, -0.0132052121482301, -0.0339239244462167, -0.0314515924335526, -0.0115375849697881, 0.00772778732783745, 0.0114742380556717, -0.00104631534123702, -0.0178026039822977, -0.024973356101223, -0.0179629599767725, -0.00392116486768719, 0.0052246221910321, 0.00273343983990358, -0.00814265851468789, -0.01776855494517, -0.0184550676854263, -0.0104266463466768, -0.000898054435713402, 0.0023661484197682, -0.00264990820268525, -0.0110936233180732, -0.015755343343821, -0.0130938306873729, -0.00579462370928907, -8.11673161295026e-05, -0.000385146562062118, -0.00589603569364903, -0.0116375064256179, -0.0128349540329063, -0.00877898553462981, -0.00308565884057437, -0.000424034055685841, -0.00263929419203351, -0.00746686951461989, -0.0106875722123308, -0.00970827319985953, -0.00554854492809924, -0.0017397208245845, -0.00126117785067666, -0.00415772277693311, -0.00772365544954445, -0.00891152532193953, -0.00683725243363459, -0.00333779377273398, -0.00127101851378544, -0.00211491930250085, -0.00486922435940026, -0.00705703089261176, -0.00684945392971228, -0.00450407404251353, -0.00199520121368384, -0.00128012807338391, -0.00270968592701302, -0.0048681448905749, -0.00585810324183138, -0.00487604738138688, -0.00278389272623663, -0.00127588537784868, -0.00144670388204673, -0.00294520862819609, -0.00436984357286943, -0.00448460277858453, -0.00320823467801509, -0.00159718551084628, -0.000911589836448399, -0.00157874055593102, -0.00289947407001221, -0.00367924970622776, -0.00321704620961398, -0.00184709728001998, -0.00064229922083066, -0.000562822235770538, -0.00173748134484343, -0.00340464690559018, -0.00449623750435235, -0.00437471994482875, -0.00318757285900911, -0.00165326328780024, -0.000524638045719409, 0.000292625983003407]

    elif fs == 1024:
        filt = [0.000523786251514105, 3.87517934412172e-06, -2.54423765234248e-05, -7.49645440718051e-05, -0.000144328118915576, -0.00023214944922881, -0.000336132448505403, -0.000452804137460562, -0.000577805794693037, -0.000705829933194352, -0.000831081709609562, -0.000947378207033754, -0.00104872204868974, -0.00112946477402538, -0.00118490775950651, -0.00121140530018079, -0.00120689016523218, -0.00117085647888139, -0.00110469818058694, -0.00101150170317264, -0.000896179124673477, -0.000765001705083662, -0.000625480898253175, -0.000485685458830873, -0.000353923617747896, -0.000237969807795437, -0.000144764874193008, -7.97709752885796e-05, -4.68846763806537e-05, -4.79843160280746e-05, -8.29667932800548e-05, -0.000149284821988953, -0.000239904746081562, -0.000351399763988775, -0.000474041997331437, -0.000599527431240235, -0.000719005614849908, -0.000823940777154933, -0.000906969727267154, -0.000962208371609206, -0.000985936643983367, -0.000976672307562149, -0.00093553408647006, -0.000865970310569729, -0.0007737504776159, -0.000666328437166561, -0.000552546595184209, -0.000441734999927768, -0.000343263059336225, -0.00026559934256479, -0.000215879403807853, -0.000199086606339213, -0.00021786541901109, -0.000272013732719035, -0.000358684657337593, -0.000472340789260326, -0.000605382184647091, -0.000748466900007391, -0.000891398284654632, -0.00102361159636094, -0.00113520847231141, -0.00121757633008173, -0.00126449795439383, -0.00127252645420843, -0.00124014713691341, -0.00117034456813089, -0.00106821724948542, -0.000941753689626347, -0.000801092248300153, -0.000657593426565862, -0.000523233494316809, -0.00040943725481188, -0.000326410617037065, -0.000282078834688209, -0.000281636264840459, -0.000326850901916812, -0.000416078089979872, -0.000544068101556255, -0.000702554571134236, -0.000880564512538193, -0.00106545241027708, -0.00124360665344725, -0.00140173486453404, -0.00152766821146689, -0.00161161110461248, -0.00164671594999791, -0.00162992019502692, -0.0015620184435744, -0.00144793110723937, -0.00129620521620477, -0.00111870363476218, -0.000929501775131885, -0.000744020570546536, -0.000577550355991849, -0.000444423208169064, -0.000356914648375199, -0.000323490793057166, -0.000349049709727141, -0.000433797280718459, -0.00057327642943646, -0.000758750140893065, -0.000977540349656455, -0.00121417927898851, -0.00145130680033178, -0.00167122742856783, -0.00185706399997781, -0.00199432830768738, -0.00207188821173791, -0.00208315446226572, -0.00202644144849668, -0.00190548587605023, -0.00172902693870749, -0.00151049116083395, -0.00126680974884041, -0.00101741894929593, -0.000782522707912656, -0.000581783429837202, -0.000432526219390972, -0.000348584397628193, -0.000338881702338067, -0.000406840488206928, -0.000549725205001181, -0.000758980584823378, -0.00102055457151002, -0.0013161135721382, -0.00162402610390772, -0.00192123625872022, -0.00218516535838315, -0.00239507231571756, -0.00253410865479646, -0.00259060664170277, -0.00255900279620246, -0.002440642831199, -0.00224358523972044, -0.0019823828524108, -0.00167684644155006, -0.00135084364542212, -0.00103029083020185, -0.000741385447378444, -0.00050833053796866, -0.000351643660371975, -0.000286189572323384, -0.000320149914294003, -0.000453963805904469, -0.000680355398174358, -0.000984481625915278, -0.00134519029337573, -0.00173626859914343, -0.00212863119424462, -0.00249227751574636, -0.00279888572035964, -0.00302383976084572, -0.00314849720684746, -0.00316151885951181, -0.00306016612217167, -0.00285049420546869, -0.0025473882548159, -0.00217327418731915, -0.00175667936508655, -0.0013302354956051, -0.000928023206592383, -0.000583010363664135, -0.00032453029796731, -0.000175689756430349, -0.000151656194995359, -0.000258017998501842, -0.000490370923710825, -0.000834189613411292, -0.00126595980337746, -0.00175459666053463, -0.0022639210400661, -0.00275513845127946, -0.00319008777153283, -0.00353397621126646, -0.00375846386043715, -0.00384379335009326, -0.00378078770704966, -0.00357156139153526, -0.00322989611286832, -0.00278018531872204, -0.00225603283978345, -0.00169757084955848, -0.00114871390290098, -0.000653557875266843, -0.000253193714317712, 1.78466867907917e-05, 0.000134236661681028, 8.23371954158403e-05, -0.000138768350341377, -0.000516624099730917, -0.00102614272845078, -0.00163129818249937, -0.00228765659056389, -0.00294556811426131, -0.00355411550254574, -0.00406483907842577, -0.00443586396535932, -0.00463509656729009, -0.00464323796421169, -0.00445539086063053, -0.00408202874111275, -0.00354836587684358, -0.0028929832031738, -0.00216489783542607, -0.00142026700784195, -0.000717887123342001, -0.000114913305299484, 0.000337908103024809, 0.000600134634659637, 0.000645348384719059, 0.0004633457201509, 6.16344611252207e-05, -0.000534940495313594, -0.00128541290824869, -0.00213570905110576, -0.00302225589960236, -0.00387696341362705, -0.00463219520922275, -0.00522632851465347, -0.00560845512759574, -0.00574298314964555, -0.00561265702228861, -0.00522052439951307, -0.00459025394773235, -0.00376498240390836, -0.00280434937882015, -0.00178055109428497, -0.000772811376036786, 0.000138341783475215, 0.00087758884036111, 0.00138063078262565, 0.00159978131003632, 0.00150802297039577, 0.00110198066104084, 0.000402715708589733, -0.000544859327404457, -0.00167499670874015, -0.00290527913183876, -0.00414274161472499, -0.00529052040557463, -0.00625554782365686, -0.00695567672652547, -0.00732672689742773, -0.00732785580108998, -0.00694584150842931, -0.00619686100311706, -0.00512657402648547, -0.00380739872523325, -0.00233414307726574, -0.00081725030272517, 0.000624900476312159, 0.00187518184181649, 0.00282658044825955, 0.00339107517663262, 0.0035071337060031, 0.00314569077682276, 0.00231361194532992, 0.00105492602579762, -0.000551197919940685, -0.002394724201724, -0.00434208413755962, -0.00624511340423112, -0.00795197384573126, -0.0093184286713754, -0.0102194444920723, -0.010559501929352, -0.0102816281122529, -0.00937344284290327, -0.00787075612496335, -0.00585720118216847, -0.00346080541339631, -0.000846586731651826, 0.00179342781154895, 0.00425323665396711, 0.00632710089399129, 0.007825094358835, 0.0085878651072025, 0.00850059132010467, 0.00750406517833473, 0.00560307134020792, 0.00287031009114264, -0.000553589931580037, -0.00446520350054599, -0.0086073205786882, -0.0126825421953879, -0.016369823601955, -0.019343699046257, -0.0212944475009119, -0.0219485993277458, -0.0210876728544585, -0.0185648387287478, -0.0143173618539298, -0.00837494325320306, -0.000862309812532803, 0.00800335495003057, 0.0179209140941678, 0.0285186398075414, 0.0393724500969978, 0.0500271666047664, 0.0600206188734744, 0.0689080546034068, 0.0762864943949476, 0.0818165696556312, 0.0852415393997446, 0.0864012938090299, 0.0852415393997446, 0.0818165696556312, 0.0762864943949476, 0.0689080546034068, 0.0600206188734744, 0.0500271666047664, 0.0393724500969978, 0.0285186398075414, 0.0179209140941678, 0.00800335495003057, -0.000862309812532803, -0.00837494325320306, -0.0143173618539298, -0.0185648387287478, -0.0210876728544585, -0.0219485993277458, -0.0212944475009119, -0.019343699046257, -0.016369823601955, -0.0126825421953879, -0.0086073205786882, -0.00446520350054599, -0.000553589931580037, 0.00287031009114264, 0.00560307134020792, 0.00750406517833473, 0.00850059132010467, 0.0085878651072025, 0.007825094358835, 0.00632710089399129, 0.00425323665396711, 0.00179342781154895, -0.000846586731651826, -0.00346080541339631, -0.00585720118216847, -0.00787075612496335, -0.00937344284290327, -0.0102816281122529, -0.010559501929352, -0.0102194444920723, -0.0093184286713754, -0.00795197384573126, -0.00624511340423112, -0.00434208413755962, -0.002394724201724, -0.000551197919940685, 0.00105492602579762, 0.00231361194532992, 0.00314569077682276, 0.0035071337060031, 0.00339107517663262, 0.00282658044825955, 0.00187518184181649, 0.000624900476312159, -0.00081725030272517, -0.00233414307726574, -0.00380739872523325, -0.00512657402648547, -0.00619686100311706, -0.00694584150842931, -0.00732785580108998, -0.00732672689742773, -0.00695567672652547, -0.00625554782365686, -0.00529052040557463, -0.00414274161472499, -0.00290527913183876, -0.00167499670874015, -0.000544859327404457, 0.000402715708589733, 0.00110198066104084, 0.00150802297039577, 0.00159978131003632, 0.00138063078262565, 0.00087758884036111, 0.000138341783475215, -0.000772811376036786, -0.00178055109428497, -0.00280434937882015, -0.00376498240390836, -0.00459025394773235, -0.00522052439951307, -0.00561265702228861, -0.00574298314964555, -0.00560845512759574, -0.00522632851465347, -0.00463219520922275, -0.00387696341362705, -0.00302225589960236, -0.00213570905110576, -0.00128541290824869, -0.000534940495313594, 6.16344611252207e-05, 0.0004633457201509, 0.000645348384719059, 0.000600134634659637, 0.000337908103024809, -0.000114913305299484, -0.000717887123342001, -0.00142026700784195, -0.00216489783542607, -0.0028929832031738, -0.00354836587684358, -0.00408202874111275, -0.00445539086063053, -0.00464323796421169, -0.00463509656729009, -0.00443586396535932, -0.00406483907842577, -0.00355411550254574, -0.00294556811426131, -0.00228765659056389, -0.00163129818249937, -0.00102614272845078, -0.000516624099730917, -0.000138768350341377, 8.23371954158403e-05, 0.000134236661681028, 1.78466867907917e-05, -0.000253193714317712, -0.000653557875266843, -0.00114871390290098, -0.00169757084955848, -0.00225603283978345, -0.00278018531872204, -0.00322989611286832, -0.00357156139153526, -0.00378078770704966, -0.00384379335009326, -0.00375846386043715, -0.00353397621126646, -0.00319008777153283, -0.00275513845127946, -0.0022639210400661, -0.00175459666053463, -0.00126595980337746, -0.000834189613411292, -0.000490370923710825, -0.000258017998501842, -0.000151656194995359, -0.000175689756430349, -0.00032453029796731, -0.000583010363664135, -0.000928023206592383, -0.0013302354956051, -0.00175667936508655, -0.00217327418731915, -0.0025473882548159, -0.00285049420546869, -0.00306016612217167, -0.00316151885951181, -0.00314849720684746, -0.00302383976084572, -0.00279888572035964, -0.00249227751574636, -0.00212863119424462, -0.00173626859914343, -0.00134519029337573, -0.000984481625915278, -0.000680355398174358, -0.000453963805904469, -0.000320149914294003, -0.000286189572323384, -0.000351643660371975, -0.00050833053796866, -0.000741385447378444, -0.00103029083020185, -0.00135084364542212, -0.00167684644155006, -0.0019823828524108, -0.00224358523972044, -0.002440642831199, -0.00255900279620246, -0.00259060664170277, -0.00253410865479646, -0.00239507231571756, -0.00218516535838315, -0.00192123625872022, -0.00162402610390772, -0.0013161135721382, -0.00102055457151002, -0.000758980584823378, -0.000549725205001181, -0.000406840488206928, -0.000338881702338067, -0.000348584397628193, -0.000432526219390972, -0.000581783429837202, -0.000782522707912656, -0.00101741894929593, -0.00126680974884041, -0.00151049116083395, -0.00172902693870749, -0.00190548587605023, -0.00202644144849668, -0.00208315446226572, -0.00207188821173791, -0.00199432830768738, -0.00185706399997781, -0.00167122742856783, -0.00145130680033178, -0.00121417927898851, -0.000977540349656455, -0.000758750140893065, -0.00057327642943646, -0.000433797280718459, -0.000349049709727141, -0.000323490793057166, -0.000356914648375199, -0.000444423208169064, -0.000577550355991849, -0.000744020570546536, -0.000929501775131885, -0.00111870363476218, -0.00129620521620477, -0.00144793110723937, -0.0015620184435744, -0.00162992019502692, -0.00164671594999791, -0.00161161110461248, -0.00152766821146689, -0.00140173486453404, -0.00124360665344725, -0.00106545241027708, -0.000880564512538193, -0.000702554571134236, -0.000544068101556255, -0.000416078089979872, -0.000326850901916812, -0.000281636264840459, -0.000282078834688209, -0.000326410617037065, -0.00040943725481188, -0.000523233494316809, -0.000657593426565862, -0.000801092248300153, -0.000941753689626347, -0.00106821724948542, -0.00117034456813089, -0.00124014713691341, -0.00127252645420843, -0.00126449795439383, -0.00121757633008173, -0.00113520847231141, -0.00102361159636094, -0.000891398284654632, -0.000748466900007391, -0.000605382184647091, -0.000472340789260326, -0.000358684657337593, -0.000272013732719035, -0.00021786541901109, -0.000199086606339213, -0.000215879403807853, -0.00026559934256479, -0.000343263059336225, -0.000441734999927768, -0.000552546595184209, -0.000666328437166561, -0.0007737504776159, -0.000865970310569729, -0.00093553408647006, -0.000976672307562149, -0.000985936643983367, -0.000962208371609206, -0.000906969727267154, -0.000823940777154933, -0.000719005614849908, -0.000599527431240235, -0.000474041997331437, -0.000351399763988775, -0.000239904746081562, -0.000149284821988953, -8.29667932800548e-05, -4.79843160280746e-05, -4.68846763806537e-05, -7.97709752885796e-05, -0.000144764874193008, -0.000237969807795437, -0.000353923617747896, -0.000485685458830873, -0.000625480898253175, -0.000765001705083662, -0.000896179124673477, -0.00101150170317264, -0.00110469818058694, -0.00117085647888139, -0.00120689016523218, -0.00121140530018079, -0.00118490775950651, -0.00112946477402538, -0.00104872204868974, -0.000947378207033754, -0.000831081709609562, -0.000705829933194352, -0.000577805794693037, -0.000452804137460562, -0.000336132448505403, -0.00023214944922881, -0.000144328118915576, -7.49645440718051e-05, -2.54423765234248e-05, 3.87517934412172e-06, 0.000523786251514105]
    else:
        print 'Please provide a filter for your sampling frequency'


    filt = np.array(filt)
    filt = filt - np.sum(filt)/filt.size

    outputSeq = [[]]

    # Convolution per channel
    for z in range(channels):
        temp = np.convolve(inputSeq[z][:], filt)
        # Filter off-set compensation
        temp = temp[int(np.ceil(len(filt)/2.))-1:];
        # Downsampling
        outputSeq.append(temp[::k])

    return np.array(outputSeq[1:])