import os,segyio,shutil,math,random,time
from SUN_new123_Frequ import *
# from SNR import *
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

mynet = myMDnet()

model = mynet.get_MDnet()

nxtr = 1
nTrace = 496
nPoint = 881
nBlockT = 32
nBlockX = 32
epochs = 15
novv = 1.5  # Window stride: larger value → less overlap, fewer windows
# nGa = math.ceil(nGather / nxtr)
input_num = 1
output_num = 1
nGather = 1
if input_num == 1:
    num= 1
else:
    num= nGather-(input_num-1)
ff = 0  #Start reading from which gather (0 = start from the first gather)
if input_num %2 == 1:
    n = math.ceil((output_num-1)/2)
    m = math.ceil((input_num-1)/2)
    p = math.ceil((input_num+1)/2)
else:
    n = math.ceil((output_num) / 2)
    m = math.ceil((input_num) / 2)

sgypath1="Cmult_test_allgather.sgy"
sgypath2="Corig_test_allgather.sgy"
sgypath3="Cprim_test_allgather.sgy"
sgypath4="predict_result_Frequ.sgy"
sgypath5="CPT_mult_Frequ.sgy"
sgypath6="subtract_Frequ.sgy"

nOverlapT = math.floor(nBlockT / novv)
nOverlapX = math.floor(nBlockX / novv)

nBx = math.ceil((nPoint - nOverlapT) / (nBlockT - nOverlapT))
if nBlockT == nPoint:
    nBx = 1
nBy = math.ceil((nTrace - nOverlapX) / (nBlockX - nOverlapX))
if nBlockX == nTrace:
    nBy = 1

print("Reading Data")
nnorig = np.zeros((nGather, 1))
nnpred = np.zeros((nGather, 1))
shutil.copyfile(sgypath1, sgypath4)
shutil.copyfile(sgypath1, sgypath5)
shutil.copyfile(sgypath3, sgypath6)

tr2=0
for k in range(num):
    tr1 = tr2
    tr2 = tr1 + nxtr * nTrace
    print('k=', k)
    print('tr1=', tr1)
    print('tr2=', tr2)
    if tr2 > nGather * nTrace:
        tr2 = nGather * nTrace
        tr1 = (nGather - nxtr) * nTrace


    yibo = np.zeros((nPoint, nGather * nTrace))
    dy=list()
    if input_num % 2 == 1:
        for i in range(-n,n+1):
            with segyio.open(sgypath3, ignore_geometry=True) as f:
                if input_num == 1:
                    inp = f.trace.raw[(ff + k + i + n + m) * nTrace:(ff + k + i + n + p) * nTrace]
                else:
                    inp = f.trace.raw[(ff +k + i+m) * nTrace:(ff +k + i+p) * nTrace]
                yibo = np.transpose(inp)
                dy.append(yibo)
    else:
        for i in range(output_num):
            with segyio.open(sgypath3, ignore_geometry=True) as f:
                inp = f.trace.raw[(ff + k + i + (m - n)) * nTrace:(ff + k + i + (m - n + 1)) * nTrace]
                yibo = np.transpose(inp)
                dy.append(yibo)

    da = list()
    nnorig= list()
    dalabel= list()
    if input_num % 2 == 1:
        for i in range(-n,n+1):
            with segyio.open(sgypath2, ignore_geometry=True) as f:
                if input_num == 1:
                    inp = f.trace.raw[(ff + k + i+n + m) * nTrace:(ff + k + i+n + p) * nTrace]
                else:
                    inp = f.trace.raw[(ff +k + i+m) * nTrace:(ff +k + i+p) * nTrace]
                orig = np.transpose(inp)
                nnorign = (abs(orig)).max()
                nnorig.append(nnorign)
                dalabeln = orig / nnorig[i+n]
                dalabel.append(dalabeln)
            temp = mynet.window_strate_vec(dalabel[i+n], nBlockT, nBlockX, nTrace, nPoint, nxtr, novv)
            temp = np.squeeze(temp)
            temp = np.reshape(temp, (-1, nBlockT, nBlockX, 1))
            temp= np.transpose(temp, [0, 2, 1, 3])
            da.append(temp)
    else:
        for i in range(output_num):
            with segyio.open(sgypath2, ignore_geometry=True) as f:
                inp = f.trace.raw[(ff + k + i + (m - n)) * nTrace:(ff + k + i + (m - n + 1)) * nTrace]
                orig = np.transpose(inp)
                nnorign = (abs(orig)).max()
                nnorig.append(nnorign)
                dalabeln = orig / nnorig[i]
                dalabel.append(dalabeln)
            temp = mynet.window_strate_vec(dalabel[i], nBlockT, nBlockX, nTrace, nPoint, nxtr, novv)
            temp = np.squeeze(temp)
            temp = np.reshape(temp, (-1, nBlockT, nBlockX, 1))
            temp = np.transpose(temp, [0, 2, 1, 3])
            da.append(temp)



    if input_num == 1:
        with segyio.open(sgypath1, ignore_geometry=True) as f:
            inp = f.trace.raw[(ff + k+n) * nTrace:(ff + k+(n+1)) * nTrace]
            prim1 = np.transpose(inp)
            nnpred1 = (abs(prim1)).max()
            dapred1 = prim1 / nnpred1

            da11 = mynet.window_strate_vec(dapred1, nBlockT, nBlockX, nTrace, nPoint, nxtr, novv)
            da11 = np.squeeze(da11)
            da11 = np.reshape(da11, (-1, nBlockT, nBlockX, 1))
            da11 = np.transpose(da11, [0, 2, 1, 3])
            da31 = da11
    else:
        dm=list()
        for i in range(input_num):
            with segyio.open(sgypath1, ignore_geometry=True) as f:
                inp = f.trace.raw[(ff + k + i ) * nTrace:(ff +k + i+1) * nTrace]
                prim = np.transpose(inp)
                nnpred = (abs(prim)).max()
                dapred = prim / nnpred
            temp = mynet.window_strate_vec(dapred, nBlockT, nBlockX, nTrace, nPoint, nxtr, novv)
            temp = np.squeeze(temp)
            temp = np.reshape(temp, (-1, nBlockT, nBlockX, 1))
            temp = np.transpose(temp, [0, 2, 1, 3])
            dm.append(temp)


    print("Training")

    t1 = time.perf_counter()
    model.compile(optimizer=Adam(lr=3e-4), loss=myloss)  #First timing
    epp = epochs
    if input_num == 1:
        history = model.fit(da11, da, batch_size=16, epochs=epp, validation_split=0, verbose=2, shuffle=True)
    else:
        history = model.fit(dm, da, batch_size=16, epochs=epp, validation_split=0, verbose=2, shuffle=True)
    t2 = time.perf_counter()
    delta_time = t2 - t1
    print("Training time (seconds)：", delta_time)

    t3 = time.perf_counter()
    if input_num == 1:
        fieldpredict_result = model.predict(da11, batch_size=1, verbose=1)
    else:
        fieldpredict_result = model.predict(dm, batch_size=1, verbose=1)
    t4 = time.perf_counter()
    delta_time = t4 - t3
    print("Inference time (seconds)：", delta_time)


    if input_num % 2 == 1:
        for i in range(-n,n+1):
            if output_num==1:
                fieldpredict_resultn = np.squeeze(fieldpredict_result)
            else:
                fieldpredict_resultn = np.squeeze(fieldpredict_result[i+n])
            fieldpredict_resultn = np.transpose(fieldpredict_resultn, [0, 2, 1])

            arr3d = np.zeros((nBx * nBy * nxtr, nBlockT, nBlockX))
            for ii in range(nBx * nBy * nxtr):
                arr3d[ii, :, :] = fieldpredict_resultn[ii, :, :]
            mubld = mynet.matblend(arr3d, nxtr, nTrace, nPoint, nBlockT, nBlockX, novv)

            if input_num == 1:
                with segyio.open(sgypath4, "r+", strict=False) as f:
                    f.trace.raw[(ff + k + i + n + m) * nTrace:(ff + k + i + n + p) * nTrace] = np.transpose(mubld) * nnorig[i + n]

                with segyio.open(sgypath5, "r+", strict=False) as f:
                    f.trace.raw[(ff + k + i + n + m) * nTrace:(ff + k + i + n + p) * nTrace] = np.transpose(dalabel[i + n] - mubld) * nnorig[i + n]

                with segyio.open(sgypath6, "r+", strict=False) as f:
                    f.trace.raw[(ff + k + i + n + m) * nTrace:(ff + k + i + n + p) * nTrace] = np.transpose(dy[i + n] - (dalabel[i + n] - mubld) * nnorig[i + n])
            else:
                with segyio.open(sgypath4, "r+", strict=False) as f:
                    f.trace.raw[(ff +k + i+m) * nTrace:(ff +k + i+p) * nTrace] = np.transpose(mubld) * nnorig[i+n]

                with segyio.open(sgypath5, "r+", strict=False) as f:
                    f.trace.raw[(ff +k + i+m) * nTrace:(ff +k + i+p) * nTrace] = np.transpose(dalabel[i+n] - mubld) * nnorig[i+n]

                with segyio.open(sgypath6, "r+", strict=False) as f:
                    f.trace.raw[(ff +k + i+m) * nTrace:(ff +k + i+p) * nTrace] = np.transpose(dy[i+n] - (dalabel[i+n] - mubld) * nnorig[i+n])

    else:
        for i in range(output_num):
            fieldpredict_resultn = np.squeeze(fieldpredict_result[i])
            fieldpredict_resultn = np.transpose(fieldpredict_resultn, [0, 2, 1])

            arr3d = np.zeros((nBx * nBy * nxtr, nBlockT, nBlockX))
            for ii in range(nBx * nBy * nxtr):
                arr3d[ii, :, :] = fieldpredict_resultn[ii, :, :]
            mubld = mynet.matblend(arr3d, nxtr, nTrace, nPoint, nBlockT, nBlockX, novv)
            with segyio.open(sgypath4, "r+", strict=False) as f:
                f.trace.raw[(ff + k + i + (m - n)) * nTrace:(ff + k + i + (m - n + 1)) * nTrace] = np.transpose(mubld) * nnorig[i]

            with segyio.open(sgypath5, "r+", strict=False) as f:
                f.trace.raw[(ff + k + i + (m - n)) * nTrace:(ff + k + i + (m - n + 1)) * nTrace] = np.transpose(dalabel[i] - mubld) * nnorig[i]

            with segyio.open(sgypath6, "r+", strict=False) as f:
                f.trace.raw[(ff + k + i + (m - n)) * nTrace:(ff + k + i + (m - n + 1)) * nTrace] = np.transpose(dy[i] - (dalabel[i] - mubld) * nnorig[i])


    if k == 0:
        plt.plot(history.history['loss'], label='training loss', color='b')
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epochs')
        plt.legend()
        plt.savefig('Trainloss.png')

