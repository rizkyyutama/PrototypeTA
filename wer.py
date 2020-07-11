from tkinter import *
from tkinter import filedialog
from datetime import datetime
from tkinter import ttk
import os
from functools import partial

global file_csv
global path_csv
global path_arsitektur
global path_weight
global path_variablemax

global file_model_arsitektur_estimasi
global file_model_weigth_estimasi
global file_model_variable_estimasi

global loaded_model
global loaded_weight
global loaded_variable_max

root = Tk()

root.title("Design Cost Estimation")
root.geometry("640x480+640+300")
root.resizable(False, False)


def home():

    judul_designcost.place(x=193, y=50)
    judul_estimation.place(x=193, y=94)
    button_trainmodel.place(x=224, y=200)
    button_estimatecost.place(x=224, y=270)
    button_exit.place(x=224, y=340)

    # hapus widget model training
    judul_modeltraining1.place_forget()
    caption_modeltraining1.place_forget()
    frame_modeltraining1.place_forget()
    button_cancel_modeltraining1.place_forget()
    button_run_modeltraining1.place_forget()
    button_back_modeltraining1.place_forget()

    # hapus widget estimate cost
    judul_estimatecost1.place_forget()
    caption_estimatecost1.place_forget()
    frame_estimatecost1.place_forget()
    button_cancel_estimatecost1.place_forget()
    button_next_estimatecost1.place_forget()
    button_back_estimatecost1.place_forget()
    judul_estimatecost2.place_forget()
    caption_estimatecost2.place_forget()
    caption_estimatecost3.place_forget()
    folder_path_estimatecost1.place_forget()
    button_filedir_estimatecost1.place_forget()

    #hapus widget estimate cost 2(Tombol next dari estimate cost 1)
    judul_estimatecost1_2.place_forget()
    caption_estimatecost1_2.place_forget()
    frame_estimatecost1_2.place_forget()
    button_cancel_estimatecost1_2.place_forget()
    button_next_estimatecost1_2.place_forget()
    button_back_estimatecost1_2.place_forget()
    judul_estimatecost2_2.place_forget()
    folder_path_estimatecost1_2.place_forget()
    button_filedir_estimatecost1_2.place_forget()

    #hapus widget trainmodel2
    judul_modeltraining1_2.place_forget()
    caption_modeltraining1_2.place_forget()
    frame_modeltraining1_2.place_forget()
    button_done_modeltraining1_2.place_forget()
    button_back_modeltraining1_2.place_forget()
    judul_modeltraining2_2.place_forget()
    caption_modeltraining2_2.place_forget()
    caption_modeltraining3_2.place_forget()
    progressbar_modeltraining_2.place_forget()
    progress_modeltraining1_2.place_forget()
    progress_modeltraining2_2.place_forget()

def trainmodel():
    judul_modeltraining1.place(x=25, y=20)
    caption_modeltraining1.place(x=25, y=60)
    frame_modeltraining1.place(x=20, y=100)
    button_cancel_modeltraining1.place(x=529, y=430)
    button_run_modeltraining1.place(x=430, y=430)
    button_back_modeltraining1.place(x=330, y=430)
    judul_modeltraining2.place(x=25, y=15)
    caption_modeltraining2.place(x=25, y=55)
    caption_modeltraining3.place(x=25, y=90)
    folder_path_modeltraining.place(x=30, y=140)
    button_filedir_modeltraining.place(x=495, y=138)

    folder_path_modeltraining.config(state="normal")
    folder_path_modeltraining.delete(1.0, END)
    progress_modeltraining1.place_forget()
    progress_modeltraining2.place_forget()
    progressbar_modeltraining.place_forget()

    judul_designcost.place_forget()
    judul_estimation.place_forget()
    button_trainmodel.place_forget()
    button_estimatecost.place_forget()
    button_exit.place_forget()

    judul_modeltraining1_2.place_forget()
    caption_modeltraining1_2.place_forget()
    frame_modeltraining1_2.place_forget()
    button_done_modeltraining1_2.place_forget()
    button_back_modeltraining1_2.place_forget()
    judul_modeltraining2_2.place_forget()
    caption_modeltraining2_2.place_forget()
    caption_modeltraining3_2.place_forget()
    progressbar_modeltraining_2.place_forget()
    progress_modeltraining1_2.place_forget()
    progress_modeltraining2_2.place_forget()

def trainmodel2():
    judul_modeltraining1_2.place(x=25, y=20)
    caption_modeltraining1_2.place(x=25, y=60)
    frame_modeltraining1_2.place(x=20, y=100)
    button_done_modeltraining1_2.place(x=430, y=430)
    button_back_modeltraining1_2.place(x=330, y=430)
    judul_modeltraining2_2.place(x=25, y=15)
    caption_modeltraining2_2.place(x=25, y=55)
    caption_modeltraining3_2.place(x=25, y=80)
    progressbar_modeltraining_2.place(x=25, y=250)
    progress_modeltraining1_2.place(x=25, y=225)
    # progress_modeltraining2_2.place(x=25, y=225) #complete

    judul_modeltraining1.place_forget()
    caption_modeltraining1.place_forget()
    frame_modeltraining1.place_forget()
    button_cancel_modeltraining1.place_forget()
    button_run_modeltraining1.place_forget()
    button_back_modeltraining1.place_forget()
    judul_modeltraining2.place_forget()
    caption_modeltraining2.place_forget()
    caption_modeltraining3.place_forget()
    folder_path_modeltraining.place_forget()
    button_filedir_modeltraining.place_forget()

    progressbar_modeltraining.place_forget()
    progress_modeltraining1.place_forget()
    progress_modeltraining2.place_forget()

    max_progress_training = 10 #sama kaya split
    progress_training = 0
    progressbar_modeltraining_2["maximum"] = max_progress_training

    #train model
    df2 = pd.read_csv(path_csv)

    dataset2 = df2.values

    X2 = dataset2[:, 1:26]  # data dari 0 ke 9
    Y2 = dataset2[:, 26]

    min_max_scaler = preprocessing.MinMaxScaler()  # untuk bikin X dari range 0 sampai 1
    X2_scale = min_max_scaler.fit_transform(X2)  # menyimpan X antara 0 dan 1 dalam array

    # X_train, X_val_and_test, Y_train, Y_val_and_test = train_test_split(X_scale, Y, test_size=0.3)
    # X_val, X_test, Y_val, Y_test = train_test_split(X_val_and_test, Y_val_and_test, test_size=0.5)

    def create_model():
        model = Sequential()
        model.add(Dense(50, input_shape=(25,), activation='relu'))
        model.add(Dropout(0.4))
        model.add(Dense(50, activation='relu'))
        model.add(Dropout(0.4))
        model.add(Dense(50, activation='relu'))
        model.add(Dropout(0.4))
        model.add(Dense(25, activation='relu'))
        model.add(Dropout(0.4))
        model.add(Dense(25, activation='relu'))
        model.add(Dropout(0.4))
        model.add(Dense(1, activation='linear'))

        model.compile(Adam(lr=0.001),
                      loss='mean_squared_error',  # 'mean_squared_error', mean_absolute_percentage_error
                      metrics=[coeff_determination])  # [coeff_determination]) #['mape'])

        return model

    # define r2
    def coeff_determination(y_true, y_pred):
        SS_res = K.sum(K.square(y_true - y_pred))
        SS_tot = K.sum(K.square(y_true - K.mean(y_true)))
        return (1 - SS_res / (SS_tot + K.epsilon()))

    n_split = 10

    hasil = []
    array_score_akhir = []
    array_score_akhir_test = []

    for train_index, test_index in KFold(n_split).split(X2_scale):

        progress_training = progress_training + 1

        x_train, x_test = X2_scale[train_index], X2_scale[test_index]
        y_train, y_test = Y2[train_index], Y2[test_index]

        es = EarlyStopping(monitor='loss', mode='min', verbose=0)

        model = create_model()
        hist = model.fit(x_train, y_train,
                         batch_size=32, epochs=1000,
                         validation_data=(x_test, y_test), verbose=0)  # , callbacks=[es])

        # model.summary()

        y_pred = model.predict(x_train)
        y_test_pred = model.predict(x_test)

        score_akhir = r2_score(y_train, y_pred)
        score_akhir_test = r2_score(y_test, y_test_pred)
        array_score_akhir.append(r2_score(y_train, y_pred))
        array_score_akhir_test.append(r2_score(y_test, y_test_pred))

        time.sleep(0.05)
        progressbar_modeltraining_2["value"] = progress_training
        progressbar_modeltraining_2.update()

        if progress_training == max_progress_training:
            progress_modeltraining2_2.place(x=25, y=225)

        else:
            progress_modeltraining1_2.place(x=25, y=225)


    def hitung_rata_rata_train():  # untuk menghitung mean R2 train
        a = 0
        for i in array_score_akhir:
            a = a + i
        mean_score_akhir_train = a / len(array_score_akhir)
        return mean_score_akhir_train

    def hitung_rata_rata_test():
        a = 0
        for i in array_score_akhir_test:
            a = a + i
        mean_score_akhir_test = a / len(array_score_akhir_test)
        return mean_score_akhir_test

    mean_r2_train = hitung_rata_rata_train()
    mean_r2_test = hitung_rata_rata_test()

    print("Mean R2 Train : ", mean_r2_train)
    print("Mean R2 Test : ", mean_r2_test)

    #save model
    model_json = model.to_json()
    with open(path_arsitektur, "w") as json_file_1:
        json_file_1.write(model_json)
    # serialize weights to HDF5
    model.save_weights(path_weight)
    print("Saved model to disk")

    score_modeltraining_2 = Label(frame_modeltraining1_2, text="R2 Score : "+str("%.2f" % mean_r2_test), justify=CENTER, font="Verdana 12 bold", height=1)
    score_modeltraining_2.place(x=200, y=150)

def estimatecost():
    judul_estimatecost1.place(x=25, y=20)
    caption_estimatecost1.place(x=25, y=60)
    frame_estimatecost1.place(x=20, y=100)
    button_cancel_estimatecost1.place(x=529, y=430)
    button_next_estimatecost1.place(x=430, y=430)
    button_back_estimatecost1.place(x=330, y=430)
    judul_estimatecost2.place(x=25, y=15)
    caption_estimatecost2.place(x=25, y=55)
    caption_estimatecost4.place(x=25, y=80)
    caption_estimatecost3.place(x=25, y=120)
    folder_path_estimatecost1.place(x=30, y=150)
    button_filedir_estimatecost1.place(x=495, y=148)

    #hapus widget home
    judul_designcost.place_forget()
    judul_estimation.place_forget()
    button_trainmodel.place_forget()
    button_estimatecost.place_forget()
    button_exit.place_forget()

    #hapus widget estimate cost 2(Tombol next dari estimate cost 1)
    judul_estimatecost1_2.place_forget()
    caption_estimatecost1_2.place_forget()
    frame_estimatecost1_2.place_forget()
    button_cancel_estimatecost1_2.place_forget()
    button_next_estimatecost1_2.place_forget()
    button_back_estimatecost1_2.place_forget()
    judul_estimatecost2_2.place_forget()
    folder_path_estimatecost1_2.place_forget()
    button_filedir_estimatecost1_2.place_forget()

def estimatecost2():

    judul_estimatecost1_2.place(x=25, y=20)
    caption_estimatecost1_2.place(x=25, y=60)
    frame_estimatecost1_2.place(x=20, y=100)
    button_cancel_estimatecost1_2.place(x=529, y=430)
    button_next_estimatecost1_2.place(x=430, y=430)
    button_back_estimatecost1_2.place(x=330, y=430)
    judul_estimatecost2_2.place(x=10, y=15)
    folder_path_estimatecost1_2.place(x=30, y=50)
    button_filedir_estimatecost1_2.place(x=495, y=48)

    judul_estimatecost1.place_forget()
    caption_estimatecost1.place_forget()
    frame_estimatecost1.place_forget()
    button_cancel_estimatecost1.place_forget()
    button_next_estimatecost1.place_forget()
    button_back_estimatecost1.place_forget()
    judul_estimatecost2.place_forget()
    caption_estimatecost2.place_forget()
    caption_estimatecost3.place_forget()
    folder_path_estimatecost1.place_forget()
    button_filedir_estimatecost1.place_forget()
    caption_estimatecost4.place_forget()

def loadfolder_modeltraining(): #buat model training

    global tanggal
    tanggal = datetime.now().strftime('%Y-%m-%d %H.%M.%S')

    folder_path_modeltraining.config(state="normal")
    folder_path_modeltraining.delete(1.0, END)
    folderpath = filedialog.askdirectory(parent=root, initialdir="/", title='Please select a directory')
    folder_path_modeltraining.insert(END, folderpath)
    folder_path_modeltraining.config(state="disable")

    nama_folder = os.path.basename(folderpath)

    global file_csv
    global path_csv
    global path_arsitektur
    global path_weight
    global path_variablemax

    file_csv = folder_path_modeltraining.get("0.0",END).rstrip()

    os.mkdir(str(file_csv)+"/Model " + str(nama_folder)  + " " + str(tanggal))

    path_csv = str(file_csv) + "/Model " + str(nama_folder) + " " + str(tanggal)+ "/Training File " + str(nama_folder) + " " + str(tanggal) + ".csv"
    path_arsitektur = path_csv[:-4] + " (1).json"
    path_weight = path_csv[:-3] + "h5"
    path_variablemax = path_csv[:-4] + " (2).json"

    print(path_csv)
    print(path_arsitektur)
    print(path_weight)
    print(path_variablemax)

    if folder_path_modeltraining.compare("end-1c", "==", "1.0"):
        pass

    else:
        save_dataset_csv()

        df1 = pd.read_csv(path_csv)

        dataset1 = df1.values

        X = dataset1[:, 1:26]  # data dari 0 ke 9
        Y = dataset1[:, 26]

        untuk_simpan_max = pd.DataFrame(X)
        array_nilai_max = []
        for i in range(0, 25):  # menghitung nilai max dari dataset, untuk scaling
            fitur_max = untuk_simpan_max[untuk_simpan_max.columns[i]].max()

            if fitur_max == 0:
                fitur_max = fitur_max + 1

            else:
                pass

            array_nilai_max.append(fitur_max)

        with open(path_variablemax, "w") as json_file_2:
            json.dump(array_nilai_max, json_file_2)

def loadfolder_estimatecost1(): #buat arsitektur

    folder_path_estimatecost1.config(state="normal")
    folder_path_estimatecost1.delete(1.0, END)
    folder_path_model_estimasi = filedialog.askdirectory(parent=root, initialdir="/", title='Please select a directory')
    folder_path_estimatecost1.insert(END, folder_path_model_estimasi)
    folder_path_estimatecost1.config(state="disable")

    global file_model_arsitektur_estimasi
    global file_model_weigth_estimasi
    global file_model_variable_estimasi

    nama_folder_model_estimasi = os.path.basename(folder_path_model_estimasi)

    tanggal_model_estimasi = nama_folder_model_estimasi[-19:]

    file_model_arsitektur_estimasi = folder_path_model_estimasi + "/Training File " + str(nama_folder_model_estimasi[6:]) + " (1).json"
    file_model_weigth_estimasi = folder_path_model_estimasi + "/Training File " + str(nama_folder_model_estimasi[6:]) + ".h5"
    file_model_variable_estimasi = folder_path_model_estimasi + "/Training File " + str(nama_folder_model_estimasi[6:]) + " (2).json"

    # print(folder_path_model_estimasi)
    # print(file_model_arsitektur_estimasi)
    # print(file_model_weigth_estimasi)
    # print(file_model_variable_estimasi)

def loadfolder_estimatecost3():

        #load model, weight, dan variable max

    # load json and create model
    json_file1 = open(file_model_arsitektur_estimasi, 'r')
    loaded_model_json1 = json_file1.read()
    json_file1.close()
    loaded_model = model_from_json(loaded_model_json1)

    # load weights into new model
    loaded_model.load_weights(file_model_weigth_estimasi)

    #load variable max
    with open(file_model_variable_estimasi) as json_file2:
        nilai_max_tiap_fitur = json.load(json_file2)

    # print(nilai_max_tiap_fitur)
    print("Model Loaded")

    folder_path_estimatecost1_2.config(state="normal")
    folder_path_estimatecost1_2.delete(1.0, END)
    folderpath_sldprt = filedialog.askopenfilename(parent=root, initialdir="/", title='Please select a file',
                                                   filetypes=(("Solidworks Part", "*.sldprt"), ("All Files", "*.*")))
    folder_path_estimatecost1_2.insert(END, folderpath_sldprt)
    folder_path_estimatecost1_2.config(state="disable")

    if folder_path_estimatecost1_2.compare("end-1c", "==", "1.0"):
        pass

    else:
        SW_estimate = folder_path_estimatecost1_2.get("0.0",END).rstrip()

        # print(SW_estimate)
        swModel = sw.OpenDoc(SW_estimate, 1)

        swModel.ShowNamedView2("*Isometric", -1)
        swModel.ViewZoomtofit2()

        saveFileFoto = SW_estimate[:-6] + "JPG"
        swModel.SaveAs3(saveFileFoto, 0, 0)

        modelName = swModel.GetTitle

        foto = Image.open(saveFileFoto)
        foto = foto.resize((350, 201), Image.ANTIALIAS)
        foto.save(SW_estimate[:-6] + "ppm", "ppm")
        FOTO2 = ImageTk.PhotoImage(file=SW_estimate[:-6]+"ppm") #2

        panel = Label(frame_estimatecost1_2, image=FOTO2)
        panel.image = FOTO2
        panel.place(x=120, y=75)

        #untuk mengetahui fitur2 yang ada pada part yang akan diestimasi
        swCustPropMgr = swModel.Extension.CustomPropertyManager("")

        biaya = swCustPropMgr.GetNames

        swFeatMgr = swModel.FeatureManager

        swFeatStat = swFeatMgr.FeatureStatistics

        jumlah_fitur = 0
        sketchcount = 0
        bossextrudecount = 0
        revolvecount = 0
        sweepcount = 0
        loftcount = 0
        boundarycount = 0
        cutextrudecount = 0
        cutrevolvecount = 0
        cutsweepcount = 0
        cutloftcount = 0
        boundarycutcount = 0
        filletcount = 0
        chamfercount = 0
        lpatterncount = 0
        cirpatterncount = 0
        mirrorcount = 0
        ribcount = 0
        draftcount = 0
        shellcount = 0
        wrapcount = 0
        intersectcount = 0
        domecount = 0
        pointcount = 0
        linecount = 0
        arccount = 0
        ellipscount = 0
        parabolacount = 0
        holecount = 0
        cborecount = 0

        arg1 = win32com.client.VARIANT(pythoncom.VT_DISPATCH, None)

        featnames = swFeatStat.FeatureNames
        feattypes = swFeatStat.FeatureTypes
        features = swFeatStat.features
        featureUpdateTimes = swFeatStat.featureUpdateTimes
        featureUpdatePercentTimes = swFeatStat.FeatureUpdatePercentageTimes

        feature_type_name = ['Nama_Part', 'Line_Num', 'Arc_Num', 'Ellips_Num', 'Boss_Extrude_Num', 'Revolve_Num',
                      'Sweep_Num', 'Loft_Num', 'Boundary_Num', 'Cut_Extrude_Num', 'Cut_Revolve_Num', 'Cut_Sweep_Num',
                      'Cut_Loft_Num', 'Boundary_Cut_Num', 'Fillet_Num', 'Chamfer_Num', 'LPattern_Num', 'CirPattern_Num',
                      'Mirror_Num', 'Rib_Num', 'Draft_Num', 'Shell_Num', 'Wrap_Num', 'Intersect_Num', 'Dome_Num',
                      'Hole_Num']  # semua fitur pada sw jadi array feature_type_name

        for x in range(0, swFeatStat.FeatureCount):
            if not featnames:  # kalo swDim kosong
                pass

            elif "Sketch" in featnames[x]:
                sketchcount = sketchcount + 1

            elif "Boss-Extrude" in featnames[x]:
                bossextrudecount = bossextrudecount + 1

            elif "Cut-Revolve" in featnames[x]:
                cutrevolvecount = cutrevolvecount + 1

            elif "Cut-Sweep" in featnames[x]:
                cutsweepcount = cutsweepcount + 1

            elif "Cut-Loft" in featnames[x]:
                cutloftcount = cutloftcount + 1

            elif "Boundary-Cut" in featnames[x]:
                boundarycutcount = boundarycutcount + 1

            elif "Cut-Extrude" in featnames[x]:
                cutextrudecount = cutextrudecount + 1

            elif "Revolve" in featnames[x]:
                revolvecount = revolvecount + 1

            elif "Sweep" in featnames[x]:
                sweepcount = sweepcount + 1

            elif "Loft" in featnames[x]:
                loftcount = loftcount + 1

            elif "Boundary" in featnames[x]:
                boundarycount = boundarycount + 1

            elif "Fillet" in featnames[x]:
                filletcount = filletcount + 1

            elif "Chamfer" in featnames[x]:
                chamfercount = chamfercount + 1

            elif "LPattern" in featnames[x]:
                lpatterncount = lpatterncount + 1

            elif "CirPattern" in featnames[x]:
                cirpatterncount = cirpatterncount + 1

            elif "Mirror" in featnames[x]:
                mirrorcount = mirrorcount + 1

            elif "Rib" in featnames[x]:
                ribcount = ribcount + 1

            elif "Draft" in featnames[x]:
                draftcount = draftcount + 1

            elif "Shell" in featnames[x]:
                shellcount = shellcount + 1

            elif "Wrap" in featnames[x]:
                wrapcount = wrapcount + 1

            elif "Intersect" in featnames[x]:
                intersectcount = intersectcount + 1

            elif "Dome" in featnames[x]:
                domecount = domecount + 1

            elif "Hole" in featnames[x]:
                holecount = holecount + 1

            elif "CBORE" in featnames[x]:
                holecount = holecount + 1

        for x in range(0, swFeatStat.FeatureCount):  # untuk mencari fitur yang merupakan sketch

            if "Sketch" in featnames[x]:
                # print("Fitur yang sketch adalah fitur ke-" + str(x) + " yaitu " + featnames[x])

                boolstatus = swModel.Extension.SelectByID2(featnames[x], "SKETCH", 0, 0, 0, False, 0, arg1,
                                                           0)  # mendefinisikan arg1 dulu diatas

                if boolstatus == True:
                    swSelMgr = swModel.SelectionManager
                    swFeat = swSelMgr.GetSelectedObject6(1, -1)  # memilih fitur yang sesuai dengan boolstatus
                    # print(swFeat.Name)
                    swSketch = swFeat.GetSpecificFeature2  # harus ada yang diselect di activedocnya

                    PointNum = swSketch.GetSketchPointsCount2
                    LineNum = swSketch.GetLineCount2(1)  # menghitung jumlah linenya
                    ArcNum = swSketch.GetArcCount
                    EllipsNum = swSketch.GetEllipseCount
                    ParabolaNum = swSketch.GetParabolaCount
                    # PolyNum = swSketch.GetPolyLineCount5(1)
                    # SplineNum = swSketch.GetSplineCount()

                    pointcount = pointcount + PointNum
                    linecount = linecount + LineNum
                    arccount = arccount + ArcNum
                    ellipscount = ellipscount + EllipsNum
                    parabolacount = parabolacount + ParabolaNum

            else:
                pass

        for biayareal in biaya:
            # print("Biaya untuk " + str(swFeatStat.PartName) + " adalah " + str(biayareal))
            pass

        array_estimasi_sw = []

        fitur_sw = [linecount, arccount, ellipscount, bossextrudecount,
                    revolvecount, sweepcount,
                    loftcount, boundarycount, cutextrudecount, cutrevolvecount, cutsweepcount, cutloftcount,
                    boundarycutcount, filletcount,
                    chamfercount, lpatterncount, cirpatterncount, mirrorcount, ribcount, draftcount, shellcount,
                    wrapcount, intersectcount,
                    domecount, holecount]

        for i in fitur_sw:
            array_estimasi_sw.append(i)

        array_hasil = [float(a) / float(b) for a, b in zip(array_estimasi_sw, nilai_max_tiap_fitur)]

        # print(array_estimasi_sw)
        # print(nilai_max_tiap_fitur)
        # print(array_hasil)

        array_final = array([array_hasil])

        print(array_final)

        harga_prediksi = loaded_model.predict(array_final)

        print(harga_prediksi)

        # string_harga_prediksi = str(harga_prediksi)
        #
        # tampilkan_harga_prediksi = string_harga_prediksi[2:-2]

        judul_estimatecost3_2 = Label(frame_estimatecost1_2, text="Rp. " + str(int(harga_prediksi)), justify=CENTER, font="Verdana 11 bold", height=1, width=13)

        judul_estimatecost3_2.place(x=210, y=280)

        modelName = swModel.GetTitle
        sw.CloseDoc(modelName)

def exit():
    root.destroy()

def mengetahuikoordinat():
    # mengetahui koordinat widget
    root.update()
    a = judul_designcost.winfo_x()
    b = judul_designcost.winfo_y()

    print(a)
    print(b)

def save_dataset_csv():

    progressbar_modeltraining.place(x=25, y=250)

    os.chdir(file_csv)  # mendapatkan list file pada folder COBACOBA

    FileSW = [os.path.abspath(x) for x in os.listdir('.') if ".SLDPRT" in os.path.abspath(x) or ".sldprt" in os.path.abspath(x)]  # menghasilkan full path name

    max_progress_csv = len(FileSW)
    progress_csv = 0
    progressbar_modeltraining["maximum"] = max_progress_csv

    with open(path_csv, 'w', newline='') as new_file:
        fieldnames = ['Nama_Part', 'Line_Num', 'Arc_Num', 'Ellips_Num', 'Boss_Extrude_Num', 'Revolve_Num',
                      'Sweep_Num', 'Loft_Num', 'Boundary_Num', 'Cut_Extrude_Num', 'Cut_Revolve_Num', 'Cut_Sweep_Num',
                      'Cut_Loft_Num', 'Boundary_Cut_Num', 'Fillet_Num', 'Chamfer_Num', 'LPattern_Num', 'CirPattern_Num',
                      'Mirror_Num', 'Rib_Num', 'Draft_Num', 'Shell_Num', 'Wrap_Num', 'Intersect_Num', 'Dome_Num',
                      'Hole_Num', 'Biaya']

        csv_writer = csv.DictWriter(new_file, fieldnames=fieldnames)
        csv_writer.writeheader()

        for TrainingFile in FileSW:
            if "~$" in TrainingFile:
                pass

            elif ".csv" in TrainingFile:
                pass

            else:
                progress_csv = progress_csv + 1
                print(TrainingFile)

                swModel = sw.OpenDoc(TrainingFile, 1)

                swCustPropMgr = swModel.Extension.CustomPropertyManager("")

                biaya = swCustPropMgr.GetNames

                swFeatMgr = swModel.FeatureManager

                swFeatStat = swFeatMgr.FeatureStatistics

                jumlah_fitur = 0
                sketchcount = 0
                bossextrudecount = 0
                revolvecount = 0
                sweepcount = 0
                loftcount = 0
                boundarycount = 0
                cutextrudecount = 0
                cutrevolvecount = 0
                cutsweepcount = 0
                cutloftcount = 0
                boundarycutcount = 0
                filletcount = 0
                chamfercount = 0
                lpatterncount = 0
                cirpatterncount = 0
                mirrorcount = 0
                ribcount = 0
                draftcount = 0
                shellcount = 0
                wrapcount = 0
                intersectcount = 0
                domecount = 0
                pointcount = 0
                linecount = 0
                arccount = 0
                ellipscount = 0
                parabolacount = 0
                holecount = 0
                cborecount = 0

                arg1 = win32com.client.VARIANT(pythoncom.VT_DISPATCH, None)

                print("")
                # print("Model name:                 ", swFeatStat.PartName)

                featnames = swFeatStat.FeatureNames
                feattypes = swFeatStat.FeatureTypes
                features = swFeatStat.features
                featureUpdateTimes = swFeatStat.featureUpdateTimes
                featureUpdatePercentTimes = swFeatStat.FeatureUpdatePercentageTimes

                feature_type_name = ["Sketch", "Boss-Extrude", "Revolve", "Revolved-Thin", "Sweep", "Loft", "Boundary",
                                     "Cut-Extrude", "Cut-Revolve", "Cut-Sweep",
                                     "Cut-Loft", "Boundary-Cut", "Fillet", "Chamfer", "LPattern", "CirPattern",
                                     "Mirror", "Rib", "Draft", "Shell", "Wrap", "Intersect",
                                     "Dome", "Hole", "CBORE"]  # semua fitur pada sw jadi array feature_type_name

                for x in range(0, swFeatStat.FeatureCount):
                    if not featnames:  # kalo swDim kosong
                        pass

                    elif "Sketch" in featnames[x]:
                        sketchcount = sketchcount + 1

                    elif "Boss-Extrude" in featnames[x]:
                        bossextrudecount = bossextrudecount + 1

                    elif "Cut-Revolve" in featnames[x]:
                        cutrevolvecount = cutrevolvecount + 1

                    elif "Cut-Sweep" in featnames[x]:
                        cutsweepcount = cutsweepcount + 1

                    elif "Cut-Loft" in featnames[x]:
                        cutloftcount = cutloftcount + 1

                    elif "Boundary-Cut" in featnames[x]:
                        boundarycutcount = boundarycutcount + 1

                    elif "Cut-Extrude" in featnames[x]:
                        cutextrudecount = cutextrudecount + 1

                    elif "Revolve" in featnames[x]:
                        revolvecount = revolvecount + 1

                    elif "Sweep" in featnames[x]:
                        sweepcount = sweepcount + 1

                    elif "Loft" in featnames[x]:
                        loftcount = loftcount + 1

                    elif "Boundary" in featnames[x]:
                        boundarycount = boundarycount + 1

                    elif "Fillet" in featnames[x]:
                        filletcount = filletcount + 1

                    elif "Chamfer" in featnames[x]:
                        chamfercount = chamfercount + 1

                    elif "LPattern" in featnames[x]:
                        lpatterncount = lpatterncount + 1

                    elif "CirPattern" in featnames[x]:
                        cirpatterncount = cirpatterncount + 1

                    elif "Mirror" in featnames[x]:
                        mirrorcount = mirrorcount + 1

                    elif "Rib" in featnames[x]:
                        ribcount = ribcount + 1

                    elif "Draft" in featnames[x]:
                        draftcount = draftcount + 1

                    elif "Shell" in featnames[x]:
                        shellcount = shellcount + 1

                    elif "Wrap" in featnames[x]:
                        wrapcount = wrapcount + 1

                    elif "Intersect" in featnames[x]:
                        intersectcount = intersectcount + 1

                    elif "Dome" in featnames[x]:
                        domecount = domecount + 1

                    elif "Hole" in featnames[x]:
                        holecount = holecount + 1

                    elif "CBORE" in featnames[x]:
                        holecount = holecount + 1

                for x in range(0, swFeatStat.FeatureCount):  # untuk mencari fitur yang merupakan sketch

                    if "Sketch" in featnames[x]:
                        # print("Fitur yang sketch adalah fitur ke-" + str(x) + " yaitu " + featnames[x])

                        boolstatus = swModel.Extension.SelectByID2(featnames[x], "SKETCH", 0, 0, 0, False, 0, arg1,
                                                                   0)  # mendefinisikan arg1 dulu diatas

                        if boolstatus == True:
                            swSelMgr = swModel.SelectionManager
                            swFeat = swSelMgr.GetSelectedObject6(1, -1)  # memilih fitur yang sesuai dengan boolstatus
                            # print(swFeat.Name)
                            swSketch = swFeat.GetSpecificFeature2  # harus ada yang diselect di activedocnya

                            PointNum = swSketch.GetSketchPointsCount2
                            LineNum = swSketch.GetLineCount2(1)  # menghitung jumlah linenya
                            ArcNum = swSketch.GetArcCount
                            EllipsNum = swSketch.GetEllipseCount
                            ParabolaNum = swSketch.GetParabolaCount
                            # PolyNum = swSketch.GetPolyLineCount5(1)
                            # SplineNum = swSketch.GetSplineCount()

                            pointcount = pointcount + PointNum
                            linecount = linecount + LineNum
                            arccount = arccount + ArcNum
                            ellipscount = ellipscount + EllipsNum
                            parabolacount = parabolacount + ParabolaNum

                    else:
                        pass

                for biayareal in biaya:
                    print("Biaya untuk " + str(swFeatStat.PartName) + " adalah " + str(biayareal))

                csv_writer.writerow({'Nama_Part': swFeatStat.PartName, 'Line_Num': linecount,
                                     'Arc_Num': arccount, 'Ellips_Num': ellipscount,
                                     'Boss_Extrude_Num': bossextrudecount,
                                     'Revolve_Num': revolvecount, 'Sweep_Num': sweepcount, 'Loft_Num': loftcount,
                                     'Boundary_Num': boundarycount, 'Cut_Extrude_Num': cutextrudecount,
                                     'Cut_Revolve_Num': cutrevolvecount, 'Cut_Sweep_Num': cutsweepcount,
                                     'Cut_Loft_Num': cutloftcount, 'Boundary_Cut_Num': boundarycutcount,
                                     'Fillet_Num': filletcount, 'Chamfer_Num': chamfercount,
                                     'LPattern_Num': lpatterncount, 'CirPattern_Num': cirpatterncount,
                                     'Mirror_Num': mirrorcount, 'Rib_Num': ribcount, 'Draft_Num': draftcount,
                                     'Shell_Num': shellcount, 'Wrap_Num': wrapcount, 'Intersect_Num': intersectcount,
                                     'Dome_Num': domecount, 'Hole_Num': holecount, 'Biaya': biayareal})

                sketchcount = 0
                bossextrudecount = 0
                revolvecount = 0
                sweepcount = 0
                loftcount = 0
                boundarycount = 0
                cutextrudecount = 0
                cutrevolvecount = 0
                cutsweepcount = 0
                cutloftcount = 0
                boundarycutcount = 0
                filletcount = 0
                chamfercount = 0
                lpatterncount = 0
                cirpatterncount = 0
                mirrorcount = 0
                ribcount = 0
                draftcount = 0
                shellcount = 0
                wrapcount = 0
                intersectcount = 0
                domecount = 0
                holecount = 0
                cborecount = 0

                pointcount = 0
                linecount = 0
                arccount = 0
                ellipscount = 0
                parabolacount = 0

                modelName = swModel.GetTitle
                sw.CloseDoc(modelName)

            time.sleep(0.05)
            progressbar_modeltraining["value"] = progress_csv
            progressbar_modeltraining.update()

            if progress_csv == max_progress_csv:
                progress_modeltraining2.place(x=25, y=225)

            else:
                progress_modeltraining1.place(x=25, y=225)
        # progressbar_modeltraining["value"] = 0

#widget Home
judul_designcost = Label(root, text="DESIGN COST", justify=CENTER, font="Verdana 20 bold", width=11, height=1)
judul_estimation = Label(root, text="ESTIMATION", justify=CENTER, font="Verdana 20 bold", width=11, height=1)
button_trainmodel = Button(root, text="Train Model", justify=CENTER, font="Verdana 14", height=1, width=13, command=trainmodel)
button_estimatecost = Button(root, text="Estimate Cost", justify=CENTER, font="Verdana 14", height=1, width=13, command=estimatecost)
button_exit = Button(root, text="Exit", justify=CENTER, font="Verdana 14", height=1, width=13, command=exit)

#widget Model Training Load Folder Training
judul_modeltraining1 = Label(root, text="Model Training", justify=CENTER, font="Verdana 18 bold")
caption_modeltraining1 = Label(root, text="Train a model that can use for estimate design cost.", justify=LEFT, font="Verdana 9")
frame_modeltraining1 = Frame(root, width=600, height=310, borderwidth=1, relief=SUNKEN)
button_cancel_modeltraining1 = Button(root, text="Cancel", justify=CENTER, font="Verdana 9", height=1, width=9, command=home)
button_run_modeltraining1 = Button(root, text="Run", justify=CENTER, font="Verdana 9", height=1, width=9, command=trainmodel2)
button_back_modeltraining1 = Button(root, text="< Back", justify=CENTER, font="Verdana 9", height=1, width=9, command=home)
judul_modeltraining2 = Label(frame_modeltraining1, text="Select a Folder", justify=CENTER, font="Verdana 14 bold", height=1, width=13)
caption_modeltraining2 = Label(frame_modeltraining1, text="All the files in this folder will be the dataset for training this model.", justify=CENTER, font="Verdana 9", height=1)
caption_modeltraining3 = Label(frame_modeltraining1, text="Folder Path:", justify=CENTER, font="Verdana 9 bold", height=1)
folder_path_modeltraining = Text(frame_modeltraining1, font="Verdana 9", width=50, height=1)
button_filedir_modeltraining = Button(frame_modeltraining1, text="...", justify=CENTER, font="Verdana 7", command=loadfolder_modeltraining, height=1)
progressbar_modeltraining = ttk.Progressbar(frame_modeltraining1, orien="horizontal", length=550, mode="determinate")
progress_modeltraining1 = Label(frame_modeltraining1, text="Creating CSV File...", justify=CENTER, font="Verdana 9 bold", height=1)
progress_modeltraining2 = Label(frame_modeltraining1, text="Complete                  ", justify=CENTER, font="Verdana 9 bold", height=1)

#widget Cost Estimation Load Model
judul_estimatecost1 = Label(root, text="Cost Estimation", justify=CENTER, font="Verdana 18 bold")
caption_estimatecost1 = Label(root, text="Estimate SolidWorks design cost from selected model.", justify=LEFT, font="Verdana 9")
frame_estimatecost1 = Frame(root, width=600, height=310, borderwidth=1, relief=SUNKEN)
button_cancel_estimatecost1 = Button(root, text="Cancel", justify=CENTER, font="Verdana 9", height=1, width=9, command=home)
button_next_estimatecost1 = Button(root, text="Next", justify=CENTER, font="Verdana 9", height=1, width=9, command=estimatecost2)
button_back_estimatecost1 = Button(root, text="< Back", justify=CENTER, font="Verdana 9", height=1, width=9, command=home)
judul_estimatecost2 = Label(frame_estimatecost1, text="Select a Model", justify=CENTER, font="Verdana 14 bold", height=1, width=13)
caption_estimatecost2 = Label(frame_estimatecost1, text="Select Folder that contains model architecture, model weight, and", justify=CENTER, font="Verdana 9", height=1)
caption_estimatecost4 = Label(frame_estimatecost1, text="stored variable.", justify=CENTER, font="Verdana 9", height=1)
caption_estimatecost3 = Label(frame_estimatecost1, text="Folder Path: ", justify=CENTER, font="Verdana 9 bold", height=1)
folder_path_estimatecost1 = Text(frame_estimatecost1, font="Verdana 9", width=50, height=1)
button_filedir_estimatecost1 = Button(frame_estimatecost1, text="...", justify=CENTER, font="Verdana 7", command=loadfolder_estimatecost1, height=1)

#widget Cost Estimation Estimate Price
judul_estimatecost1_2 = Label(root, text="Cost Estimation", justify=CENTER, font="Verdana 18 bold")
caption_estimatecost1_2 = Label(root, text="Estimate SolidWorks design cost from selected model.", justify=LEFT, font="Verdana 9")
frame_estimatecost1_2 = Frame(root, width=600, height=310, borderwidth=1, relief=SUNKEN)
button_cancel_estimatecost1_2 = Button(root, text="Cancel", justify=CENTER, font="Verdana 9", height=1, width=9, command=home)
button_next_estimatecost1_2 = Button(root, text="Done", justify=CENTER, font="Verdana 9", height=1, width=9, command=home)
button_back_estimatecost1_2 = Button(root, text="< Back", justify=CENTER, font="Verdana 9", height=1, width=9, command=estimatecost)
judul_estimatecost2_2 = Label(frame_estimatecost1_2, text="Select a File", justify=CENTER, font="Verdana 14 bold", height=1, width=13)
folder_path_estimatecost1_2 = Text(frame_estimatecost1_2, font="Verdana 9", width=50, height=1)
button_filedir_estimatecost1_2 = Button(frame_estimatecost1_2, text="...", justify=CENTER, font="Verdana 7", command=loadfolder_estimatecost3, height=1)



#widget Model Training Proses Training
judul_modeltraining1_2 = Label(root, text="Model Training", justify=CENTER, font="Verdana 18 bold")
caption_modeltraining1_2 = Label(root, text="Train a model that can use for estimate design cost.", justify=LEFT, font="Verdana 9")
frame_modeltraining1_2 = Frame(root, width=600, height=310, borderwidth=1, relief=SUNKEN)
button_done_modeltraining1_2 = Button(root, text="Done", justify=CENTER, font="Verdana 9", height=1, width=9, command=home)
button_back_modeltraining1_2 = Button(root, text="< Back", justify=CENTER, font="Verdana 9", height=1, width=9, command=trainmodel)
judul_modeltraining2_2 = Label(frame_modeltraining1_2, text="Create a Model", justify=CENTER, font="Verdana 14 bold", height=1, width=13)
caption_modeltraining2_2 = Label(frame_modeltraining1_2, text="The model is trained using previous dataset. Trained model may have", justify=CENTER, font="Verdana 9", height=1)
caption_modeltraining3_2 = Label(frame_modeltraining1_2, text="better performance rathen than shown below.", justify=CENTER, font="Verdana 9", height=1)
progressbar_modeltraining_2 = ttk.Progressbar(frame_modeltraining1_2, orien="horizontal", length=550, mode="determinate")
progress_modeltraining1_2 = Label(frame_modeltraining1_2, text="Training Model...", justify=CENTER, font="Verdana 9 bold", height=1)
progress_modeltraining2_2 = Label(frame_modeltraining1_2, text="Complete               ", justify=CENTER, font="Verdana 9 bold", height=1)


import os
import win32com.client
import pythoncom
import csv
from PIL import ImageTk, Image
import time
import pandas as pd
import json
from sklearn import preprocessing
from keras import backend as K
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.optimizers import Adam
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from keras.callbacks import EarlyStopping
from keras.models import model_from_json
from numpy import array

swYearLastDigit = 8
sw = win32com.client.Dispatch("SldWorks.Application")

home()
# trainmodel()
root.mainloop() #biar windownya ga ke close

os.system('TASKKILL /F /IM "SLDWORKS.exe"')