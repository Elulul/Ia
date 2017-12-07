# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 10:04:19 2017

@author: Cedric
"""

# import xlrd Useless
import pandas as pd
import numpy as np
#import tkFileDialog

import matplotlib.pyplot as plt
#
import numpy as np
# import matplotlib.pyplot as plt
import scipy.fftpack
import sys



#####python 3
import tkinter.filedialog as tkFileDialog


#return the dataframe for the file named filename

def from_xlsx_to_df(filename):

     """
     This function takes as input the excel file, parses it and convrert relevant informations into a dataFrame. 
     :param filename: the path of the excel file 
    
    """

    # ouverture du fichier Excel
    # filename = "/home/photon/Documents/Education/3A/PI/data/0210/17100203374_081_20171003_024259.xlsx"
    # feuilles dans le classeur
   print( "filename obtained : " + filename)                                                #affichage du fichier courant
   xl = pd.ExcelFile(filename)                                                      #ouverture panda du fichier courant
   df = xl.parse("Sheet1", skiprows=13)                                               #parsing panda du fichier courant
   print("first 13 rows skipped")
   df = df[['Unnamed: 0', 'ACC1 fast_descript', 'ACC2 fast_descript']]  #selection des colones d'interet : le temps e                                                                                           # les deux accelerometres
   df.columns = ["time", "ACC1", "ACC2"]  #renommage des colonnes selectionnées
   print("columns separated")
   return df

#Argument ACC should be either ACC1 or ACC2, return xf & yf
def from_df_to_fft(df,interval,ACC):
    """
    Compute the fft of the value from an intervval and from one accelerometre
    :param df : the dataframe
    :param interval : l'intervallle de temps sur lequel on veut faire la fft (liste de 2 valeurs)
    :param AAC: l'acceloremetre qu'on veut prendre en compte (string "ACC1" or "ACC2")
    :return : yf la fft sur les valeurs d'accelorometre et xf le temps passé en fréquuence
    """
    # Number of samplepoints
    # sample spacing
    # x = np.linspace(0.0, N*T, N)
    # x = df.time[:quarterLength]
    # abscisse des frequences
#    plt.figure()                                                                    # Creation de la figure pour plotter
    # --------------ACC1 : Transformee de Fourier------------------------
    deltaT = df.time[1]
    xf = np.linspace(0.0, 1.0 / (2.0 * deltaT), ((interval[1]-interval[0])/deltaT) // 2)
    if(ACC == "ACC1"):
        y = df.ACC1[interval[0]  : interval[1]]      # mesures d'accelerometre 1, quart 1
        yf = scipy.fftpack.fft(y)
    else:
        if(ACC == "ACC2"):
            y = df.ACC2[interval[0] : interval[1]]      # mesures d'accelerometre 1, quart 1
            yf = scipy.fftpack.fft(y)
        else:
            return "error"
    return xf,yf

# return TOctave,dbPerTOctave
def from_fft_to_third_octave(xf,yf,quarterLength,initOctaveSize):
    N = quarterLength
    yf_calc = 2.0 / N * np.abs(yf[:N // 2])
    numOctaves = 0

    maxfreq = xf[len(xf) - 1]
    tempMaxFreq = 0
    while (tempMaxFreq < maxfreq):
        numOctaves += 1
        tempMaxFreq += initOctaveSize * (2 ** numOctaves)
    dbPerTOctave = []
    TOctave = []
    sup = 11.048543456+initOctaveSize / 3;
    octaveWidth = initOctaveSize
    currentSum = 0
    entriesUsed = 0
    infxval = xf[1]
    i = 1
    while xf[i] < 11.048543456: #on commence le premier octave à 11 Hertz pour être centré sur 15.625 Hertz
        i+=1

    while i < len(xf):
        if xf[i] < sup:
            currentSum += yf_calc[i]
            entriesUsed += 1
        else:
            dbPerTOctave.append(currentSum / entriesUsed)
            TOctave.append((infxval + xf[i]) / 2)
            currentSum = yf_calc[i]
            entriesUsed = 1

            if (len(dbPerTOctave) % 3) == 0:  # si on passe à l'octave suivant, on double la largeur d'un octave
                octaveWidth = 2 * octaveWidth
                # print "octave width: "+str(octaveWidth) + "\n"
                # print "tiers d'octave number: "+str(len(dbPerTOctave)) + "\n"

            infxval = xf[i]
            # print "sup :" +str(sup)
            # print "1/3 * octaveWidth :" +str(octaveWidth/3)
            sup = infxval + octaveWidth / 3
            # print "sup :" +str(sup)
        i += 1
    return TOctave,dbPerTOctave

# return TOctave,dbPerTOctave
def from_fft_to_mean_fft(xf,yf,quarterLength,initOctaveSize):
    dbPerTOctave = []
    TOctave = []
    N = quarterLength
    yf_calc = 2.0 / N * np.abs(yf[:N // 2])
    octaveWidth = initOctaveSize

    sup = octaveWidth;

    currentSum = 0
    entriesUsed = 0
    infxval = xf[1]
    i = 1
    while i < len(xf):
        if xf[i] < sup:
            currentSum += yf_calc[i]
            entriesUsed += 1
        else:
            dbPerTOctave.append(currentSum / entriesUsed)              # calcul et enregistrement de l'amplitude moyenne
            TOctave.append((infxval + xf[i]) / 2)                     # calcul et enregistrement de la frequence moyenne

            currentSum = yf_calc[i]
            entriesUsed = 1

            infxval = xf[i]
            sup = infxval + octaveWidth
        i += 1
    return TOctave,dbPerTOctave

def third_octave(centre):
    res = []
    listcentre = []
    moy = 0
    fl = 0
    fu = 0
    
    while(centre >  15):
        fd = 2**(1/6)
        fu = centre * fd
        fl = centre / fd
        moy = 0
        cpt = 1
        i = 0
        print()
        print("centre : " + str(centre))
        print("fu : " + str(fu))
        print("fl : " + str(fl))
        while((i < len(yf_calc)- 2) and xf[i] < fl ):
            i = i + 1
    #    print(i)
        while((i < len(yf_calc)- 2) and (xf[i] < fu) ):
           moy = moy + yf_calc[i]
           i = i +1
           cpt = cpt + 1
        print(moy)
        moy = moy / cpt
        res.append(moy)
        listcentre.append(centre)
        centre = fl / fd
        
    centre = 1000 * fd ** 2
    listcentre.reverse()
    while(centre < 20000 ):
        fd = 2**(1/6)
        fu = centre * fd
        fl = centre / fd
        moy = 0
        cpt = 1
        i = 0
        print()
        print("centre : " + str(centre))
        print("fu : " + str(fu))
        print("fl : " + str(fl))
        while((i < len(yf_calc)- 2) and xf[i] < fl ):
            i = i + 1
    #    print(i)
        while((i < len(yf_calc)- 2) and (xf[i] < fu) ):
           moy = moy + yf_calc[i]
           i = i +1
           cpt = cpt + 1
        print(moy)
        moy = moy / cpt
        res.append(moy)
        listcentre.append(centre)
        centre = fu * fd
    return res,listcentre
        



def main(filename):


    df = from_xlsx_to_df(filename)
    ## lecture par colonne
    # colonne1 = sh.col_values(0)
    # print colonne1
    # [u'id', 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]


    # aaa=np.fft.fft(df.ACC1,quarterLength)
    # freq = np.fft.fftfreq(df.time[:quarterLength].shape[-1])
    # plt.plot(freq, aaa.real, freq)#,  df.time[:quarterLength], aaa.imag)
    # plt.show()
    # print "fft calculated"

    # print df.time
    quarterLength = len(df.time)//4
    numeroQuart = 1
    xf,yf = from_df_to_fft(df,quarterLength,numeroQuart,"ACC1")
    # Number of samplepoints
    N = quarterLength
    # sample spacing
    T = df.time[1]

    # x = np.linspace(0.0, N*T, N)
    # x = df.time[:quarterLength]
    print("test")
                                                # abscisse des frequences

#    plt.figure()                                                                    # Creation de la figure pour plotter

    # --------------ACC1 : Transformee de Fourier------------------------

                                                      # fft

    ax = plt.subplot(211)                                                                      # creation region de plot
    ax.set_title('ACC1 : Transformee de Fourier ')
#    xf = np.linspace(0.0, 1.0 / (2.0 * T), N // 2)  
    plt.plot(xf, 2.0 / N * np.abs(yf[:N // 2]))                                     # affichage amplitude en fct de freq

    # --------------ACC2 : Transformee de Fourier------------------------

    xf,yf = from_df_to_fft(df,quarterLength,numeroQuart,"ACC2")

    ax = plt.subplot(212)
    ax.set_title('ACC2 : Transformee de Fourier ')
    yf_calc = 2.0 / N * np.abs(yf[:N // 2])
    plt.plot(xf, yf_calc)
    plt.show()
    
    



    # --------------ACC2 : Tiers d'octave && ACC2 : Transformee de Fourier moyenne (pas=5)------------------------

    #TODO Tiers d'octave à reimplementer en utilisant les formules ..*2^1/6 et ../2^1/6 pour un tiers d'octave plus
    #TODO precisement centre sur 1000 Hertz

    # .............ACC2 : Tiers d'octave

    initOctaveSize = 11.048543456;  # taille du premier octave
    TOctave,dbPerTOctave = from_fft_to_third_octave(xf,yf,quarterLength,initOctaveSize)

#    k = 1
#    numOctaves = 0
#
#    maxfreq = xf[len(xf) - 1]
#    tempMaxFreq = 0
#    while (tempMaxFreq < maxfreq):
#        numOctaves += 1
#        tempMaxFreq += initOctaveSize * (2 ** numOctaves)
#
#    numTiersOctave = numOctaves * 3
#
#    dbPerTOctave = []
#    TOctave = []
#
#    inf = 0
#    sup = 11.048543456+initOctaveSize / 3;
#
#    octaveWidth = initOctaveSize
#
#    currentSum = 0
#    entriesUsed = 0
#    infxval = xf[1]
#    i = 1
#
#    i = 1
#    while xf[i] < 11.048543456: #on commence le premier octave à 11 Hertz pour être centré sur 15.625 Hertz
#        i+=1
#
#    while i < len(xf):
#        if xf[i] < sup:
#            currentSum += yf_calc[i]
#            entriesUsed += 1
#        else:
#            dbPerTOctave.append(currentSum / entriesUsed)
#            TOctave.append((infxval + xf[i]) / 2)
#            currentSum = yf_calc[i]
#            entriesUsed = 1
#
#            if (len(dbPerTOctave) % 3) == 0:  # si on passe à l'octave suivant, on double la largeur d'un octave
#                octaveWidth = 2 * octaveWidth
#                # print "octave width: "+str(octaveWidth) + "\n"
#                # print "tiers d'octave number: "+str(len(dbPerTOctave)) + "\n"
#
#            infxval = xf[i]
#            # print "sup :" +str(sup)
#            # print "1/3 * octaveWidth :" +str(octaveWidth/3)
#            sup = infxval + octaveWidth / 3
#            # print "sup :" +str(sup)
#        i += 1


    plt.figure(2)
    ax = plt.subplot(211)
    ax.set_title("ACC2 : Tiers d'octave")
    # ax.title.set_position(('outward', 40))
    ax.plot(np.linspace(0.0, len(dbPerTOctave)/3,len(dbPerTOctave)), dbPerTOctave)

    newax = ax.figure.add_axes(ax.get_position())
    newax.patch.set_visible(False)
    newax.yaxis.set_visible(False)
    newax.xaxis.set_ticks_position('top')
    newax.xaxis.set_label_position('top')

    newax.set_xticks(np.linspace(0.0, len(dbPerTOctave)/3,len(dbPerTOctave)))
    newax.set_xticklabels(np.asarray(TOctave, dtype=int))


    ax.set_xlabel('indice de tiers doctaves')
    newax.set_xlabel('frequence de centre de tiers doctave')
    # plt.show()

    # .............ACC2 : Transformee de Fourier moyenne (pas=5)

    initOctaveSize = 20;  # taille du premier octave
    TOctave,dbPerTOctave = from_fft_to_mean_fft(xf,yf,quarterLength,initOctaveSize)
    

#    dbPerTOctave = []
#    TOctave = []
#
#    octaveWidth = initOctaveSize
#
#    sup = octaveWidth;
#
#    currentSum = 0
#    entriesUsed = 0
#    infxval = xf[1]
#    i = 1
#    while i < len(xf):
#        if xf[i] < sup:
#            currentSum += yf_calc[i]
#            entriesUsed += 1
#        else:
#            dbPerTOctave.append(currentSum / entriesUsed)              # calcul et enregistrement de l'amplitude moyenne
#            TOctave.append((infxval + xf[i]) / 2)                     # calcul et enregistrement de la frequence moyenne
#
#            currentSum = yf_calc[i]
#            entriesUsed = 1
#
#            infxval = xf[i]
#            sup = infxval + octaveWidth
#        i += 1

    ax = plt.subplot(212)
    ax.set_title('ACC2 : Transformee de Fourier moyenne (pas='+str(initOctaveSize)+') ')

    ax.plot(TOctave, dbPerTOctave)

    # --------------affichage de toutes les courbes plottees------------------------

    plt.show()


if __name__ == "__main__":
    if len(sys.argv) == 2:
        main(sys.argv[1])
    elif len(sys.argv) == 1:
            print('Parsing.py : argument missing, switching to graphical file selection.')
            main(tkFileDialog.askopenfilename())
    else:
        print('Parsing.py : too many arguments, ignoring all but the first.')
        main(sys.argv[1])
