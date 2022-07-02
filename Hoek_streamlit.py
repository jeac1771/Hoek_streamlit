# -*- coding: utf-8 -*-
"""
Created on Sun Sep 29 13:07:05 2019

@author: Jorge Alvarez Correa
"""

## environment creado: conda create -n str_plots python=3.9
## pip install spyder
## pip install streamlit
## pip install seaborn
## pip install matplotlib
## pip install streamlit-aggrid

## para ejecutar:
## streamlit run c:\users\jalvarez\desktop\AppMLJEACv4.py [ARGUMENTS]

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
#from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from scipy.optimize import curve_fit
import scipy as sy
from numpy import sqrt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import (MultipleLocator)
import streamlit as st
from st_aggrid import GridOptionsBuilder, AgGrid, GridUpdateMode, DataReturnMode



st.title("Visualización de UG")

uploaded_file = st.file_uploader("Choose a file")
if uploaded_file is not None:
    clasificar_j = pd.read_csv(uploaded_file, sep=';')
    #st.write(clasificar_j)
    
    st.markdown('A continuación escoger si realizar la visualización utilizando intervalos de confianza o no:')
    agree = st.checkbox("Confidence Intervals")

    st.markdown('Escoger ensayos a graficar (mayor o igual a 2 ensayos)')    
    clasificar_j['Filtro'] = True

    listaa =[]*len(clasificar_j['SigmaS1'])
    listab =[]*len(clasificar_j['SigmaS3'])
    esfuerzos = []*len(clasificar_j['SigmaS1'])
    
    gb = GridOptionsBuilder.from_dataframe(clasificar_j)
    gb.configure_pagination(paginationAutoPageSize=True) #Add pagination
    gb.configure_side_bar() #Add a sidebar
    gb.configure_selection('multiple', use_checkbox=True, groupSelectsChildren="Group checkbox select children") #Enable multi-row selection
    gridOptions = gb.build()

    grid_response = AgGrid(
    clasificar_j,
    gridOptions=gridOptions,
    data_return_mode='AS_INPUT', 
    update_mode='MODEL_CHANGED', 
    fit_columns_on_grid_load=False,
    theme='blue', #Add theme color to the table
    enable_enterprise_modules=True,
    height=350, 
    width='100%',
    reload_data=True
    )

    data = grid_response['data']
    selected = grid_response['selected_rows'] 
    df_v2 = pd.DataFrame(selected) #Pass the selected rows to a new dataframe df
  
    print(len(df_v2))

##################################################################################################################
##################################################################################################################
############################################### DEFINIR ESFUERZOS ################################################

    #listaa.clear()
    #listab.clear()
    #esfuerzos.clear()
    
    if len(df_v2) > 1 :
    
        def patatab (df):
            for row in df['SigmaS1']:
                listaa.append(row)
            for row in df['SigmaS3']:
                listab.append(row)
            for i in range(len(listaa)):
                esfuerzos.append([listab[i],listaa[i]])
                
        patatab(df_v2)
        zip(*esfuerzos)
        
    
    ############## Algoritmo Mohr-Coloumb ###################
    
    
    # definir largo de listas a utilizar
    
        S1negativo =[]
        S1postivo =[]
        S3negativo =[]
        S3postivo =[]
        comboS3 =[]
        comboS1 =[]
    
        S1 = np.asarray(df_v2['SigmaS1'])
        S3 = np.asarray(df_v2['SigmaS3'])
    
    
    # Filtra la lista en valores de S3 negativos y positivos
    # a es S3 y b es S1   
    
        def dividirlista (a,b):
                
            S1negativo.clear()
            S1postivo.clear()
            S3negativo.clear()
            S3postivo.clear()
            comboS3.clear()
            comboS1.clear()
            
            global FlatComboS3, FlatComboS1
            
            for row in range(len(b)):
                if a[row] < 0:
                    S1negativo.append(b[row])
                else:
                    S1postivo.append(b[row])
            for row in range(len(a)):
                if a[row] < 0:
                    S3negativo.append(a[row])
                else:
                    S3postivo.append(a[row])
            
            comboS3.append([S1negativo,S3postivo])
            comboS1.append([S3negativo,S1postivo])
        
        # Convierte las lista de listas de listas (3d List) en 1d array.
        
            flat_list2DS3 = [item for sublist in comboS3 for item in sublist]
            flat_list1DS3 = [item for sublist in flat_list2DS3 for item in sublist]
            ComboS3array=np.array([flat_list1DS3])
            FlatComboS3 = ComboS3array.flatten()
        
            flat_list2DS1 = [item for sublist in comboS1 for item in sublist]
            flat_list1DS1 = [item for sublist in flat_list2DS1 for item in sublist]
            ComboS1array=np.array([flat_list1DS1])
            FlatComboS1 = ComboS1array.flatten()
                
                
        dividirlista(S3, S1)        
        
    # Definimos la función de Hoek combinada para S3 positivos y S3 negativos, para el caso S3 negativo se asume que S1 siempre es 0.
        
        def hoekgraph (x,a,b):
            return (x + a * (sqrt(b * (x / a) + 1)))
        
        def hoekpositivo (x,a,b):
            if a > 1 and b > 1:
                return (x + a * (sqrt(b * (x / a) + 1)))
            else:
                return (10000000000000 + x)
            
        def hoeknegativo (x,a,b):
            return (0.5 * (b*a - sqrt(pow(b,2) + 4) * a)) + x
        
        def combinedHoek(comboData,a,b):
            extract1 = comboData[:len(S3negativo)]
            extract2 = comboData[len(S3negativo):]
            
            result1 = hoeknegativo(extract1,a,b)
            result2 = hoekpositivo(extract2,a,b)
        
            return np.append(result1, result2)
        
        p1 = sy.array([5, 15])
    
    
        popt2, pcov2 = curve_fit(combinedHoek, FlatComboS3, FlatComboS1, p1, method='lm', maxfev=1000)
       
        
        s3ficticio = [c*0.1 - 10 for c in range(400)]
        s1mogi = [a * 3.4 for a in s3ficticio]
        
        ############################  Construcción grafico complejo ##############################
        
        filtrohist = []
        
        def filtrocero(): #filtro para histograma con S3 = 0
            for i in range(len(esfuerzos)):
                if (listab[i] == 0):
                    filtrohist.append([listaa[i]])
        filtrocero()
        
        
        
        def graf_porcategoria(a):
        
            sns.lmplot(x = 'SigmaS3', y = 'SigmaS1', 
                       data = df_v2, 
                       hue = a, 
                       fit_reg = False)
            plt.xlabel('SigmaS3')
            plt.ylabel('SigmaS1')
            plt.title('Relación Esfuerzos Categoría')
            plt.show(block=False) # con este comando al compilar, los graficos funcionan correctamente.
            
        
        def filtermogi(a):
        
            for i in range(len(a['SigmaS3'])):
                if a['SigmaS3'][i] * 3.4 > a['SigmaS1'][i]:
                    a['Filtro'][i] = False
            
        filtermogi(df_v2) 
        
        s3boxplot = [c*5  for c in range(9)]
        boxlist1=[]
        
        def filter_hoekplotall_label(a, b, c, d, e, j, h, l, m, z): ### PENDIENTE: INCLUIR PLOT LABEL CON ALLRESULTS
        
            # First, create the figure
            fig = plt.figure(1, figsize=(15,8))
    
            # Now, create the gridspec structure, as required
            gs = gridspec.GridSpec(1,5, height_ratios=[1], width_ratios=[0.2,0.2,1,0.02,0.58])
    
            # 3 rows, 4 columns, each with the required size ratios. 
            # Also make sure the margins and spacing are apropriate
    
            gs.update(left=0.05, right=0.95, bottom=0.08, top=0.93, wspace=0.02, hspace=0.03)
    
            # First, the scatter plot
            # Use the gridspec magic to place it
            # --------------------------------------------------------
            ax1 = plt.subplot(gs[0,2]) # place it where it should be.
            # --------------------------------------------------------
            
            # The plot itself
            ax1.scatter(a, b,color='darkorange',alpha=0.8,label='Esfuerzos (Mpa) - Filter', s = 70)
            ax1.scatter(c, d,color='blue',alpha=0.7,label='Esfuerzos (Mpa)', s = 70)
            ax1.plot(s3ficticio,hoekgraph(s3ficticio, *e),'r-',label='Fit: $\sigma_{ci}$=%.0f, $m_i$=%.0f' % tuple(e))
            ax1.plot(s3ficticio, s1mogi ,'--', color= "gray", label='Línea de Mogi')
                    
            ax1.legend(loc='best')
            
            #if button1 == 'Yes':
            #    for k in range(len(alldf)):
            #        if 250 > alldf['SigmaS1'][k]:
            #            ax1.annotate(alldf[z][k], xy=(alldf['SigmaS3'][k], alldf['SigmaS1'][k]),
            #                         xytext=(10,-5), textcoords='offset points',
            #                         family='sans-serif', fontsize=10, color='darkslategrey')
    
            # Define the limits, labels, ticks as required
            ax1.grid(True)
            ax1.set_xlim([-10,30])
            ax1.set_ylim([0,250])
            #ax1.set_xlabel(r' ') # Force this empty !
            ax1.set_xticks(np.linspace(-10,30,5)) # Force this to what I want - for consistency with histogram below !
            ax1.set_yticks(np.linspace(0,250,6)) # Force this to what I want - for consistency with histogram below !
            ax1.xaxis.set_minor_locator(MultipleLocator(1))
            ax1.yaxis.set_minor_locator(MultipleLocator(10))
            ax1.xaxis.grid(True, which='minor')
            ax1.yaxis.grid(True, which='minor')
            ax1.xaxis.grid(color='black', which='major')
            ax1.yaxis.grid(color='black', which='major')
            #ax1.set_xticklabels([]) # Force this empty !
            ax1.set_ylabel(r'Esfuerzo Principal Mayor, S1 (Mpa)')
            ax1.set_xlabel(r'Esfuerzo Principal Menor, S3 (Mpa)')
    
            # And now the histogram
            # Use the gridspec magic to place it
            # --------------------------------------------------------
            ax1v = plt.subplot(gs[0,0])
            # --------------------------------------------------------
    
            # Plot the data
            binwidth = 5
            xymax = max(np.max(np.abs(S3)), np.max(np.abs(S1)))
            lim = (int(xymax/binwidth) + 1)*binwidth
            bins = np.arange(-lim, lim + binwidth, binwidth)
            if len(j) > 0:
                ax1v.hist(*zip(*j), bins=bins, orientation='horizontal',color='blue',alpha=1, edgecolor='white')
            ax1v.invert_xaxis()
            ax1v.spines['left'].set_position(('axes', 0))
            ax1v.set_frame_on(True)
            ax1v.patch.set_visible(True)
            
            # Define the limits, labels, ticks as required
            ax1v.set_yticks(np.linspace(0,250,6)) # Ensures we have the same ticks as the scatter plot !
            #ax1v.set_xticklabels([])
            ax1v.set_yticklabels([])
            ax1v.set_ylim([0,250])
            ax1v.set_xticks([0, 2, 4, 6, 8, 10, 12])
            ax1v.grid(True)
            ax1v.set_xlabel(r'N° Ensayos UCS')
    
            # And now the text box
            # Use the gridspec magic to place it
            # --------------------------------------------------------
            ax2t = plt.subplot(gs[0,4])
            # --------------------------------------------------------
    
            # print textstr
            if len(j) > 0:
                textstr = 'Resultados:\n\nAjuste H-B\n$\sigma_{ci}$=%.0f, $m_i$=%.0f\n\nEstadística UCS (Mpa)'% tuple(e) + '\nMedia=' + str(int(l)) + ' Desv='+ str(int(m))
            else:
                textstr = 'Resultados:\n\nAjuste H-B\n$\sigma_{ci}$=%.0f, $m_i$=%.0f\n\nEstadística UCS (Mpa)'% tuple(e) + '\nMedia= - Desv= -'
            props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
            ax2t.text(0, 0.5, textstr, fontsize=14,bbox=props)
            ax2t.set_xticklabels([])
            ax2t.set_yticklabels([])
            ax2t.set_xlabel(r' ')
            ax2t.set_ylabel(r' ')
            ax2t.grid(False)
            ax2t.set_frame_on(False)
            ax2t.patch.set_visible(False)
            ax2t.tick_params(which='both', left=False, bottom=False, top=False, labelbottom=False)
            
            st.pyplot(fig)
            
        #numero de muestras
        size=300
        
        # Initializar replicas: bs_slope_reps, bs_intercept_reps
        y_sm = np.empty(size)
        pcovall = np.empty(size)
        list_ysm = []
        list_ysm.clear()
        
        alldf = df_v2.copy()
        filtermogi(alldf)
        
        x=np.asarray(alldf.loc[(alldf['Filtro'] == True),'SigmaS3'])
        y=np.asarray(alldf.loc[(alldf['Filtro'] == True),'SigmaS1'])
        #Indices
        inds=np.arange(len(x))
        
        # Generar replicates
        for i in range(size):
        
            bs_inds = np.random.choice(inds, size=len(inds))
            bs_x, bs_y = x[bs_inds], y[bs_inds]
            ## falta el filtro de mogi
            
            dividirlista(bs_x, bs_y)
            combinedHoek(FlatComboS3,5,15)
            y_sm, pcovall = curve_fit(combinedHoek, FlatComboS3, FlatComboS1, p1, method='lm', maxfev=1000)
            list_ysm.append([y_sm[0], y_sm[1]])
            
        np_list = np.array(list_ysm)
        
        print('a: IC (95%):',np.percentile(np_list[:,0], [2.5, 97.5]))     
        print('b: IC (95%):',np.percentile(np_list[:,1], [2.5, 97.5]))
            
        def filter_hoekplotall_label_ci(a, b, c, d, e, j, h, l, m, z): ### PENDIENTE: INCLUIR PLOT LABEL CON ALLRESULTS
        
            # First, create the figure
            fig = plt.figure(1, figsize=(15,8))
    
            # Now, create the gridspec structure, as required
            gs = gridspec.GridSpec(1,5, height_ratios=[1], width_ratios=[0.2,0.2,1,0.02,0.58])
    
            # 3 rows, 4 columns, each with the required size ratios. 
            # Also make sure the margins and spacing are apropriate
    
            gs.update(left=0.05, right=0.95, bottom=0.08, top=0.93, wspace=0.02, hspace=0.03)
    
            # First, the scatter plot
            # Use the gridspec magic to place it
            # --------------------------------------------------------
            ax1 = plt.subplot(gs[0,2]) # place it where it should be.
            # --------------------------------------------------------
            
            # The plot itself
            ax1.scatter(a, b,color='darkorange',alpha=0.8,label='Esfuerzos (Mpa) - Filter', s = 70)
            ax1.scatter(c, d,color='blue',alpha=0.7,label='Esfuerzos (Mpa)', s = 70)
            ax1.plot(s3ficticio,hoekgraph(s3ficticio, *e),'r-',label='Fit: $\sigma_{ci}$=%.0f, $m_i$=%.0f' % tuple(e))
            for i in range(len(np_list)):
                new_s3 = []
                for idx in s3ficticio:
                    if (np_list[i,1] * (idx / np_list[i,0]) + 1) >= 0:
                        new_s3.append(idx)
                ax1.plot(new_s3,hoekgraph(new_s3, np_list[i,0], np_list[i,1]),'r-', color = 'aquamarine', alpha = 0.1, zorder=1)
                new_s3.clear()
            ax1.plot(s3ficticio, s1mogi ,'--', color= "gray", label='Línea de Mogi')
                    
            ax1.legend(loc='best')
            
            #if button1 == 'Yes':
            #    for k in range(len(alldf)):
            #        if 250 > alldf['SigmaS1'][k]:
            #            ax1.annotate(alldf[z][k], xy=(alldf['SigmaS3'][k], alldf['SigmaS1'][k]),
            #                         xytext=(10,-5), textcoords='offset points',
            #                         family='sans-serif', fontsize=10, color='darkslategrey')
    
            # Define the limits, labels, ticks as required
            ax1.grid(True)
            ax1.set_xlim([-10,30])
            ax1.set_ylim([0,250])
            #ax1.set_xlabel(r' ') # Force this empty !
            ax1.set_xticks(np.linspace(-10,30,5)) # Force this to what I want - for consistency with histogram below !
            ax1.set_yticks(np.linspace(0,250,6)) # Force this to what I want - for consistency with histogram below !
            ax1.xaxis.set_minor_locator(MultipleLocator(1))
            ax1.yaxis.set_minor_locator(MultipleLocator(10))
            ax1.xaxis.grid(True, which='minor')
            ax1.yaxis.grid(True, which='minor')
            ax1.xaxis.grid(color='black', which='major')
            ax1.yaxis.grid(color='black', which='major')
            #ax1.set_xticklabels([]) # Force this empty !
            ax1.set_ylabel(r'Esfuerzo Principal Mayor, S1 (Mpa)')
            ax1.set_xlabel(r'Esfuerzo Principal Menor, S3 (Mpa)')
    
            # And now the histogram
            # Use the gridspec magic to place it
            # --------------------------------------------------------
            ax1v = plt.subplot(gs[0,0])
            # --------------------------------------------------------
    
            # Plot the data
            binwidth = 5
            xymax = max(np.max(np.abs(S3)), np.max(np.abs(S1)))
            lim = (int(xymax/binwidth) + 1)*binwidth
            bins = np.arange(-lim, lim + binwidth, binwidth)
            if len(j) > 0:
                ax1v.hist(*zip(*j), bins=bins, orientation='horizontal',color='blue',alpha=1, edgecolor='white')
            ax1v.invert_xaxis()
            ax1v.spines['left'].set_position(('axes', 0))
            ax1v.set_frame_on(True)
            ax1v.patch.set_visible(True)
            
            # Define the limits, labels, ticks as required
            ax1v.set_yticks(np.linspace(0,250,6)) # Ensures we have the same ticks as the scatter plot !
            #ax1v.set_xticklabels([])
            ax1v.set_yticklabels([])
            ax1v.set_ylim([0,250])
            ax1v.set_xticks([0, 2, 4, 6, 8, 10, 12])
            ax1v.grid(True)
            ax1v.set_xlabel(r'N° Ensayos UCS')
    
            # And now the text box
            # Use the gridspec magic to place it
            # --------------------------------------------------------
            ax2t = plt.subplot(gs[0,4])
            # --------------------------------------------------------
    
            # print textstr
            if len(j) > 0:
                textstr = 'Resultados:\n\nAjuste H-B\n$\sigma_{ci}$=%.0f, $m_i$=%.0f\n\nEstadística UCS (Mpa)'% tuple(e) + '\nMedia=' + str(int(l)) + ' Desv='+ str(int(m))
            else:
                textstr = 'Resultados:\n\nAjuste H-B\n$\sigma_{ci}$=%.0f, $m_i$=%.0f\n\nEstadística UCS (Mpa)'% tuple(e) + '\nMedia= - Desv= -'
            props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
            ax2t.text(0, 0.5, textstr, fontsize=14,bbox=props)
            ax2t.set_xticklabels([])
            ax2t.set_yticklabels([])
            ax2t.set_xlabel(r' ')
            ax2t.set_ylabel(r' ')
            ax2t.grid(False)
            ax2t.set_frame_on(False)
            ax2t.patch.set_visible(False)
            ax2t.tick_params(which='both', left=False, bottom=False, top=False, labelbottom=False)
            
            st.pyplot(fig)
            
        allhoek = []
        strhoek = []
        listaball = []
        listaaall = []
        filtrohistall = []
        cell_text = []
        mystress = [str(list(df_v2))]
        litohoek = [] # append LITO
        althoek = [] # append ALT
        minhoek = [] # append MIN
        lendata = [] # append Cantidad de datos usados para estimación de sci y mi
        cell_texta = []
        listaballa = []
        listaaalla = []
        filtrohistalla = []
        cell_textb = []
        listaballb = []
        listaaallb = []
        filtrohistallb = []
            
        def selectfilterplot(a, b, c):
    
             global superduper1, superduper2, superduper3, alldf, indexdataf
             allhoek.clear()
             strhoek.clear()
             cell_text.clear()
             listaball.clear()
             listaaall.clear()
             filtrohistall.clear()
    
             alldf = pd.DataFrame(data=df_v2.loc[(df_v2[a] == b),df_v2.columns.str.contains('|'.join(mystress))])
             alldfmirror = pd.DataFrame(data=df_v2.loc[(df_v2[a] == b),df_v2.columns.str.contains('|'.join(mystress))])
             alldf = alldf.reset_index(drop=True)
             indexdataf = pd.DataFrame(data=alldfmirror.index.values,columns = ['index'])
             #alldf['Filtro'] = True #nuevo 15-12-2019
            
             if len(alldf) > 1: #PARA QUE REALICE LOS AJUSTES (MÁS DE 2 DATOS)
                 
                 filtermogi(alldf)
                 S1all = np.asarray(alldf.loc[(alldf['Filtro'] == True),'SigmaS1'])
                 S3all = np.asarray(alldf.loc[(alldf['Filtro'] == True),'SigmaS3'])
                 S1f = np.asarray(alldf.loc[(alldf['Filtro'] == False),'SigmaS1'])
                 S3f = np.asarray(alldf.loc[(alldf['Filtro'] == False),'SigmaS3'])
                 
                 for row in alldf.loc[(alldf['Filtro'] == True),'SigmaS3']:
                     listaball.append(row)
                 for row in alldf.loc[(alldf['Filtro'] == True),'SigmaS1']:
                     listaaall.append(row)
                 for t in range(len(listaball)):
                     if (listaball[t] == 0):
                         filtrohistall.append([listaaall[t]])
                 if len(filtrohistall) > 0:
                         meanhist=np.mean(filtrohistall)
                         stdhist=np.std(filtrohistall)
                 else:
                     meanhist=0
                     stdhist=0
                     
                 if len(S1all) > 1: #PARA CUANDO TIENE MAS DE 2 VALORES, PERO ESTOS SON FILTRADOS..
                         
                     dividirlista (S3all,S1all)
                     combinedHoek(FlatComboS3,5,15)
                     poptall, pcovall = curve_fit(combinedHoek, FlatComboS3, FlatComboS1, p1, method='lm', maxfev=1000)
                     if poptall[1] > 50 and len(filtrohistall) > 0:
                         poptall[1] = 50
                         poptall[0] = meanhist
                     if poptall[1] > 50 and len(filtrohistall) == 0:
                         poptall[0] = np.mean(S1all)
                         poptall[1] = 5
                     allhoek.append(np.round(poptall, 2))
                     
                     #residuals = S1all - hoekgraph(S3all, *poptall)
                     #ss_res = np.sum(residuals**2)
                     
    
                     filter_hoekplotall_label(S3f, S1f, S3all, S1all, poptall, filtrohistall, b, meanhist, stdhist, 0)
    
    
             else:
                 poptall=0
                 
        def selectfilterplot_ci(a, b, c):
    
            global superduper1, superduper2, superduper3, alldf, indexdataf
            allhoek.clear()
            strhoek.clear()
            cell_text.clear()
            listaball.clear()
            listaaall.clear()
            filtrohistall.clear()
            
            alldf = pd.DataFrame(data=df_v2.loc[(df_v2[a] == b),df_v2.columns.str.contains('|'.join(mystress))])
            alldfmirror = pd.DataFrame(data=df_v2.loc[(df_v2[a] == b),df_v2.columns.str.contains('|'.join(mystress))])
            alldf = alldf.reset_index(drop=True)
            indexdataf = pd.DataFrame(data=alldfmirror.index.values,columns = ['index'])

                    
            if len(alldf) > 1: #PARA QUE REALICE LOS AJUSTES (MÁS DE 2 DATOS)
                         
                filtermogi(alldf)
                S1all = np.asarray(alldf.loc[(alldf['Filtro'] == True),'SigmaS1'])
                S3all = np.asarray(alldf.loc[(alldf['Filtro'] == True),'SigmaS3'])
                S1f = np.asarray(alldf.loc[(alldf['Filtro'] == False),'SigmaS1'])
                S3f = np.asarray(alldf.loc[(alldf['Filtro'] == False),'SigmaS3'])
                         
                for row in alldf.loc[(alldf['Filtro'] == True),'SigmaS3']:
                    listaball.append(row)
                for row in alldf.loc[(alldf['Filtro'] == True),'SigmaS1']:
                    listaaall.append(row)
                for t in range(len(listaball)):
                    if (listaball[t] == 0):
                        filtrohistall.append([listaaall[t]])
                if len(filtrohistall) > 0:
                    meanhist=np.mean(filtrohistall)
                    stdhist=np.std(filtrohistall)
                else:
                    meanhist=0
                    stdhist=0
                             
                if len(S1all) > 1: #PARA CUANDO TIENE MAS DE 2 VALORES, PERO ESTOS SON FILTRADOS..
                                 
                    dividirlista (S3all,S1all)
                    combinedHoek(FlatComboS3,5,15)
                    poptall, pcovall = curve_fit(combinedHoek, FlatComboS3, FlatComboS1, p1, method='lm', maxfev=1000)
                    print(poptall)
                    if poptall[1] > 50 and len(filtrohistall) > 0:
                        poptall[1] = 50
                        poptall[0] = meanhist
                    if poptall[1] > 50 and len(filtrohistall) == 0:
                        poptall[0] = np.mean(S1all)
                        poptall[1] = 5
                    allhoek.append(np.round(poptall, 2))
                             
            
                    filter_hoekplotall_label_ci(S3f, S1f, S3all, S1all, poptall, filtrohistall, b, meanhist, stdhist, 0)
            
            
            else:
                poptall=0
                 
        if agree:
            
            plot_function = selectfilterplot_ci
            
        else:
            
            plot_function = selectfilterplot
                 
        plot_function('LIT_UG', 'AND', 'SigmaS1')





