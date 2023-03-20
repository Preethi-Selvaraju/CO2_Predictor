# -*- coding: utf-8 -*-
"""
Created on Thu Mar 16 17:23:20 2023

@author: HP
"""

import requests, time, random, datetime,ftplib
from datetime import date
import base64
import os, re, matplotlib
from reportlab.pdfgen import canvas
from reportlab.lib import utils
from reportlab.lib.pagesizes import letter
from reportlab.lib.pagesizes import landscape, portrait
from reportlab.platypus import Image
import matplotlib.pyplot as plt
from matplotlib.patches import Shadow
from PyPDF2 import PdfMerger, PdfReader
import numpy as np 
import pandas as pd
from PIL import Image
import streamlit as st
import netCDF4 as nc
import math
from streamlit_folium import folium_static
from streamlit_folium import st_folium
import folium
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt 
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from tensorflow.keras.models import Sequential
from keras.layers import Dense,LSTM,Activation,Bidirectional, Flatten, Convolution1D, Dropout,MaxPooling1D
from keras.optimizers import SGD

st.set_page_config(
page_title="Carbon dioxide Predictor",
page_icon="🌎"
)


def add_bg_from_local(image_file):
    with open(image_file, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
    st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url(data:image/{"jpg"};base64,{encoded_string.decode()});
        background-size: cover
    }}
    </style>
    """,
    unsafe_allow_html=True
    )


try:
    st.sidebar.markdown("<h1 style='text-align: center; color: black;'>🧭 Navigation Bar 🧭</h1>", unsafe_allow_html=True)
    nav = st.sidebar.radio("",["Home 🏡","Individual Emission 👨‍👩‍👧‍👦","Group Emission 🌐"])
    if nav == "Home 🏡":
      add_bg_from_local("background.jpg")
      
      st.markdown("<h1 style='color:black;text-align: center;font-family:times new roman;font-size:30pt;font-weight: bold;'>CARBON DIOXIDE PREDICTOR</h1>", unsafe_allow_html=True)
      st.markdown("<h1 style='color:green;text-align: center;font-family:times new roman;font-size:25pt;font-weight: bold;'>It's time to reduce Carbon!</h1>", unsafe_allow_html=True)        
   
    if nav == "Group Emission 🌐":

        st.markdown(f"""<h1 style='text-align: center; font-weight:bold;color:black;background-color:powderblue;font-size:20pt;'>Know the CO2 level at your area⚠️</h1>""",unsafe_allow_html=True)
        m = folium.Map(location=None, width='100%', height='100%', left='0%', top='0%', position='relative', tiles='OpenStreetMap', attr=None, min_zoom=0, max_zoom=18, zoom_start=10, min_lat=- 90, max_lat=90, min_lon=- 180, max_lon=180, max_bounds=True, crs='EPSG3857', control_scale=False, prefer_canvas=False, no_touch=False, disable_3d=False, png_enabled=False, zoom_control=True)
        m.add_child(folium.LatLngPopup())
        map = st_folium(m)
        try:

            user_lat=map['last_clicked']['lat']
            user_lon=map['last_clicked']['lng'] 

        except:
            st.warning("No location choosen")

        today = date.today()

        date1 = st.date_input('Date', value =  pd.to_datetime('2023-01-01'),min_value= pd.to_datetime('2023-01-01'),max_value= pd.to_datetime(today))
        
        user_lat=float(user_lat)
        user_lon=float(user_lon)
        
        predict_days=abs((datetime.strptime('2022-12-31',"%Y-%m-%d")-datetime.strptime(str(date1),"%Y-%m-%d")).days)

        if st.button("Predict"):

            df_all=pd.DataFrame(columns=['DATE','CO2'])
            i=0
            for root, dirs, files in os.walk("data"):
                for file in files:
                    if os.path.splitext(file)[1] == '.nc4':
                        filePath = os.path.join(root, file)
                    ds = nc.Dataset(filePath)

                    df=pd.DataFrame(columns=["Latitude","Longitude","xco2"])

                    df["Longitude"] = ds['longitude'][:]
                    df["Latitude"] = ds['latitude'][:]
                    df["xco2"]=ds['xco2'][:]

                    #Repalce inplace 
                    df.fillna(0,inplace=True)

                    df_first=df.loc[(df['Latitude'] >user_lat) &(df['Latitude'] < user_lat+20) & (df['Longitude']> user_lon)&(df['Longitude']< user_lon+20 ),'xco2']
                    res=df_first.mean()


                    df_all.loc[i,"DATE"] = file[15:17]+"/"+file[13:15]+"/20"+file[11:13]
                    df_all.loc[i,"CO2"] = res

                    i+=1
            df_all.fillna(df_all['CO2'].mean(),inplace=True)
            df_all.to_csv(r"days_combined.csv")
            st.write(df_all)

 ##############################################
 #            PREDICTION MODULE               #
 ##############################################
 ### Data Collection
            
            data_frame=pd.read_csv(r"days_combined.csv")
            df1=data_frame.reset_index()['CO2']

             ### LSTM are sensitive to the scale of the data. so we apply MinMax scaler 

            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            df1 =scaler.fit_transform(np.array(df1).reshape(-1,1)) 

             ##splitting dataset into train and test split
            training_size=int(len(df1)*0.70)
            test_size=len(df1)-training_size
            train_data,test_data=df1[0:training_size,:],df1[training_size:len(df1),:1]


             # convert an array of values into a dataset matrix
            def create_dataset(dataset, time_step=1):
                dataX, dataY = [], []
                for i in range(len(dataset)-time_step-1):         
                    a = dataset[i:(i+time_step), 0]   ###i=0, 0,1,2,3-----99   100 
                    dataX.append(a)
                    dataY.append(dataset[i + time_step, 0])
                return np.array(dataX), np.array(dataY)

             # reshape into X=t,t+1,t+2,t+3 and Y=t+4
            time_step = 2
            X_train, y_train = create_dataset(train_data, time_step)
            X_test, ytest = create_dataset(test_data, time_step)


            # reshape input to be [samples, time steps, features] which is required for LSTM
            X_train =X_train.reshape(X_train.shape[0],X_train.shape[1] , 1)
            X_test = X_test.reshape(X_test.shape[0],X_test.shape[1] , 1)

            
             ### Create the BI-LSTM model
            model = Sequential()
            model.add(Bidirectional(LSTM(100, input_shape=(time_step,1))))
            model.add(Dense(1,activation="sigmoid"))
            model.compile(loss='mean_squared_error', optimizer='adam',metrics=['accuracy'])
            model.fit(X_train,y_train,validation_data=(X_test,ytest),epochs=25,batch_size=2,verbose=1)


            ### Lets Do the prediction and check performance metrics
            train_predict=model.predict(X_train)
            test_predict=model.predict(X_test)

            ##Transformback to original form
            train_predict=scaler.inverse_transform(train_predict)
            test_predict=scaler.inverse_transform(test_predict)

            ### Calculate RMSE performance metrics
            math.sqrt(mean_squared_error(y_train,train_predict))

            ### Test Data RMSE
            math.sqrt(mean_squared_error(ytest,test_predict))

            ### Plotting 
            # shift train predictions for plotting
            look_back=2
            trainPredictPlot = np.empty_like(df1)
            trainPredictPlot[:, :] = np.nan
            trainPredictPlot[look_back:len(train_predict)+look_back, :] = train_predict
            # shift test predictions for plotting
            testPredictPlot = np.empty_like(df1)
            testPredictPlot[:, :] = np.nan
            testPredictPlot[len(train_predict)+(look_back*2)+1:len(df1)-1, :] = test_predict

            x_input=test_data[len(test_data)-time_step:].reshape(1,-1)

            temp_input=list(x_input)
            temp_input=temp_input[0].tolist()

            # demonstrate prediction for next days
            lst_output=[]
            n_steps=2
            i=0
            while(i<predict_days):

                if(len(temp_input)>n_steps):
                    x_input=np.array(temp_input[1:])
                    print("{} day input {}".format(i,x_input))
                    x_input=x_input.reshape(1,-1)
                    x_input = x_input.reshape((1, n_steps, 1))
                    yhat = model.predict(x_input, verbose=0)
                    print("{} day output {}".format(i,yhat))
                    temp_input.extend(yhat[0].tolist())
                    temp_input=temp_input[1:]
                    lst_output.extend(yhat.tolist())
                    i=i+1
                else:
                    x_input = x_input.reshape((1, n_steps,1))
                    yhat = model.predict(x_input, verbose=0)
                    print(yhat[0])
                    temp_input.extend(yhat[0].tolist())
                    print(len(temp_input))
                    lst_output.extend(yhat.tolist())
                    i=i+1

   
            co2_output=pd.DataFrame(scaler.inverse_transform(lst_output),columns=['CO2 Concentration 🏭'])
            output= (co2_output.at[predict_days-1,'CO2 Concentration 🏭'])


            st.info("BiLSTM model predicts:")
            if(math.isnan(output)):
                st.error("Unable to find co2 concentration at the specified location")
            else:
                st.write("""<style>[data-testid="stMetricDelta"] svg {display: none;}</style>""",unsafe_allow_html=True)   
                if(output<400):
                    st.metric(label="Amount of CO2 (in ppm)", value=round(output,3), delta="parts per millions")
                else:
                    st.metric(label="Amount of CO2 (in ppm)", value=round(output,3), delta="parts per millions",delta_color="inverse")


            ### Create the 1d cnn model
            nb_timesteps=X_train.shape[1]
            nb_features=X_train.shape[2]
            model = Sequential()
            model.add(Convolution1D(filters=64, kernel_size=1, input_shape=(nb_timesteps,nb_features)))
            model.add(Convolution1D(filters=32, kernel_size=1))
            model.add(MaxPooling1D(pool_size=2))
            model.add(Flatten())
            model.add(Dropout(0.2))
            model.add(Dense(100, activation='relu'))
            model.add(Dense(1,activation="sigmoid"))
            model.compile(loss='mean_squared_error',optimizer='adam',metrics=['accuracy'])
            model.fit(X_train, y_train, epochs=20, validation_data=(X_test, ytest),batch_size=64)
            ### Lets Do the prediction and check performance metrics
            train_predict=model.predict(X_train)
            test_predict=model.predict(X_test)

             ##Transformback to original form
            train_predict=scaler.inverse_transform(train_predict)
            test_predict=scaler.inverse_transform(test_predict)

            ### Calculate RMSE performance metrics
            math.sqrt(mean_squared_error(y_train,train_predict))

            ### Test Data RMSE
            math.sqrt(mean_squared_error(ytest,test_predict))

            ### Plotting 
            # shift train predictions for plotting
            look_back=2
            trainPredictPlot = np.empty_like(df1)
            trainPredictPlot[:, :] = np.nan
            trainPredictPlot[look_back:len(train_predict)+look_back, :] = train_predict
            # shift test predictions for plotting
            testPredictPlot = np.empty_like(df1)
            testPredictPlot[:, :] = np.nan
            testPredictPlot[len(train_predict)+(look_back*2)+1:len(df1)-1, :] = test_predict

            x_input=test_data[len(test_data)-time_step:].reshape(1,-1)

            temp_input=list(x_input)
            temp_input=temp_input[0].tolist()

            # demonstrate prediction for next days
            lst_output=[]
            n_steps=2
            i=0
            while(i<predict_days):

                if(len(temp_input)>n_steps):
                    x_input=np.array(temp_input[1:])
                    print("{} day input {}".format(i,x_input))
                    x_input=x_input.reshape(1,-1)
                    x_input = x_input.reshape((1, n_steps, 1))
                    yhat = model.predict(x_input, verbose=0)
                    print("{} day output {}".format(i,yhat))
                    temp_input.extend(yhat[0].tolist())
                    temp_input=temp_input[1:]
                    lst_output.extend(yhat.tolist())
                    i=i+1
                else:
                    x_input = x_input.reshape((1, n_steps,1))
                    yhat = model.predict(x_input, verbose=0)
                    print(yhat[0])
                    temp_input.extend(yhat[0].tolist())
                    print(len(temp_input))
                    lst_output.extend(yhat.tolist())
                    i=i+1

   
            co2_output=pd.DataFrame(scaler.inverse_transform(lst_output),columns=['CO2 Concentration 🏭'])
            output= (co2_output.at[predict_days-1,'CO2 Concentration 🏭'])

            st.info("1D CNN model predicts:")
            if(math.isnan(output)):
                st.error("Unable to find co2 concentration at the specified location")
            else:
                st.write("""<style>[data-testid="stMetricDelta"] svg {display: none;}</style>""",unsafe_allow_html=True)   
                if(output<400):
                    st.metric(label="Amount of CO2 (in ppm)", value=round(output,3), delta="parts per millions")
                else:
                    st.metric(label="Amount of CO2 (in ppm)", value=round(output,3), delta="parts per millions",delta_color="inverse")

    
    def clean_answer(answer):
        answer=answer.replace(' ','').replace('$','')
    
        if answer == 'y':
            answer='yes'
        elif answer == 'n':
            answer='no'
        # this is for intent querying 
        return answer
    
    


    
    def make_report(footprint, footprintdelta):
        # load the data here 
        g=open('report.txt').read()
    
        if footprintdelta < 0:
            footprintdelta = str(footprintdelta) + ' less than'
        else:
            footprintdelta = str(footprintdelta) + ' greater than'
    
        footprint=str(footprint) + ' tons of CO2/year'
    
        g=g.replace('[INSERT_FOOTPRINT_HERE]', footprint)
        g=g.replace('[INSERT_FOOTPRINTDELTA_HERE]', footprintdelta)
    
        return g
    
    # create a pdf 
    def calculate_footprint(answers):
        # create carbon calculator 
    
        # initialize answers 
        answer_1 = answers[0]
        answer_2 = answers[1]
        answer_3 = answers[2]
        answer_4 = answers[3]
        answer_5 = answers[4]
        answer_6 = answers[5]
        answer_7 = answers[6]
        answer_8 = answers[7]
        answer_9 = answers[8]
        answer_10 = answers[9]
        answer_11 = answers[10]
    
        # Electric bill = 7,252.76 kg CO2/year 
        # $0.1327/kwh/0.62 kg CO2/kwh = $0.214/kg CO2  - all we need to do is divide monthly bill by this.
        # electric bill = (electric bill / people in household) / ($0.214/kgCo2)     
        
        try:
            answer_1=answer_1.replace('$','')
            electric_=(int(answer_2)/int(answer_1))*12/0.214
        except:
            st.error('--> Error on electric CO2 calculation')
    
        # Flights = 602.448 kg CO2/year (if yes)
        # 286.88 kg CO2/flight 
        try:
            flight_= float(answer_3)*286.88 
        except:
            print('--> error on flight CO2 calculation')
            flight_=602.448
    
        # Transportation = 0.
        # 6,525.0 kg CO2/year (if drive only), 4,470.0 kg CO2/year (if mixed), 2,415.0 kg/year (if public)
        # 0.435 kg CO2/mile driving, 0.298 kg CO2/mile 50%/50% public transport and driving, and 0.161 kg CO2/mile (if public)
        # assume 220 working days/year (w/ vacation)
        try:
            transportation_=0
            if answer_4 == 'yes' and answer_6 == 'no':
                transportation_=float(answer_5)*1.61* 0.435*2*220
    
            elif answer_4 == 'yes' and answer_6 == 'yes':
                transportation_=float(answer_5)*1.61*0.298*2*220
    
            elif answer_4 == 'no' and answer_6 == 'yes':
                transportation_=float(answer_5)*1.61*0.161*2*220
    
            # Uber trips 
            # 45.27 kg CO2/year (average) 
            # 6 miles * 0.435 kg Co2/ mile = 2.61 kg CO2/trip 
            transportation_=transportation_+float(answer_8)*2.61*12
    
        except:
            st.error('--> error on transportation CO2 caclulation')
            transportation=4515.27
    
        # Vegetarian - assume footprint from food 
        try:
            if answer_9 == 'yes':
                food_=1542.21406
            # meat lover 
            elif answer_10 == 'yes':
                food_=2993.70964
            else:
                food_=2267.96185
        except:
            st.error('--> error on food CO2 calculation')
            food_=2267.96185
    
        # do you use amazon? --> retail, etc. 
        answer_11=answer_11.replace('$','').replace(' ','')
        retail_=0.1289*float(answer_11)
       
        footprint=electric_+flight_+transportation_+food_+retail_
        footprintbytype=[electric_, flight_, transportation_, food_, retail_]
    
        # compared to averages (kg Co2/year)
        footprint_avg = 14660.85
        footprintbytype_avg = [7252.76, 602.45, 4515.27, 2267.96, 22.41]
    
        footprint_delta=footprint-footprint_avg
        footprintbytype_delta=list(np.array(footprintbytype)-np.array(footprintbytype_avg))
    
        labels_footprint=['electric (kg Co2/year)', 'flight (kg Co2/year)', 'transportation (kg Co2/year)', 'food (kg Co2/year)', 'retail (kg Co2/year)']
        labels_footprintbytype = 'total kg Co2/year'
    
        return footprint, footprintbytype, footprint_delta, footprintbytype_delta, labels_footprint, labels_footprintbytype   
                
                
    def cover_page(pdfname, surveyname, company, date, sampleid):
        #st.write("hii from cover")
        c=canvas.Canvas(pdfname, pagesize=portrait(letter))
        c.drawImage("background.jpg", 0, 200, width=700,height=400, preserveAspectRatio=False)
        c.setFont('Helvetica-Bold', 25, leading=None)
        c.drawCentredString(300,500,"Carbon Footprint Report")
        c.setFont('Helvetica', 16, leading=None)
        c.drawCentredString(300,470,"%s"%(surveyname))
        c.drawCentredString(300,440,"%s"%(date[0:10]))
        c.save()
        return pdfname
    
    
    def make_graphs(individual_means, individual_means_2):
    
        # bar graph compared to average in each category (2 phase bar graph)
        labels = ['Electricity consumption (kwh * 1000)', 'No. of flights per year', 'Miles travelled per year (thousands)', 'No. of uber trips per year', 'Food choice (tons of CO2 emissions/year)']
        population_means = [11.698, 2.1, 15, 7.86, 2.5]
        population_means=list(map(int,population_means))
    
        print(labels)
        print(individual_means)
        print(population_means)
    
        x = np.arange(len(labels))  # the label locations
        width = 0.35  # the width of the bars
    
        fig, ax = plt.subplots()
        rects1 = ax.bar(x - width/2, individual_means, width, label='Your score', color='green')
        rects2 = ax.bar(x + width/2, population_means, width, label='Average score', color='red')
    
        # Add some text for labels, title and custom x-axis tick labels, etc.
        ax.set_ylabel('Scores')
        ax.set_title('Scores by label')
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation='vertical') #rotation='vertical', fontsize='x-small',
        ax.legend()
    
        def autolabel(rects):
            """Attach a text label above each bar in *rects*, displaying its height."""
            for rect in rects:
                height = rect.get_height()
                ax.annotate('{}'.format(height),
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 2),  # 3 points vertical offset
                            textcoords="offset points",
                            ha='center', va='bottom')
    
    
        autolabel(rects1)
        autolabel(rects2)
    
        fig.tight_layout()
        # plt.show()
        plt.savefig('bar.png', format="png")
    
    
        # bar 2 
        labels = ['electricity', 'flights', 'transportation', 'food', 'retail']
        population_means = [7252.76, 602.45, 4515.27, 2267.96, 22.41]
        population_means=list(map(int,population_means))
        individual_means_2=list(map(int, individual_means_2))
    
        print(labels)
        print(individual_means_2)
        print(population_means)
    
        x = np.arange(len(labels))  # the label locations
        width = 0.35  # the width of the bars
    
        fig, ax = plt.subplots()
        rects1 = ax.bar(x - width/2, individual_means_2, width, label='Your score', color='#5dcf60')
        rects2 = ax.bar(x + width/2, population_means, width, label='Average score', color='#595959')
    
        # Add some text for labels, title and custom x-axis tick labels, etc.
        ax.set_ylabel('kg Co2/year')
        ax.set_title('Scores by label')
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation='vertical') #rotation='vertical', fontsize='x-small',
        ax.legend()
    
        def autolabel(rects):
            """Attach a text label above each bar in *rects*, displaying its height."""
            for rect in rects:
                height = rect.get_height()
                ax.annotate('{}'.format(height),
                            xy=(rect.get_x() + rect.get_width(), height),
                            xytext=(0, 3),  # 3 points vertical offset
                            textcoords="offset points",
                            ha='center', va='bottom')
    
    
        autolabel(rects1)
        autolabel(rects2)
    
        fig.tight_layout()
        # plt.show()
        plt.savefig('bar_2.png', format="png")
    
        # % of contributions to your carbon footpint
        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
    
        labels = ['Electricity', 'Flights', 'Transport', 'Food', 'Retail']
        fracs = [individual_means_2[0], individual_means_2[1], individual_means_2[2], individual_means_2[3], individual_means_2[4]]
        colors = ['yellow', 'orange', 'pink', 'green', 'red', 'violet']
        explode = (0, 0, 0, 0, 0)
    
        # We want to draw the shadow for each pie but we will not use "shadow"
        # option as it does'n save the references to the shadow patches.
        pies = ax.pie(fracs, explode=explode, labels=labels, autopct='%1.1f%%', colors=colors)
    
        for w in pies[0]:
            # set the id with the label.
            w.set_gid(w.get_label())
    
            # we don't want to draw the edge of the pie
            w.set_edgecolor("none")
    
        for w in pies[0]:
            # create shadow patch
            s = Shadow(w, -0.01, -0.01)
            s.set_gid(w.get_gid() + "_shadow")
            s.set_zorder(w.get_zorder() - 0.1)
            ax.add_patch(s)
    
        # save
        plt.savefig('pi.png', format="png")
    
    
    def make_bar_pdf(pdfname, logo):
    
        c=canvas.Canvas(pdfname, pagesize=portrait(letter))
        c.setFont('Helvetica-Bold', 16, leading=None)
        c.drawCentredString(300,600,"Your carbon consumption relative to the average American (units)")
        c.drawImage(logo, 0, 200, width=600,height=300, preserveAspectRatio=True)
        c.save()

    
    def make_pie_pdf(pdfname):
        c=canvas.Canvas(pdfname, pagesize=portrait(letter))
        logo="pi.png"
        c.setFont('Helvetica-Bold', 16, leading=None)
        c.drawCentredString(300,600,"Your carbon consumption: by category")
        c.drawImage(logo, 0, 200, width=600,height=300, preserveAspectRatio=True)
        c.save()
    
    
    def make_lastpage(pdfname):
        c=canvas.Canvas(pdfname, pagesize=portrait(letter))
        c.drawImage("background.jpg", 0, 200, width=700,height=400, preserveAspectRatio=False)
        c.setFont('Helvetica-Bold', 20, leading=None)
        c.drawCentredString(300,500,"Thanks for taking this survey.")
        c.setFont('Helvetica', 16, leading=None)
        c.drawCentredString(300, 460, "Reduce your carbon footprint!")
        c.save()    
    
    def merge_pdfs(pdflist):
        merger = PdfMerger()
        for pdf in pdflist:
           merger.append(PdfReader(open(pdf, 'rb')))
        merger.write('final_report.pdf')
        
    def get_binary_file_downloader_html(bin_file, file_label='File'):
            with open(bin_file, 'rb') as f:
                data = f.read()           
            bin_str = base64.b64encode(data).decode()
            href = f'<a href="data:application/octet-stream;base64,{bin_str}" download="{os.path.basename(bin_file)}">Download {file_label}</a>'
            return href

    def improvement_pdf(pdfname, truthlist):
        # areas of improvement (recommend areas of improvement)
        # how to be more involved
        
        c=canvas.Canvas(pdfname, pagesize=portrait(letter))
        c.setFont('Helvetica-Bold', 16, leading=None)
        c.drawCentredString(300,600,"Here are some recommendations to become a better citizen:")

        recommendation_list=['✍ Take efforts to clean nature (e.g. like recycling off the street)',
                       '✍ Collect and use clean energy (e.g. wind or solar power)',
                       '✍ Eat less meat when you can going out',
                       '✍ Buy local produce from farmers markets on weekends',
                       '✍ Exercise instead of uber to work',
                       '✍ Make eco-friendly purchases',
                       '✍ Plant trees or take care of plants in your house',
                       '✍ Try to recycle and reduce waste when you can',
                       '✍Take fewer flights and/or reduce your own transportation',
                       '✍ Turn off your air conditioning unit and use less electricity when you can',
                       '✍ Take showers for 5 minutes or less']
    
        recommendations=list()
        for i in range(len(truthlist)):
            if truthlist[i] == False:
                recommendations.append(recommendation_list[i])
    
        c.setFont('Helvetica', 11, leading=None)
        height=550
        for i in range(len(recommendations)):
            c.drawCentredString(300,height,recommendations[i])
            height=height-20
    
        c.save()
            
    
    ##############################################################################
    ##                            MAIN SCRIPT                                   ##
    ##############################################################################
    if nav == "Individual Emission 👨‍👩‍👧‍👦":
        st.markdown(f"""<h1 style='text-align: center; font-weight:bold;color:white;background-color:green;font-size:20pt;'>Let's find your carbon footprint! 😉 </h1>""",unsafe_allow_html=True)
        st.write("")
        st.image("https://i.pinimg.com/originals/7e/69/ec/7e69eca344ca1465da94d698ded08e8e.gif", width=300)
        email=st.text_input('What is your email? \n')
        if email:
            st.image("https://i.pinimg.com/originals/61/b2/d3/61b2d33f39927afa72e5f57a28cc7c83.gif", width=300)
            answer_1 = st.text_input('How many people are in your household? (e.g. 2) \n')
            answer_1=clean_answer(answer_1)
            
            if answer_1: 
                st.image("https://cdn.dribbble.com/users/282923/screenshots/11050247/paymentsbilling.gif", width=300)
                answer_2 = st.text_input('What is your electric bill monthly?  (e.g.₹ 50) \n')
                answer_2=clean_answer(answer_2)
                if answer_2:
                    st.image("https://cdn.dribbble.com/users/846207/screenshots/7617197/media/e87a923768846bc12f00539d66e80931.gif", width=300)
                    answer_3 = st.text_input('How many flights do you take per year? (e.g. 10) \n')
                    answer_3=clean_answer(answer_3)
                    if answer_3:
                        st.image("https://i.pinimg.com/originals/1f/b3/fd/1fb3fd287f851da90e3ec73b10be294a.gif",width=300)             
                        answer_4 = st.text_input('Do you own a car? (e.g. n | y) \n')
                        answer_4=clean_answer(answer_4)
                        if answer_4:
                            answer_5 = st.text_input('What is your average distance to commute to/from work in miles - for example 21? (e.g. 10) \n')
                            answer_5=clean_answer(answer_5)
                            if answer_5:
                                st.image("https://cdn.dribbble.com/users/2374064/screenshots/4737393/bus-truning.gif",width=250)
                                answer_6= st.text_input('Do you use public transportation? (e.g. y)\n')
                                answer_6=clean_answer(answer_6)
                                if answer_6:
                                    st.image("https://thedutchdoor.in/wp-content/uploads/2019/08/rickshaw.gif",width=200)
                                    answer_7 = st.text_input('Do you use uber/redtaxi/ola or another ride sharing platforms? (e.g. y) \n')
                                    answer_7=clean_answer(answer_7)                                   
                                    if answer_7 == 'yes':
                                        answer_8 =st.text_input("How many ride-sharing trips do you complete per month? (e.g. 10) \n")
                                        answer_8=clean_answer(answer_8)
                                    else:
                                        answer_8 = '0'
                                    if answer_7:
                                        st.image("https://static.wixstatic.com/media/975a91_ca6e48ebcffe42ecbcdfca8b306d4d17~mv2.gif",width=300)
                                        answer_9 =st.text_input('Are you a vegetarian? (e.g. n) \n')
                                        answer_9=clean_answer(answer_9)
                                        if answer_9:
                                            answer_10= st.text_input('Do you eat meat more than 3 times each week? (e.g. y) \n')
                                            answer_10=clean_answer(answer_10)
                                            if answer_10:     
                                                answer_11 = st.text_input('How much money do you spend on stuffs per month ? (e.g. ₹150) \n')
                                                answer_11=clean_answer(answer_11)
                
     
            answers=[answer_1, answer_2, answer_3, answer_4, answer_5,
                     answer_6, answer_7, answer_8, answer_9, answer_10, answer_11]
        

        
        ## report on recommendations pop up + saved in directory
        footprint, footprintbytype, footprint_delta, footprintbytype_delta, labels_footprint, labels_footprintbytype =calculate_footprint(answers)
        
        data = {'email': email,
                'answers': answers,
                'footprint': footprint,
                'footprintbytype': footprintbytype,
                'footprint_delta': footprint_delta,
                'footprintbytype_delta': footprintbytype_delta,
                'labels_footprint': labels_footprint,
                'labels_footprintbytype': labels_footprintbytype}
        if(st.button('Predict')):
            st.success("YOUR EMISSION is {} kilograms of CO2/year".format(round(footprint,2)))
      
            
            ########################################################
            ##              Now create the PDF                    ##
            ########################################################
    
           

            # individual_means = ['Electricity consumption (kwh * 1000)', '# of flights per year', '# of driven miles/year (thousands)', '# of uber trips/year', 'food choice (tons of CO2 emissions/year)']
            if answer_4 == 'yes' and answer_6 == 'no':
                individual_means = [(int(answer_2)/0.1327)*12/1000, int(answer_3), int(answer_5)*220*2/1000, int(answer_8)*12, footprintbytype[3]/1000]
            elif answer_4 == 'yes' and answer_6 == 'yes':
                individual_means = [(int(answer_2)/0.1327)*12/1000, int(answer_3), int(answer_5)*220*2/1000, int(answer_8)*12,  footprintbytype[3]/1000]
            elif answer_4 == 'no' and answer_6 == 'yes':
                individual_means = [(int(answer_2)/0.1327)*12/1000, int(answer_3), int(answer_5)*220*2/1000, int(answer_8)*12, footprintbytype[3]/1000]
            else:
                individual_means = [(int(answer_2)/0.1327)*12/1000, int(answer_3), 0, int(answer_8)*12, footprintbytype[3]/1000]
            
            individual_means=list(map(int,individual_means))
   
            #['electric (kg Co2/year)', 'flight (kg Co2/year)', 'transportation (kg Co2/year)', 'food (kg Co2/year)', 'retail (kg Co2/year)']
            truthlist=[False, False, False, False, False, False, False, False, False, False, False]
            try:
                cover_page("1.pdf", email, 'User', str(datetime.now()), '100')
                make_graphs(individual_means, footprintbytype)
                make_bar_pdf("2.pdf",'bar.png')
                make_pie_pdf("3.pdf")
                improvement_pdf("4.pdf", truthlist)
                make_lastpage("5.pdf")
                pdflist=["1.pdf","2.pdf","3.pdf","4.pdf","5.pdf"]
                merge_pdfs(pdflist)
                st.markdown(get_binary_file_downloader_html("final_report.pdf", 'Your Final Report📝 '), unsafe_allow_html=True) 
            except Exception as e:
                st.error(e)         

except:
  # Prevent the error from propagating into your Streamlit app.
  pass