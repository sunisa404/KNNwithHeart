from sklearn.neighbors import KNeighborsClassifier
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# st.header('sunisa') 
st.title('การทำนายโรคหัวใจด้วย Machine Learning')
st.image("./img/HeartWithMachineLearning.jpg")

col1, col2 = st.columns(2)

with col1:
   st.header("เป็นโรคหัวใจ")
   st.image("./img/yousick.jpg")

with col2:
   st.header("ไม่เป็นโรคหัวใจ")
   st.image("./img/healty.png")

html_7 = """
<div style="background-color:#c5f18a;padding:15px;border-radius:15px 15px 15px 15px;border-style:'solid';border-color:black">
<center><h5>ข้อมูล หัวใจ หรือข้อมูลสุขภาพหัวใจสำหรับทำนาย</h5></center>
</div>
"""
st.markdown(html_7, unsafe_allow_html=True)
st.markdown("")
st.markdown("")

st.subheader("ข้อมูลส่วนแรก 10 แถว")
dt = pd.read_csv("./data/Heart3.csv")
st.write(dt.head(10))
st.subheader("ข้อมูลส่วนสุดท้าย 10 แถว")
st.write(dt.tail(10))

dt1 = dt['Age'].sum()
dt2 = dt['Sex'].sum()
dt3 = dt['ChestPainType'].sum()
dt4 = dt['RestingBP'].sum()
dt5 = dt['Cholesterol'].sum()
dt6 = dt['FastingBS'].sum()
dt7 = dt['RestingECG'].sum()
dt8 = dt['MaxHR'].sum()
dt9 = dt['ExerciseAngina'].sum()
dt10 = dt['Oldpeak'].sum()
dt11 = dt['ST_Slope'].sum()


dx = [dt1, dt2, dt3, dt4,dt5, dt6, dt7,dt8, dt9, dt10, dt11]
dx2 = pd.DataFrame(dx, index=["d1", "d2", "d3", "d4","d5", "d6", "d7", "d8","d9", "d10", "d11"])

if st.button("แสดงการจินตทัศน์ข้อมูล"):
   #st.write(dt.head(10))
   st.bar_chart(dx2)
   st.button("ไม่แสดงข้อมูล")
else:
    st.write("ไม่แสดงข้อมูล")

html_7 = """
<div style="background-color:#c5f18a;padding:15px;border-radius:15px 15px 15px 15px;border-style:'solid';border-color:black">
<center><h5> Dataset สำหรับทำนาย</h5></center>
</div>
"""
st.markdown(html_7, unsafe_allow_html=True)
st.markdown("")
st.markdown("")

st.subheader("ข้อมูลส่วนแรก 10 แถว")
dt = pd.read_csv("./data/Heart3.csv")
st.write(dt.head(10))
st.subheader("ข้อมูลส่วนสุดท้าย 10 แถว")
st.write(dt.tail(10))

# สถิติพื้นฐาน
st.subheader("📈 สถิติพื้นฐานของข้อมูล")
st.write(dt.describe())

# การเลือกแสดงกราฟตามฟีเจอร์
st.subheader("📌 เลือกฟีเจอร์เพื่อดูการกระจายข้อมูล")
feature = st.selectbox("เลือกฟีเจอร์", dt.columns[:-1])

# วาดกราฟ boxplot
st.write(f"### 🎯 Boxplot: {feature} การจำแนก")
fig, ax = plt.subplots()
sns.boxplot(data=dt, x='HeartDisease', y=feature, ax=ax)
st.pyplot(fig)

# วาด pairplot
if st.checkbox("แสดง Pairplot (ใช้เวลาประมวลผลเล็กน้อย)"):
    st.write("### Pairplot: การกระจายของข้อมูลทั้งหมด")
    fig2 = sns.pairplot(dt, hue='HeartDisease')
    st.pyplot(fig2)

html_8 = """
<div style="background-color:#6BD5DA;padding:15px;border-radius:15px 15px 15px 15px;border-style:'solid';border-color:black">
<center><h5>ทำนายข้อมูล</h5></center>
</div>
"""
st.markdown(html_8, unsafe_allow_html=True)
st.markdown("")

A1 = st.number_input("กรุณาเลือกข้อมูล Age")
A2 = st.number_input("กรุณาเลือกข้อมูล Sex")
A3 = st.number_input("กรุณาเลือกข้อมูล ChestPainType")
A4 = st.number_input("กรุณาเลือกข้อมูล RestingBP")
A5 = st.number_input("กรุณาเลือกข้อมูล Cholesterol")
A6 = st.number_input("กรุณาเลือกข้อมูล FastingBS")
A7 = st.number_input("กรุณาเลือกข้อมูล RestingECG")
A8 = st.number_input("กรุณาเลือกข้อมูล MaxHR")
A9 = st.number_input("กรุณาเลือกข้อมูล ExerciseAngina")
A10 = st.number_input("กรุณาเลือกข้อมูล Oldpeak")
A11 = st.number_input("กรุณาเลือกข้อมูล ST_Slope")

if st.button("ทำนายผล"):
    #st.write("ทำนาย")
   dt = pd.read_csv("./data/Heart3.csv") 
   X = dt.drop('HeartDisease', axis=1)
   y = dt.HeartDisease   

   Knn_model = KNeighborsClassifier(n_neighbors=3)
   Knn_model.fit(X, y)  
    
   x_input = np.array([[A1, A2,A3, A4,A5, A6,A7, A8,A9, A10,A11]])
   st.write(Knn_model.predict(x_input))
   
   out=Knn_model.predict(x_input)

   if out[0] == 1:
    st.image("./img/yousick.jpg")
   else:      
    st.image("./img/healty.png")
else:
    st.write("ไม่ทำนาย")
