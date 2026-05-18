import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder

st.set_page_config(page_title="Student Performance Predictor", page_icon="")

# Custom CSS for grey + blue professional background
page_bg = """
<style>
[data-testid="stAppViewContainer"] {
    background: linear-gradient(to right, #ece9e6, #ffffff); /* Grey to white gradient */
}
[data-testid="stHeader"] {
    background: none;
}
[data-testid="stSidebar"] {
    background-color: #f0f4f8; /* Soft blue-grey sidebar */
}
</style>
"""
st.markdown(page_bg, unsafe_allow_html=True)


df = pd.read_csv('student-mat.csv')
categorical_cols = ['school', 'sex', 'address', 'famsize', 'Pstatus', 'Mjob', 'Fjob', 
                   'reason', 'guardian', 'schoolsup', 'famsup', 'paid', 'activities',
                   'nursery', 'higher', 'internet', 'romantic']

label_encoders = {}
df_encoded = df.copy()
for col in categorical_cols:
    le = LabelEncoder()
    df_encoded[col] = le.fit_transform(df[col])
    label_encoders[col] = le

st.title("Student Performance Prediction System")
st.markdown("Fill in the student details below to predict their final grade")

st.info("Note: Grade is out of 20. Passing marks = 10")

# Sidebar Navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to:", ["Prediction Form", "Analytics Dashboard", "About Project"])


# Conditional rendering based on selection
if page == "Prediction Form": 
    st.header("📈 Student Performance Prediction Form")  
    st.subheader("1. Personal Information")
    col1, col2 = st.columns(2)

    with col1:
        school = st.selectbox(
            "School Name", 
            ['GOVT', 'MS (Mehran school)'],
            help="GP = Govt pene ladho colony School, MS = mehran school of science and technology"
        )
        school = 'GP' if 'GP' in school else 'MS'
        
        sex = st.selectbox(
            "Gender", 
            ['F (Female)', 'M (Male)','Other'],
            help="Select student's gender"
        )
        sex = 'F' if 'F' in sex else 'M'
        
        age = st.number_input(
            "Student Age", 
            min_value=15, 
            max_value=22, 
            value=16,
            help="Age between 15-22 years"
        )
        
        address = st.selectbox(
            "Home Address", 
            ['U (Urban)', 'R (Rural)'],
            help="U = Urban city area, R = Rural village area"
        )
        address = 'U' if 'U' in address else 'R'

    with col2:
        famsize = st.selectbox(
            "Family Size", 
            ['GT3 (Greater than 3)', 'LE3 (Less than or equal to 3)'],
            help="Total family members"
        )
        famsize = 'GT3' if 'GT3' in famsize else 'LE3'
        
        Pstatus = st.selectbox(
            "Parents Living Together?", 
            ['T (Together)', 'A (Apart)'],
            help="Are parents living together or apart?"
        )
        Pstatus = 'T' if 'T' in Pstatus else 'A'
        
        guardian = st.selectbox(
            "Guardian (Who takes care?)", 
            ['mother', 'father', 'other'],
            help="Who is the student's main guardian?"
        )

    st.subheader("2. Parents Education & Job")
    col3, col4 = st.columns(2)

    with col3:
        st.markdown("**Mother's Details**")
        Medu = st.select_slider(
            "Mother Education Level",
            options=[0, 1, 2, 3, 4],
            value=2,
            help="0=none, 1=primary, 2=5-9 grade, 3=secondary, 4=higher education"
        )
        Mjob = st.selectbox(
            "Mother's Job", 
            ['teacher', 'health', 'services', 'at_home', 'other'],
            help="What does mother do for work?"
        )
    with col4:
        st.markdown("**Father's Details**")
        Fedu = st.select_slider(
            "Father Education Level",
            options=[0, 1, 2, 3, 4],
            value=2,
            help="0=none, 1=primary, 2=5-9 grade, 3=secondary, 4=higher education"
        )
        Fjob = st.selectbox(
            "Father's Job", 
            ['teacher', 'health', 'services', 'at_home', 'other'],
            help="What does father do for work?"
        )

    st.subheader("3. Study Related")
    col5, col6 = st.columns(2)


    with col5:
        reason = st.selectbox(
            "Why choose this school?", 
            ['home (near home)', 'reputation (good name)', 'course (courses)', 'other'],
            help="Main reason for choosing this school"
        )
        reason = reason.split(' ')[0]
        
        studytime = st.select_slider(
            "Weekly Study Time",
            options=[1, 2, 3, 4],
            value=2,
            help="1=<2hrs, 2=2-5hrs, 3=5-10hrs, 4=>10hrs"
        )
        
        failures = st.select_slider(
            "Past Class Failures",
            options=[0, 1, 2, 3],
            value=0,
            help="Number of times student has failed in past"
        )

    with col6:
        traveltime = st.select_slider(
            "Home to School Travel Time",
            options=[1, 2, 3, 4],
            value=1,
            help="1=<15min, 2=15-30min, 3=30-60min, 4=>60min"
        )
        
        absences = st.number_input(
            "Number of School Absences", 
            min_value=0, 
            max_value=100, 
            value=0,
            help="Total days absent from school"
        )

    st.subheader("4. Support & Activities")
    col7, col8 = st.columns(2)    


    with col7:
        schoolsup = st.selectbox("Extra School Support?", ['yes', 'no'])
        famsup = st.selectbox("Family Support?", ['yes', 'no'])
        paid = st.selectbox("Paid Extra Classes?", ['yes', 'no'])
        nursery = st.selectbox("Attended Nursery School?", ['yes', 'no'])

    with col8:
        higher = st.selectbox("Wants Higher Education?", ['yes', 'no'])
        internet = st.selectbox("Internet at Home?", ['yes', 'no'])
        romantic = st.selectbox("In Romantic Relationship?", ['yes', 'no'])
        activities = st.selectbox("Extra Activities?", ['yes', 'no'])

    st.subheader("5. Lifestyle & Social")
    col9, col10 = st.columns(2)
    

    with col9:
      famrel = st.select_slider("Family Relationship Quality",options=[1, 2, 3, 4, 5],value=4,help="1=very bad, 5=excellent")
      freetime = st.select_slider("Free Time After School",options=[1, 2, 3, 4, 5],value=3, help="1=very low, 5=very high")
      goout = st.select_slider("Going Out with Friends",options=[1, 2, 3, 4, 5],value=3,help="1=very low, 5=very high" )

    with col10:
     Dalc = st.select_slider(
        "Workday Alcohol Consumption",
        options=[1, 2, 3, 4, 5],
        value=1,
        help="1=very low, 5=very high"
    )
     Walc = st.select_slider(
        "Weekend Alcohol Consumption",
        options=[1, 2, 3, 4, 5],
        value=1,
        help="1=very low, 5=very high"
    )
     health = st.select_slider(
        "Current Health Status",
        options=[1, 2, 3, 4, 5],
        value=5,
        help="1=very bad, 5=very good"
    )
     st.markdown("---")

    
    if st.button("Predict Performance", type="primary", use_container_width=True):
        input_data = {
            'school': school, 'sex': sex, 'age': age, 'address': address,
            'famsize': famsize, 'Pstatus': Pstatus, 'Medu': Medu, 'Fedu': Fedu,
            'Mjob': Mjob, 'Fjob': Fjob, 'reason': reason, 'guardian': guardian,
            'traveltime': traveltime, 'studytime': studytime, 'failures': failures,
            'schoolsup': schoolsup, 'famsup': famsup, 'paid': paid,
            'activities': activities, 'nursery': nursery, 'higher': higher,
            'internet': internet, 'romantic': romantic, 'famrel': famrel,
            'freetime': freetime, 'goout': goout, 'Dalc': Dalc, 'Walc': Walc,
            'health': health, 'absences': absences
        }
        
        input_df = pd.DataFrame([input_data])
        
        for col in categorical_cols:
            input_df[col] = label_encoders[col].transform(input_df[col])
        
        scaler = StandardScaler()
        X_train = df_encoded.drop(['G1', 'G2', 'G3'], axis=1)
        scaler.fit(X_train)
        input_scaled = scaler.transform(input_df)
        
        from sklearn.neural_network import MLPRegressor
        model = MLPRegressor(
            hidden_layer_sizes=(128, 64, 32, 16),
            activation='relu',
            solver='adam',
            max_iter=500,
            random_state=42
        )
        model.fit(scaler.transform(X_train), df['G3'])
        
        prediction = model.predict(input_scaled)[0]
        prediction = max(0, min(20, prediction))
        
        st.markdown("---")
        st.subheader("Prediction Result")
        
        col_res1, col_res2, col_res3 = st.columns(3)
        with col_res1:
            st.metric("Predicted Grade", f"{prediction:.1f}/20")
        with col_res2:
            status = "PASS" if prediction >= 10 else "FAIL"
            color = "green" if status == "PASS" else "red"
            st.markdown(f"### :{color}[{status}]")
        with col_res3:
            percentage = (prediction / 20) * 100
            st.metric("Score", f"{percentage:.0f}%")
        
        if prediction >= 15:
            st.success("Excellent performance! This student is likely to excel!")
        elif prediction >= 10:
            st.info("Good performance! The student should pass.")
        else:
            st.warning("At risk! Extra support recommended.")

    st.markdown("---")
    st.caption("Student Performance Prediction System | ANN Model")

elif page == "Analytics Dashboard":
    st.header("📊 Analytics Dashboard")
    st.write("Professional insights into student performance data")

    import matplotlib.pyplot as plt
    import seaborn as sns
    import plotly.express as px

    # Histogram of Final Grades
    fig1, ax1 = plt.subplots()
    sns.histplot(df['G3'], bins=10, kde=True, color="skyblue", ax=ax1)
    ax1.set_title("Distribution of Final Grades (G3)")
    ax1.set_xlabel("Final Grade")
    ax1.set_ylabel("Number of Students")
    st.pyplot(fig1)

    # Boxplot: Absences vs Grades
    fig2, ax2 = plt.subplots()
    sns.boxplot(x=df['G3'], y=df['absences'], palette="coolwarm", ax=ax2)
    ax2.set_title("Absences vs Final Grade")
    ax2.set_xlabel("Final Grade")
    ax2.set_ylabel("Absences")
    st.pyplot(fig2)

    # Scatterplot with regression line
    fig3 = px.scatter(df, x="studytime", y="G3", color="sex",
                      trendline="ols", title="Study Time vs Final Grade")
    st.plotly_chart(fig3, use_container_width=True)

    # Bar chart: Average grade by school
    avg_grade = df.groupby("school")['G3'].mean().reset_index()
    fig4 = px.bar(avg_grade, x="school", y="G3", color="school",
                  title="Average Final Grade by School")
    st.plotly_chart(fig4, use_container_width=True)

elif page == "About Project":
    st.header("ℹ️ About This Project")

    st.markdown("""
    ###  Project Overview
    This application predicts **student performance** using socio-economic and academic factors.  
    It leverages **Machine Learning (ANN - MLPRegressor)** to estimate final grades.

    ### Dataset
    - Source: [UCI Student Performance Dataset](https://archive.ics.uci.edu/ml/datasets/Student+Performance)
    - Records: 395 students
    - Features: Demographics, family background, study habits, lifestyle, and academic history

    ### Model
    - Algorithm: Artificial Neural Network (MLPRegressor)
    - Layers: 128 → 64 → 32 → 16 (ReLU activation)
    - Optimizer: Adam
    - Target: Final Grade (G3)

    ### Features
    - Multi-page navigation (Prediction Form, Analytics Dashboard, About Project)
    - Interactive charts and professional visualizations
    - Prediction results with pass/fail status and percentage
    - Clean UI with custom styling

    ### Purpose
    - Help educators identify at-risk students
    - Provide insights into factors affecting performance
    - Encourage data-driven decision making in education
    """)

    st.success("✅ This project demonstrates how AI can support education by predicting student outcomes.")

