# import streamlit as st
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import classification_report

# attendance_df = pd.read_csv("attendance_logs.csv")
# events_df = pd.read_csv("event_participation.csv")
# lms_df = pd.read_csv("lms_usage.csv")

# st.title("📊 Smart Campus Insights")
# st.sidebar.header("🔍 Filters")

# students = attendance_df['StudentID'].unique()
# selected_students = st.sidebar.multiselect("Select Students", students, default=students)

# filtered_attendance = attendance_df[attendance_df['StudentID'].isin(selected_students)]
# filtered_events = events_df[events_df['StudentID'].isin(selected_students)]
# filtered_lms = lms_df[lms_df['StudentID'].isin(selected_students)]

# st.subheader("📋 Attendance Trends")
# attendance_summary = filtered_attendance.groupby(['Date', 'Status']).size().unstack(fill_value=0)
# st.line_chart(attendance_summary)

# st.subheader("🎓 Event Participation")
# event_counts = filtered_events['EventName'].value_counts()
# st.bar_chart(event_counts)

# st.subheader("💻 LMS Usage Patterns")
# lms_summary = filtered_lms.groupby('StudentID')[['SessionDuration', 'PagesViewed']].mean()
# st.dataframe(lms_summary)

# st.subheader("🤖 Predict Student Engagement Risk")

# ml_data = pd.merge(attendance_df.groupby('StudentID')['Status'].apply(lambda x: (x == 'Absent').mean()).reset_index(name='AbsenceRate'),
#                    lms_df.groupby('StudentID')[['SessionDuration', 'PagesViewed']].mean().reset_index(),
#                    on='StudentID')

# ml_data['Engagement'] = (ml_data['AbsenceRate'] < 0.2).astype(int)

# X = ml_data[['AbsenceRate', 'SessionDuration', 'PagesViewed']]
# y = ml_data['Engagement']
# X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
# model = DecisionTreeClassifier()
# model.fit(X_train, y_train)
# y_pred = model.predict(X_test)

# st.text("Model Performance:")
# st.text(classification_report(y_test, y_pred))

# st.subheader("📈 Predict Engagement for New Student")
# absence_rate = st.number_input("Absence Rate (0 to 1)", min_value=0.0, max_value=1.0, value=0.1)
# session_duration = st.number_input("Average Session Duration (minutes)", min_value=0.0, value=30.0)
# pages_viewed = st.number_input("Average Pages Viewed", min_value=0.0, value=10.0)

# if st.button("Predict Engagement"):
#     prediction = model.predict([[absence_rate, session_duration, pages_viewed]])
#     result = "Engaged" if prediction[0] == 1 else "At Risk"
#     st.success(f"Predicted Engagement Status: {result}")

# import streamlit as st
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import classification_report

# # ------------------------
# # Load Data
# # ------------------------
# attendance_df = pd.read_csv("attendance_logs.csv")
# events_df = pd.read_csv("event_participation.csv")
# lms_df = pd.read_csv("lms_usage.csv")

# st.title("📊 Smart Campus Insights")
# st.sidebar.header("🔍 Filters")

# # ------------------------
# # Student ID Checkbox Filter
# # ------------------------
# students = sorted(attendance_df['StudentID'].unique())

# st.sidebar.subheader("Select Students")

# select_all = st.sidebar.checkbox("Select All Students", value=True)

# selected_students = []

# if select_all:
#     selected_students = students
# else:
#     for s in students:
#         if st.sidebar.checkbox(f"Student {s}", value=False):
#             selected_students.append(s)

# # Avoid empty state
# if len(selected_students) == 0:
#     st.warning("Please select at least one student to proceed.")
#     st.stop()

# # ------------------------
# # Filter Data
# # ------------------------
# filtered_attendance = attendance_df[attendance_df['StudentID'].isin(selected_students)]
# filtered_events = events_df[events_df['StudentID'].isin(selected_students)]
# filtered_lms = lms_df[lms_df['StudentID'].isin(selected_students)]

# # ------------------------
# # Attendance Trends
# # ------------------------
# st.subheader("📋 Attendance Trends")
# attendance_summary = filtered_attendance.groupby(['Date', 'Status']).size().unstack(fill_value=0)
# st.line_chart(attendance_summary)

# # ------------------------
# # Event Participation
# # ------------------------
# st.subheader("🎓 Event Participation")
# event_counts = filtered_events['EventName'].value_counts()
# st.bar_chart(event_counts)

# # ------------------------
# # LMS Usage
# # ------------------------
# st.subheader("💻 LMS Usage Patterns")
# lms_summary = filtered_lms.groupby('StudentID')[['SessionDuration', 'PagesViewed']].mean()
# st.dataframe(lms_summary)

# # ------------------------
# # ML Model
# # ------------------------
# st.subheader("🤖 Predict Student Engagement Risk")

# ml_data = pd.merge(
#     attendance_df.groupby('StudentID')['Status']
#     .apply(lambda x: (x == 'Absent').mean())
#     .reset_index(name='AbsenceRate'),
#     lms_df.groupby('StudentID')[['SessionDuration', 'PagesViewed']]
#     .mean()
#     .reset_index(),
#     on='StudentID'
# )

# ml_data['Engagement'] = (ml_data['AbsenceRate'] < 0.2).astype(int)

# X = ml_data[['AbsenceRate', 'SessionDuration', 'PagesViewed']]
# y = ml_data['Engagement']

# X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
# model = DecisionTreeClassifier()
# model.fit(X_train, y_train)
# y_pred = model.predict(X_test)

# st.text("Model Performance:")
# st.text(classification_report(y_test, y_pred))

# # ------------------------
# # Prediction Input
# # ------------------------
# st.subheader("📈 Predict Engagement for New Student")

# absence_rate = st.number_input("Absence Rate (0 to 1)", min_value=0.0, max_value=1.0, value=0.1)
# session_duration = st.number_input("Average Session Duration (minutes)", min_value=0.0, value=30.0)
# pages_viewed = st.number_input("Average Pages Viewed", min_value=0.0, value=10.0)

# if st.button("Predict Engagement"):
#     prediction = model.predict([[absence_rate, session_duration, pages_viewed]])
#     result = "Engaged" if prediction[0] == 1 else "At Risk"
#     st.success(f"Predicted Engagement Status: {result}")
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# -----------------------------
# Load Data
# -----------------------------
attendance_df = pd.read_csv("attendance_logs.csv")
events_df = pd.read_csv("event_participation.csv")
lms_df = pd.read_csv("lms_usage.csv")

st.title("📊 Smart Campus Insights Dashboard")
st.write("Analyze attendance, events, LMS usage, and predict student engagement risk.")

# -----------------------------
# Sidebar – Student Selection UI
# -----------------------------

st.sidebar.header("🎯 Student Filters")

students = sorted(attendance_df['StudentID'].unique())

# Primary: Searchable select box
selected_single = st.sidebar.selectbox(
    "Select a Student (Searchable)",
    options=["All Students"] + students,
    index=0
)

# Advanced section: checkboxes
with st.sidebar.expander("🔧 Advanced: Select Multiple Students"):
    selected_multi = []
    cols = st.columns(2)
    for i, student in enumerate(students):
        with cols[i % 2]:
            if st.checkbox(student, value=False):
                selected_multi.append(student)

# Logic for selection
if selected_single != "All Students":
    selected_students = [selected_single]
elif selected_multi:
    selected_students = selected_multi
else:
    selected_students = students

# -----------------------------
# Filter Data
# -----------------------------
filtered_attendance = attendance_df[attendance_df['StudentID'].isin(selected_students)]
filtered_events = events_df[events_df['StudentID'].isin(selected_students)]
filtered_lms = lms_df[lms_df['StudentID'].isin(selected_students)]

# -----------------------------
# Attendance Trends
# -----------------------------
st.subheader("📋 Attendance Trends")

attendance_summary = filtered_attendance.groupby(['Date', 'Status']).size().unstack(fill_value=0)

if attendance_summary.empty:
    st.warning("No attendance data for selected student(s).")
else:
    st.line_chart(attendance_summary)

# -----------------------------
# Event Participation
# -----------------------------
st.subheader("🎓 Event Participation")

event_counts = filtered_events['EventName'].value_counts()

if event_counts.empty:
    st.warning("No event participation data available.")
else:
    st.bar_chart(event_counts)

# -----------------------------
# LMS Usage
# -----------------------------
st.subheader("💻 LMS Usage Patterns")

lms_summary = filtered_lms.groupby('StudentID')[['SessionDuration', 'PagesViewed']].mean()

if lms_summary.empty:
    st.warning("No LMS usage data available.")
else:
    st.dataframe(lms_summary)

# -----------------------------
# Machine Learning Model
# -----------------------------
st.subheader("🤖 Predict Student Engagement Risk")

ml_data = pd.merge(
    attendance_df.groupby('StudentID')['Status']
    .apply(lambda x: (x == 'Absent').mean())
    .reset_index(name='AbsenceRate'),
    lms_df.groupby('StudentID')[['SessionDuration', 'PagesViewed']]
    .mean()
    .reset_index(),
    on='StudentID'
)

ml_data['Engagement'] = (ml_data['AbsenceRate'] < 0.2).astype(int)

X = ml_data[['AbsenceRate', 'SessionDuration', 'PagesViewed']]
y = ml_data['Engagement']

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

model = DecisionTreeClassifier()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

st.text("Model Performance:")
st.text(classification_report(y_test, y_pred))

# -----------------------------
# Prediction UI
# -----------------------------
st.subheader("📈 Predict Engagement for New Student")

absence_rate = st.number_input("Absence Rate (0 to 1)", 0.0, 1.0, 0.1)
session_duration = st.number_input("Average Session Duration (minutes)", 0.0, 500.0, 30.0)
pages_viewed = st.number_input("Average Pages Viewed", 0.0, 100.0, 10.0)

if st.button("Predict Engagement"):
    prediction = model.predict([[absence_rate, session_duration, pages_viewed]])
    result = "Engaged" if prediction[0] == 1 else "At Risk"
    st.success(f"Predicted Engagement Status: **{result}**")
