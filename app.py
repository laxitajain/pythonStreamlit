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

# import streamlit as st
# import pandas as pd
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import classification_report, accuracy_score

# # -----------------------------
# # Load Data
# # -----------------------------
# attendance_df = pd.read_csv("attendance_logs.csv")
# events_df = pd.read_csv("event_participation.csv")
# lms_df = pd.read_csv("lms_usage.csv")

# st.title("📊 Smart Campus Insights")
# st.sidebar.header("Settings")

# students = attendance_df['StudentID'].unique()

# # -----------------------------
# # BEST UI: Student selection via expandable checkboxes
# # -----------------------------
# with st.sidebar.expander("🎓 Select Students"):
#     selected_students = []
#     for sid in students:
#         if st.checkbox(sid, value=True):
#             selected_students.append(sid)

# # -----------------------------
# # Apply student filter
# # -----------------------------
# filtered_attendance = attendance_df[attendance_df['StudentID'].isin(selected_students)]
# filtered_events = events_df[events_df['StudentID'].isin(selected_students)]
# filtered_lms = lms_df[lms_df['StudentID'].isin(selected_students)]

# st.subheader("📋 Attendance Trends")
# attendance_summary = filtered_attendance.groupby(['Date', 'Status']).size().unstack(fill_value=0)
# st.line_chart(attendance_summary)

# st.subheader("🎓 Event Participation")
# st.bar_chart(filtered_events['EventName'].value_counts())

# st.subheader("💻 LMS Usage Patterns")
# lms_summary = filtered_lms.groupby('StudentID')[['SessionDuration', 'PagesViewed']].mean()
# st.dataframe(lms_summary)

# # -----------------------------
# # Build ML dataset
# # -----------------------------
# ml_data = pd.merge(
#     attendance_df.groupby('StudentID')['Status']
#     .apply(lambda x: (x == 'Absent').mean())
#     .reset_index(name='AbsenceRate'),
#     lms_df.groupby('StudentID')[['SessionDuration', 'PagesViewed']].mean().reset_index(),
#     on='StudentID'
# )

# ml_data['Engagement'] = (ml_data['AbsenceRate'] < 0.2).astype(int)

# # -----------------------------
# # DATA POISONING FEATURE
# # -----------------------------
# st.sidebar.subheader("⚠️ Data Poisoning Controls")
# poison_rate = st.sidebar.slider("Poisoning Level (%)", 0, 50, 0)

# poisoned_data = ml_data.copy()

# if poison_rate > 0:
#     st.sidebar.write(f"Injecting {poison_rate}% poisoned samples…")

#     n = len(poisoned_data)
#     k = int(n * (poison_rate / 100))

#     # Choose k random rows to poison
#     poison_idx = poisoned_data.sample(k).index

#     # LABEL FLIPPING ATTACK
#     poisoned_data.loc[poison_idx, "Engagement"] = 1 - poisoned_data.loc[poison_idx, "Engagement"]

#     st.warning(f"⚠️ {k} labels flipped (data poisoned).")

# # -----------------------------
# # Train Model
# # -----------------------------
# X = poisoned_data[['AbsenceRate', 'SessionDuration', 'PagesViewed']]
# y = poisoned_data['Engagement']

# X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
# model = DecisionTreeClassifier()
# model.fit(X_train, y_train)
# y_pred = model.predict(X_test)

# st.subheader("🤖 Model Performance")
# st.text(classification_report(y_test, y_pred))
# st.metric("Accuracy", f"{accuracy_score(y_test, y_pred)*100:.2f}%")

# # -----------------------------
# # Predict new student
# # -----------------------------
# st.subheader("📈 Predict Engagement for New Student")

# absence_rate = st.number_input("Absence Rate (0 to 1)", min_value=0.0, max_value=1.0, value=0.1)
# session_duration = st.number_input("Average Session Duration (minutes)", min_value=0.0, value=30.0)
# pages_viewed = st.number_input("Average Pages Viewed", min_value=0.0, value=10.0)

# if st.button("Predict Engagement"):
#     pred = model.predict([[absence_rate, session_duration, pages_viewed]])[0]
#     result = "Engaged" if pred == 1 else "At Risk"
#     st.success(f"Predicted Engagement Status: {result}")


# import streamlit as st
# import pandas as pd
# import numpy as np
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import classification_report, accuracy_score

# # ---------------------------
# # Load Data
# # ---------------------------
# attendance_df = pd.read_csv("attendance_logs.csv")
# events_df = pd.read_csv("event_participation.csv")
# lms_df = pd.read_csv("lms_usage.csv")

# # ---------------------------
# # Sidebar UI
# # ---------------------------
# st.sidebar.header("🎛️ Student Filter")

# students = attendance_df['StudentID'].unique()

# # Checkbox to select all
# select_all = st.sidebar.checkbox("Select All Students", value=True)

# # Multiselect with search bar
# if select_all:
#     selected_students = st.sidebar.multiselect(
#         "Choose Students", students, default=list(students)
#     )
# else:
#     selected_students = st.sidebar.multiselect(
#         "Choose Students", students
#     )

# # ---------------------------
# # Filter Data
# # ---------------------------
# filtered_attendance = attendance_df[attendance_df["StudentID"].isin(selected_students)]
# filtered_events = events_df[events_df["StudentID"].isin(selected_students)]
# filtered_lms = lms_df[lms_df["StudentID"].isin(selected_students)]

# st.title("📊 Smart Campus Insights Dashboard")

# # ---------------------------
# # Attendance Chart
# # ---------------------------
# st.subheader("📋 Attendance Trends")
# attendance_summary = filtered_attendance.groupby(["Date", "Status"]).size().unstack(fill_value=0)
# st.line_chart(attendance_summary)

# # ---------------------------
# # Participation Chart
# # ---------------------------
# st.subheader("🎓 Event Participation")
# event_counts = filtered_events["EventName"].value_counts()
# st.bar_chart(event_counts)

# # ---------------------------
# # LMS Usage Table
# # ---------------------------
# st.subheader("💻 LMS Usage Patterns")
# lms_summary = filtered_lms.groupby("StudentID")[["SessionDuration", "PagesViewed"]].mean()
# st.dataframe(lms_summary)

# # ---------------------------
# # Build Clean ML Dataset
# # ---------------------------
# st.subheader("🤖 Student Engagement Prediction Model")

# # Compute absence rate
# absence_rate_df = attendance_df.groupby("StudentID")["Status"].apply(
#     lambda x: (x == "Absent").mean()
# ).reset_index(name="AbsenceRate")

# lms_summary_df = lms_df.groupby("StudentID")[["SessionDuration", "PagesViewed"]].mean().reset_index()

# ml_data = pd.merge(absence_rate_df, lms_summary_df, on="StudentID")
# ml_data["Engagement"] = (ml_data["AbsenceRate"] < 0.2).astype(int)

# # ---------------------------
# # Data Poisoning Controls
# # ---------------------------
# st.sidebar.header("⚠️ Data Poisoning Settings")

# apply_poison = st.sidebar.checkbox("Enable Data Poisoning", value=False)

# poison_strength = st.sidebar.slider(
#     "Poisoning Strength (0 = none, 1 = max)",
#     min_value=0.0, max_value=1.0, value=0.0
# )

# poisoned_data = ml_data.copy()

# if apply_poison:
#     # Increase absence rate artificially
#     poisoned_data["AbsenceRate"] = poisoned_data["AbsenceRate"] + poison_strength * 0.5
#     poisoned_data["AbsenceRate"] = poisoned_data["AbsenceRate"].clip(0, 1)

# # ---------------------------
# # Train normal model
# # ---------------------------
# X = ml_data[["AbsenceRate", "SessionDuration", "PagesViewed"]]
# y = ml_data["Engagement"]
# X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
# clean_model = DecisionTreeClassifier()
# clean_model.fit(X_train, y_train)
# clean_pred = clean_model.predict(X_test)
# clean_acc = accuracy_score(y_test, clean_pred)

# # ---------------------------
# # Train poisoned model
# # ---------------------------
# Xp = poisoned_data[["AbsenceRate", "SessionDuration", "PagesViewed"]]
# yp = poisoned_data["Engagement"]
# Xp_train, Xp_test, yp_train, yp_test = train_test_split(Xp, yp, random_state=42)
# poison_model = DecisionTreeClassifier()
# poison_model.fit(Xp_train, yp_train)
# poison_pred = poison_model.predict(Xp_test)
# poison_acc = accuracy_score(yp_test, poison_pred)

# # Show accuracy comparison
# st.subheader("📉 Model Accuracy Comparison")
# st.write(pd.DataFrame({
#     "Model": ["Clean Model", "Poisoned Model"],
#     "Accuracy": [clean_acc, poison_acc]
# }))

# # ---------------------------
# # User Prediction
# # ---------------------------
# st.subheader("📈 Predict Engagement for a New Student")

# absence_rate = st.number_input(
#     "Absence Rate (0 to 1)", min_value=0.0, max_value=1.0, value=0.1
# )

# session_duration = st.number_input(
#     "Average Session Duration (minutes)", min_value=0.0, value=30.0
# )

# pages_viewed = st.number_input(
#     "Average Pages Viewed", min_value=0.0, value=10.0
# )

# if st.button("Predict"):
#     clean_out = clean_model.predict([[absence_rate, session_duration, pages_viewed]])[0]
#     poison_out = poison_model.predict([[absence_rate, session_duration, pages_viewed]])[0]

#     clean_label = "Engaged" if clean_out == 1 else "At Risk"
#     poison_label = "Engaged" if poison_out == 1 else "At Risk"

#     st.success(f"Clean Model Prediction: **{clean_label}**")
#     st.warning(f"Poisoned Model Prediction: **{poison_label}**")

#     if clean_label != poison_label:
#         st.error("⚠️ Data Poisoning Impact: Prediction Changed!")
#     else:
#         st.info("No visible impact of poisoning on this input.")


import streamlit as st
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# =======================================
# LOAD DATA
# =======================================
attendance_df = pd.read_csv("attendance_logs.csv")
events_df = pd.read_csv("event_participation.csv")
lms_df = pd.read_csv("lms_usage.csv")

st.title("📊 Smart Campus Insights")
st.sidebar.header("🔍 Filters")

# =======================================
# STUDENT FILTER UI 
# (Select All + Multi-Select + Search Bar)
# =======================================

students = sorted(attendance_df['StudentID'].unique())  # sorted list

# Search bar for filtering student list
search_query = st.sidebar.text_input("🔎 Search Student ID")

if search_query:
    filtered_students = [s for s in students if search_query.lower() in str(s).lower()]
else:
    filtered_students = students

# Select all toggle
select_all = st.sidebar.checkbox("Select All Students", value=True)

if select_all:
    selected_students = st.sidebar.multiselect("Students", filtered_students, default=filtered_students)
else:
    selected_students = st.sidebar.multiselect("Students", filtered_students)

# Apply filtering
filtered_attendance = attendance_df[attendance_df['StudentID'].isin(selected_students)]
filtered_events = events_df[events_df['StudentID'].isin(selected_students)]
filtered_lms = lms_df[        lms_df['StudentID'].isin(selected_students)]

# =======================================
# VISUALIZATIONS
# =======================================

st.subheader("📋 Attendance Trends")
attendance_summary = filtered_attendance.groupby(['Date', 'Status']).size().unstack(fill_value=0)
st.line_chart(attendance_summary)

st.subheader("🎓 Event Participation")
event_counts = filtered_events['EventName'].value_counts()
st.bar_chart(event_counts)

st.subheader("💻 LMS Usage Patterns")
lms_summary = filtered_lms.groupby('StudentID')[['SessionDuration', 'PagesViewed']].mean()
st.dataframe(lms_summary)

# =======================================
# PREP ML DATA
# =======================================

ml_data = pd.merge(
    attendance_df.groupby('StudentID')['Status'].apply(lambda x: (x == 'Absent').mean()).reset_index(name='AbsenceRate'),
    lms_df.groupby('StudentID')[['SessionDuration', 'PagesViewed']].mean().reset_index(),
    on='StudentID'
)

ml_data['Engagement'] = (ml_data['AbsenceRate'] < 0.2).astype(int)

# =======================================
# DATA POISONING 
# (Safe, controlled, measurable)
# =======================================

st.sidebar.header("⚠️ Data Poisoning Simulation")

poison = st.sidebar.checkbox("Enable Data Poisoning")
poison_strength = st.sidebar.slider("Poison %", 0, 50, 0)

poisoned_data = ml_data.copy()

if poison:
    num_poison = int((poison_strength / 100) * len(poisoned_data))
    poison_indices = np.random.choice(poisoned_data.index, num_poison, replace=False)

    # Flip labels (Engaged ↔ Not Engaged)
    poisoned_data.loc[poison_indices, 'Engagement'] = 1 - poisoned_data.loc[
        poison_indices, 'Engagement'
    ]

    # Add noisy features
    poisoned_data.loc[poison_indices, 'AbsenceRate'] = np.random.uniform(0.7, 1.0, size=num_poison)
    poisoned_data.loc[poison_indices, 'SessionDuration'] = np.random.uniform(0, 200, size=num_poison)
    poisoned_data.loc[poison_indices, 'PagesViewed'] = np.random.randint(0, 50, size=num_poison)

# =======================================
# MODEL TRAINING – CLEAN MODEL
# =======================================

st.subheader("🤖 Model Training")

X_clean = ml_data[['AbsenceRate', 'SessionDuration', 'PagesViewed']]
y_clean = ml_data['Engagement']
X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X_clean, y_clean, random_state=42)

clean_model = DecisionTreeClassifier()
clean_model.fit(X_train_c, y_train_c)
clean_pred = clean_model.predict(X_test_c)
clean_acc = accuracy_score(y_test_c, clean_pred)

# =======================================
# MODEL TRAINING – POISONED MODEL
# =======================================

X_p = poisoned_data[['AbsenceRate', 'SessionDuration', 'PagesViewed']]
y_p = poisoned_data['Engagement']
Xp_train, Xp_test, yp_train, yp_test = train_test_split(X_p, y_p, random_state=42)

poison_model = DecisionTreeClassifier()
poison_model.fit(Xp_train, yp_train)
poison_pred = poison_model.predict(Xp_test)
poison_acc = accuracy_score(yp_test, poison_pred)

# =======================================
# COMPARISON OUTPUT
# =======================================

st.write("### 🧪 Model Accuracy Comparison")
st.write(f"**Clean Model Accuracy:** {clean_acc:.2f}")
st.write(f"**Poisoned Model Accuracy:** {poison_acc:.2f}")

if poison:
    st.error("⚠️ Data Poisoning Enabled — Model reliability decreased.")
else:
    st.success("Data is clean — Model performing normally.")

# =======================================
# PREDICTION UI
# =======================================

st.subheader("📈 Predict Engagement for New Student")

absence_rate = st.number_input("Absence Rate (0 to 1)", 0.0, 1.0, 0.1)
session_duration = st.number_input("Average Session Duration (minutes)", 0.0, 300.0, 30.0)
pages_viewed = st.number_input("Average Pages Viewed", 0.0, 300.0, 10.0)

if st.button("Predict Engagement"):
    pred = poison_model.predict([[absence_rate, session_duration, pages_viewed]])
    result = "Engaged" if pred[0] == 1 else "At Risk"
    st.success(f"Predicted Engagement Status: {result}")
