import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression, ElasticNet
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.svm import SVR
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
import plotly.express as px
from keras.models import Sequential
from keras.layers import Dense

st.set_page_config(page_title="AI Regression Models", page_icon="📊", layout="wide")

def mm(model__, dic_param):
    main_pip = Pipeline(steps=[
        ("preprocces", preproccessing),
        ("model", model__)
    ])
    param_dic = dic_param
    with st.spinner("⏳ Training your model, please wait..."):
        final_model = RandomizedSearchCV(main_pip, param_distributions=param_dic, n_iter=5, cv=5)
        final_model.fit(data.drop(target, axis="columns"), data[target])

    train_data = data.drop(target, axis="columns")
    train_columns = train_data.columns
    numerical_train_data = train_data.select_dtypes(include=["number"]).columns
    categorical_train_data = train_data.select_dtypes(include=["object"]).columns
    values = {}
    st.subheader("🔢 Enter the input values for prediction")
    for x in train_columns:
        if x in numerical_train_data:
            values[x] = st.number_input(f"Enter the value for {x} : ")
        else:
            values[x] = st.selectbox(f"Choose the value for {x}:", options=train_data[x].unique().tolist())
    test_data = pd.DataFrame([values])
    st.write("### ✅ Your Input Data")
    st.write(test_data)

    bu = st.button("🚀 Predict the value")
    if bu:
        with st.spinner("🔍 Calculating prediction..."):
            predict_data = final_model.predict(test_data)
            st.success(f"✅ Predicted Value: {predict_data[0]}")

st.title("📌 AI Regression Models Playground")
st.markdown("""
Welcome to **AI Model Playground** 🚀.  
Here you can:
- **Upload your dataset** 📂  
- **Visualize your data** with interactive graphs 📊  
- **Train powerful machine learning models** 🤖  
- **Predict outcomes** based on your input!  
""")

file = st.file_uploader("📂 Upload your CSV dataset:", type=["csv"])

if file is not None:
    data = pd.read_csv(file)
    data.reset_index(drop=True, inplace=True)

    columns = data.columns
    choosed_cols = st.multiselect("📌 Select columns to display:", default=columns, options=columns)
    NumberOfRows = st.slider("🔍 Select number of rows to show:", max_value=1000, min_value=5, step=5)
    st.write("### 🔎 Data Preview")
    st.write(data.loc[0:NumberOfRows, choosed_cols])

    st.subheader("🎯 Select your target column")
    numerical_columns_target = data.select_dtypes(include="number").columns
    target = st.selectbox("Choose the target column:", options=numerical_columns_target)
    columns = columns.to_list()
    columns.remove(target)

    true_col_data = data.drop(target, axis="columns")
    numerical_data = true_col_data.select_dtypes(include=["number"]).columns
    categorical_data = true_col_data.select_dtypes(include=["object"]).columns

    if data[target].dtype == "object":
        imputer = SimpleImputer(strategy="most_frequent")
        data[target] = imputer.fit_transform([data[target]])[0]
        encoder = LabelEncoder()
        data[target] = encoder.fit_transform(data[target])
    else:
        imputer = SimpleImputer(strategy="mean")
        data[target] = (imputer.fit_transform([data[target]]))[0]

    st.subheader("📈 Choose your Regression Model")
    numerical_pipe = Pipeline(steps=[
        ("mean", SimpleImputer(strategy="mean")),
        ("standardization", StandardScaler()),
    ])
    cat_pipe = Pipeline(steps=[
        ("median", SimpleImputer(strategy="most_frequent")),
        ("encoding", OneHotEncoder())
    ])
    preproccessing = ColumnTransformer(transformers=[
        ("numerical", numerical_pipe, numerical_data),
        ("categorical", cat_pipe, categorical_data)
    ])

    choosed_model = st.selectbox("Select Model", options=[
        "LinearRegression", "ElasticNet", "DecisionTreeRegressor", "RandomForestRegressor",
        "GradientBoostingRegressor", "XGBoost", "SVR", "KNeighborsRegressor", "DNN"
    ])

    # Same model logic (no changes)
    if choosed_model == "LinearRegression":
        main_pip = Pipeline(steps=[
            ("preprocces", preproccessing),
            ("model", LinearRegression())
        ])
        main_pip.fit(data.drop(target, axis="columns"), data[target])

        st.subheader("🔢 Enter the input values for prediction")
        train_data = data.drop(target, axis="columns")
        train_columns = train_data.columns
        numerical_train_data = train_data.select_dtypes(include=["number"]).columns
        categorical_train_data = train_data.select_dtypes(include=["object"]).columns
        values = {}
        for x in train_columns:
            if x in numerical_train_data:
                values[x] = st.number_input(f"Enter the value for {x} : ")
            else:
                values[x] = st.selectbox(f"Choose the value for {x}:", options=train_data[x].unique().tolist())
        test_data = pd.DataFrame([values])
        st.write(test_data)
        if st.button("🚀 Predict the value"):
            predict_data = main_pip.predict(test_data)
            st.success(f"✅ Predicted Value: {predict_data[0]}")

    elif choosed_model == "SVR":
        mm(SVR(), {
            "model__kernel": ['linear', 'rbf', 'poly'],
            'model__degree': [2, 3, 4],
            "model__C": [0.01, 0.1, 1, 10, 100],
            'model__epsilon': [0.01, 0.1, 0.5, 1.0],
            'model__gamma': ['scale', 'auto']
        })

    elif choosed_model == "XGBoost":
        mm(XGBRegressor(), {
            'model__n_estimators': [100, 200, 300],
            'model__max_depth': [3, 4, 5, 6],
            'model__learning_rate': [0.01, 0.05, 0.1, 0.2],
            'model__subsample': [0.6, 0.8, 1.0],
            'model__colsample_bytree': [0.6, 0.8, 1.0],
            'model__gamma': [0, 0.1, 0.3],
            'model__reg_lambda': [0.1, 1, 5],
            'model__reg_alpha': [0, 0.1, 1]
        })

    elif choosed_model == "ElasticNet":
        mm(ElasticNet(), {
            'model__alpha': [0.01, 0.1, 1, 10],
            'model__l1_ratio': [0, 0.25, 0.5, 0.75, 1],
            'model__max_iter': [1000, 5000]
        })

    elif choosed_model == "DecisionTreeRegressor":
        mm(DecisionTreeRegressor(), {
            'model__max_depth': [None, 3, 5, 7],
            'model__min_samples_split': [2, 5, 10],
            'model__min_samples_leaf': [1, 2, 4]
        })

    elif choosed_model == "RandomForestRegressor":
        mm(RandomForestRegressor(), {
            'model__n_estimators': [100, 200, 500],
            'model__max_depth': [None, 3, 5, 7],
            'model__min_samples_split': [2, 5, 10],
            'model__min_samples_leaf': [1, 2, 4]
        })

    elif choosed_model == "GradientBoostingRegressor":
        mm(GradientBoostingRegressor(), {
            'model__n_estimators': [100, 200, 500],
            'model__learning_rate': [0.01, 0.05, 0.1],
            'model__max_depth': [3, 4, 5]
        })

    elif choosed_model == "KNeighborsRegressor":
        mm(KNeighborsRegressor(), {
            'model__n_neighbors': [3, 5, 7],
            'model__weights': ['uniform', 'distance']
        })

    elif choosed_model == "DNN":
        model = Sequential([
            Dense(256, activation="tanh"),
            Dense(128, activation="tanh"),
            Dense(64, activation="tanh"),
            Dense(32, activation="tanh"),
            Dense(16, activation="tanh"),
            Dense(8, activation="tanh"),
            Dense(1, activation="linear"),
        ])
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        dd = preproccessing.fit_transform(data.drop(target, axis="columns"))
        model.fit(dd, data[target])

        train_data = data.drop(target, axis="columns")
        train_columns = train_data.columns
        values = {}
        for x in train_columns:
            if x in numerical_data:
                values[x] = st.number_input(f"Enter the value for {x} : ", key=f"num_{x}")
            else:
                values[x] = st.selectbox(f"Choose the value for {x}:", options=train_data[x].unique().tolist(), key=f"cat_{x}")

        test_data = pd.DataFrame([values])
        st.write(test_data)

        if st.button("🚀 Predict with DNN"):
            with st.spinner("🔍 Predicting..."):
                test_data = preproccessing.transform(test_data)
                predict_data = model.predict(test_data)
                st.success(f"✅ Predicted Value: {predict_data[0][0]}")

    st.markdown("## 📊 Data Visualization")
    numerical_columns_gaph = data.select_dtypes(include="number").columns
    cat_columns_gaph = data.select_dtypes(include="object").columns
    data_graph = data
    for x in numerical_columns_gaph:
        data_graph[x] = data_graph[x].fillna(data_graph[x].mean())
    for x in cat_columns_gaph:
        data_graph[x] = data_graph[x].fillna(data_graph[x].mode().iloc[0])

    tab1, tab2, tab3 = st.tabs(["📌 Numerical Plots", "📈 Distribution", "📊 Categorical Analysis"])
    with tab1:
        x = st.selectbox("Choose X-axis:", options=(data_graph.select_dtypes(include=["number"])).columns, key="tab_1_selecting")
        y = st.selectbox("Choose Y-axis:", options=(data_graph.select_dtypes(include=["number"])).columns)
        col1, col2 = st.columns(2)
        with col1:
            fig_1 = px.scatter(data_graph, x=x, y=y, title="Scatter Plot", template="plotly_dark")
            st.plotly_chart(fig_1, use_container_width=True)
        with col2:
            fig_2 = px.line(data_graph, x=x, y=y, title="Line Plot", template="plotly_dark")
            st.plotly_chart(fig_2, use_container_width=True)

    with tab2:
        x = st.selectbox("Choose the column:", options=(data_graph.select_dtypes(include=["number"])).columns, key="tab_2_selecting")
        fig_3 = px.box(data_graph, y=x, title="Box Plot", template="plotly_dark")
        st.plotly_chart(fig_3, use_container_width=True)

    with tab3:
        x = st.selectbox("Choose categorical feature:", options=(data_graph.select_dtypes(include=["object"])).columns, key="tab_3_selectingx")
        y = st.selectbox("Choose numerical feature:", options=(data_graph.select_dtypes(include=["number"])).columns, key="tab_3_selectingy")
        fig_4 = px.histogram(data_graph, x=x, y=y, title="Categorical vs Numerical", template="plotly_dark")
        st.plotly_chart(fig_4, use_container_width=True)



