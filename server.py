import streamlit as st 
import pickle

print('Successfully executed ')

model = pickle.load(open('model1.pkl', 'rb'))

def predict_pro(list1):
    
    print("inside predict function")
    prediction=model.predict(list1)
    
    return prediction

# def processing(l):
#     # Your dictionary
#     category_to_binary = {'Male': 1, 
#                            'Fiber Optic': (0, 1), 'No': (0, 0)}

# # Extract the tuple for 'DSL'
# dsl_tuple = category_to_binary['DSL']

# print(dsl_tuple)
# print("Inside Processing")
# print(l)

def main():

    pred_arr = []
    gender_Male_dict = {"Male":1,"Female":0}
    yes_no_dict = {"Yes":1,"No":0}
    lines_dict = {"Yes":(1,0), "No":(0,1), "NA":(0,0)}
    internet_service_dict = {"DSL":(1,0), "Fibre Optic":(0,1), "No":(0,0)}
    online_security_dict = {"Yes":(0,1), "No":(1,0), "NA":(1,0)} 
    online_backup_dict = {"Yes":(0,1), "No":(1,0), "NA":(1,0)}
    device_protection_dict = {"Yes":(1,0), "No":(1,0), "NA":(1,0)}
    tech_support_dict = {"Yes":(1,0), "No":(1,0), "NA":(1,0)}
    stream_tv_dict = {"Yes":(0,1), "No":(1,0), "NA":(1,0)}
    stream_movies_dict = {"Yes":(0,1), "No":(1,0), "NA":(1,0)}
    contract_dict = {"Monthly":(0,0), "One":(1,0), "Two":(0,1)}
    mail_dict = {"Bank":(0,0,0), "Credit":(1,0,0), "Electronic":(0,1,0), "Mailed": (0,0,1)}
    

    st.title("Telco Churn Prediction")
    html_temp = """
    <div style="background-color:#b3db86;padding:8px">
    <h2 style="color:black;text-align:center;">Predicting Customer Churn</h2>
    </div>
    """
    st.markdown(html_temp,unsafe_allow_html=True)

    tenure = st.slider("Enter tenure (months)", min_value=0, max_value=500)
    pred_arr.extend([tenure])
    # print(pred_arr)

    MonthlyCharges = st.slider("Monthly Charges", min_value=0, max_value=300)
    pred_arr.extend([MonthlyCharges])
    # print(pred_arr)

    TotalCharges = st.slider("Total Charges", min_value=0, max_value=15000)
    pred_arr.extend([TotalCharges])
    # print(pred_arr)

    gender_Male = st.selectbox("Gender",("Male", "Female"))
    # print("Gender male streamlit value")
    # print(gender_Male)
    # print("subset dict value")
    # print(gender_Male_dict[gender_Male])
    pred_arr.extend([gender_Male_dict[gender_Male]])
    # print(pred_arr)

    SeniorCitizen_Yes = st.selectbox("Are you a senior citizen?", ("Yes", "No"))
    pred_arr.extend([yes_no_dict[SeniorCitizen_Yes]])
    # print(pred_arr)

    partner = st.selectbox("Do you have a partner?", ("Yes", "No"))
    pred_arr.extend([yes_no_dict[partner]])
    # print(pred_arr)

    dependents = st.selectbox("Is anyone dependent on you?", ("Yes", "No"))
    pred_arr.extend([yes_no_dict[dependents]])
    # print(pred_arr)

    phone_service = st.selectbox("Do you have a phone service?", ("Yes", "No"))
    pred_arr.extend([yes_no_dict[phone_service]])
    # print(pred_arr)

    multi_lines = st.selectbox("Do you have multiple lines", ("Yes", "No", "NA"))
    pred_arr.extend(list(lines_dict[multi_lines]))
    # print(pred_arr)

    internet_service = st.selectbox("Select your internet service type",("DSL", "Fibre Optic", "None"))
    pred_arr.extend(list(internet_service_dict[internet_service]))
    # print(pred_arr)

    online_security = st.selectbox("Have you opted for Online Security?", ("Yes", "No"))
    pred_arr.extend(list(online_security_dict[online_security]))
    # print(pred_arr)

    online_backup = st.selectbox("Have you opted for Online Backup?", ("Yes", "No"))
    pred_arr.extend(list(online_backup_dict[online_backup]))
    # print(pred_arr)

    device_protection = st.selectbox("Have you opted for Device Protection?", ("Yes", "No", "NA"))
    pred_arr.extend(list(device_protection_dict[device_protection]))
    # print(pred_arr)

    tech_support = st.selectbox("Have you opted for Tech Support?", ("Yes", "No","NA"))
    pred_arr.extend(list(tech_support_dict[tech_support]))
    # print(pred_arr)
    
    tv = st.selectbox("Do you stream TV?", ("Yes", "No","NA"))
    pred_arr.extend(list(stream_tv_dict[tv]))
    # print(pred_arr)

    movies = st.selectbox("Do you stream movies?", ("Yes", "No","NA"))
    pred_arr.extend(list(stream_movies_dict[movies]))
    # print(pred_arr)

    contract_month = st.selectbox("Select contract?", ("Monthly", "One","Two"))
    pred_arr.extend(list(contract_dict[contract_month]))

    billing = st.selectbox("Have you opted for paperless billing?", ("Yes", "No"))
    pred_arr.extend([yes_no_dict[billing]])

    mailed_check = st.selectbox("Payment using mailed check?", ("Bank", "Credit", "Electronic", "Mailed"))
    pred_arr.extend(list(mail_dict[mailed_check]))
    # print(pred_arr)
    
    result=""
    if st.button("Predict"):
        
        print(len(pred_arr))
        # pred_arr = [[1,89.35,89.35,0,0,0,1,1,0,0,1,0,0,0,0,0,0,0,0,0,0,1,0,1,0,0,1,0,1,0]]
        result = predict_pro([pred_arr])

        print("Printing result")
        print(result)
        
        st.success('The output is {}'.format(result))
    if st.button("About"):
        st.text("Built with Streamlit, By Mayur Parab and Siddharth Jadhwani")

main()
