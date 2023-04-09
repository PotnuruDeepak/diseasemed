from flask import Flask, render_template, request, redirect, url_for
import joblib
import pandas as pd

app = Flask(__name__)
final_data = []
l1 = []
l2 = []

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/prediction', methods=['GET', 'POST'])
def prediction():
    indexes = request.form["content"].split(" ")
    indexes = list(map(int, indexes))
    symptons  = ['itching',
                'skin_rash',
                'nodal_skin_eruptions',
                'continuous_sneezing',
                'shivering',
                'chills',
                'joint_pain',
                'stomach_pain',
                'acidity',
                'ulcers_on_tongue',
                'muscle_wasting',
                'vomiting',
                'burning_micturition',
                'spotting_ urination',
                'fatigue',
                'weight_gain',
                'anxiety',
                'cold_hands_and_feets',
                'mood_swings',
                'weight_loss',
                'restlessness',
                'lethargy',
                'patches_in_throat',
                'irregular_sugar_level',
                'cough',
                'high_fever',
                'sunken_eyes',
                'breathlessness',
                'sweating',
                'dehydration',
                'indigestion',
                'headache',
                'yellowish_skin',
                'dark_urine',
                'nausea',
                'loss_of_appetite',
                'pain_behind_the_eyes',
                'back_pain',
                'constipation',
                'abdominal_pain',
                'diarrhoea',
                'mild_fever',
                'yellow_urine',
                'yellowing_of_eyes',
                'acute_liver_failure',
                'fluid_overload',
                'swelling_of_stomach',
                'swelled_lymph_nodes',
                'malaise',
                'blurred_and_distorted_vision',
                'phlegm',
                'throat_irritation',
                'redness_of_eyes',
                'sinus_pressure',
                'runny_nose',
                'congestion',
                'chest_pain',
                'weakness_in_limbs',
                'fast_heart_rate',
                'pain_during_bowel_movements',
                'pain_in_anal_region',
                'bloody_stool',
                'irritation_in_anus',
                'neck_pain',
                'dizziness',
                'cramps',
                'bruising',
                'obesity',
                'swollen_legs',
                'swollen_blood_vessels',
                'puffy_face_and_eyes',
                'enlarged_thyroid',
                'brittle_nails',
                'swollen_extremeties',
                'excessive_hunger',
                'extra_marital_contacts',
                'drying_and_tingling_lips',
                'slurred_speech',
                'knee_pain',
                'hip_joint_pain',
                'muscle_weakness',
                'stiff_neck',
                'swelling_joints',
                'movement_stiffness',
                'spinning_movements',
                'loss_of_balance',
                'unsteadiness',
                'weakness_of_one_body_side',
                'loss_of_smell',
                'bladder_discomfort',
                'foul_smell_of urine',
                'continuous_feel_of_urine',
                'passage_of_gases',
                'internal_itching',
                'toxic_look_(typhos)',
                'depression',
                'irritability',
                'muscle_pain',
                'altered_sensorium',
                'red_spots_over_body',
                'belly_pain',
                'abnormal_menstruation',
                'dischromic _patches',
                'watering_from_eyes',
                'increased_appetite',
                'polyuria',
                'family_history',
                'mucoid_sputum',
                'rusty_sputum',
                'lack_of_concentration',
                'visual_disturbances',
                'receiving_blood_transfusion',
                'receiving_unsterile_injections',
                'coma',
                'stomach_bleeding',
                'distention_of_abdomen',
                'history_of_alcohol_consumption',
                'fluid_overload.1',
                'blood_in_sputum',
                'prominent_veins_on_calf',
                'palpitations',
                'painful_walking',
                'pus_filled_pimples',
                'blackheads',
                'scurring',
                'skin_peeling',
                'silver_like_dusting',
                'small_dents_in_nails',
                'inflammatory_nails',
                'blister',
                'red_sore_around_nose',
                'yellow_crust_ooze'] 
    diseases = {0 : 'Acne', 
            1 : 'Allergies', 
            2 : 'Arthritis', 
            3 : 'Diabetes', 
            4 : 'Fungal', 
            5 : 'Hypertension', 
            6 : 'Hyperthyroidism', 
            7 : 'Hypothyroidism', 
            8 : 'Migraine', 
            9 : 'Vertigo'}
                   
    count = 1
    p = [[]]
    while count != 133:
        if count in indexes:
            p[0].append(1)
        else:
            p[0].append(0)
        count+=1

    pickled_model = joblib.load()
    i = pickled_model.predict(p)
    l = i.tolist()
    df_dict = {}
    for i in range(10) :
        df_dict[l[0][i]] = diseases[i] 
    
    for i in sorted(df_dict,reverse=True):
        l1.append(i)
        l2.append(df_dict[i])
    data = {'Disease': l2, 'Probability': l1}
    print(data)       
    
    final_data.append([
        {'REASON': l2[0],
         'PROBABILITY':l1[0]},
        {'REASON': l2[1],
         'PROBABILITY':l1[1]},
        {'REASON': l2[2],
         'PROBABILITY':l1[3]}
    ])
    return render_template('prediction.html', data=final_data[0])

@app.route('/recommendation', methods=['GET', 'POST'])
def recommendation():
    mapreduce_output = pd.read_csv()
    mapreduce_output_original = mapreduce_output.drop_duplicates()
    reasons = list(mapreduce_output_original['Reason'].unique())
    
    l = []
    z = l2[:3]
    print(z)
    
    for i in reasons:
        r = list(mapreduce_output_original[mapreduce_output_original['Reason'] == i][:5]['Reason'])
        dr = list(mapreduce_output_original[mapreduce_output_original['Reason'] == i][:5]['Drug_Name'])
        manu = list(mapreduce_output_original[mapreduce_output_original['Reason'] == i][:5]['Manufacturer'])
        rat = list(mapreduce_output_original[mapreduce_output_original['Reason'] == i][:5]['Rating\t'])  
        
        for j in range(len(r)):
            if r[j] != 'Reason':
                l.append({"REASON":r[j], 
                "DRUG_NAME":dr[j], 
                "MANUFACTURER":manu[j], 
                "RATING":rat[j]})
    final = []
    for a in range(len(l)):
        for b in l[a].keys():
            if l[a][b] == z[0]:
                final.append(l[a])
                break
            if l[a][b] == z[1]:
                final.append(l[a])
                break
            if l[a][b] == z[2]:
                final.append(l[a])
                break
    print(final)    
        

    return render_template('recommendation.html', data=final)

#if __name__ == '__main__':
#	 app.run(debug=True)

