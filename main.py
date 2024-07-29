from flask import Flask, request, render_template, jsonify
import ast
import numpy as np
import pandas as pd
import pickle
from fuzzywuzzy import process
import ast
# Load datasets
symptom = pd.read_csv('datasets/symtoms_df.csv')
precaution = pd.read_csv('datasets/precautions_df.csv')
workout = pd.read_csv('datasets/workout_df.csv')
description = pd.read_csv('datasets/description.csv')
medication = pd.read_csv('datasets/medications.csv')
diet = pd.read_csv('datasets/diets.csv')

# Load model
svc = pickle.load(open("models/svc.pkl", 'rb'))

app = Flask(__name__)

# Helper function
def helper(dis):
    desc = description[description['Disease'] == dis]['Description']
    desc = " ".join([w for w in desc])

    prec = precaution[precaution["Disease"] == dis][["Precaution_1", "Precaution_2", "Precaution_3", "Precaution_4"]]
    prec = [col for col in prec.values] if not prec.empty else []

    med = medication[medication['Disease'] == dis]['Medication']
    med = [med for med in med.values] if not med.empty else []

    die = diet[diet['Disease'] == dis]['Diet']
    die = [die for die in die.values] if not die.empty else []

    wrkout = workout[workout['disease'] == dis]['workout']
    wrkout = [wo for wo in wrkout.values] if not wrkout.empty else []

    return desc, prec, med, die, wrkout

disease_list = {
    15: 'Fungal infection', 4: 'Allergy', 16: 'GERD', 9: 'Chronic cholestasis', 14: 'Drug Reaction',
    33: 'Peptic ulcer disease', 1: 'AIDS', 12: 'Diabetes ', 17: 'Gastroenteritis', 6: 'Bronchial Asthma',
    23: 'Hypertension ', 30: 'Migraine', 7: 'Cervical spondylosis', 32: 'Paralysis (brain hemorrhage)',
    28: 'Jaundice', 29: 'Malaria', 8: 'Chicken pox', 11: 'Dengue', 37: 'Typhoid', 40: 'hepatitis A',
    19: 'Hepatitis B', 20: 'Hepatitis C', 21: 'Hepatitis D', 22: 'Hepatitis E', 3: 'Alcoholic hepatitis',
    36: 'Tuberculosis', 10: 'Common Cold', 34: 'Pneumonia', 13: 'Dimorphic hemorrhoids(piles)',
    18: 'Heart attack', 39: 'Varicose veins', 26: 'Hypothyroidism', 24: 'Hyperthyroidism',
    25: 'Hypoglycemia', 31: 'Osteoarthritis', 5: 'Arthritis', 0: '(vertigo) Paroxysmal Positional Vertigo',
    2: 'Acne', 38: 'Urinary tract infection', 35: 'Psoriasis', 27: 'Impetigo'
}

symptoms_dict = {
    'itching': 0, 'skin_rash': 1, 'nodal_skin_eruptions': 2, 'continuous_sneezing': 3, 'shivering': 4,
    'chills': 5, 'joint_pain': 6, 'stomach_pain': 7, 'acidity': 8, 'ulcers_on_tongue': 9,
    'muscle_wasting': 10, 'vomiting': 11, 'burning_micturition': 12, 'spotting_urination': 13,
    'fatigue': 14, 'weight_gain': 15, 'anxiety': 16, 'cold_hands_and_feets': 17, 'mood_swings': 18,
    'weight_loss': 19, 'restlessness': 20, 'lethargy': 21, 'patches_in_throat': 22,
    'irregular_sugar_level': 23, 'cough': 24, 'high_fever': 25, 'sunken_eyes': 26, 'breathlessness': 27,
    'sweating': 28, 'dehydration': 29, 'indigestion': 30, 'headache': 31, 'yellowish_skin': 32,
    'dark_urine': 33, 'nausea': 34, 'loss_of_appetite': 35, 'pain_behind_the_eyes': 36, 'back_pain': 37,
    'constipation': 38, 'abdominal_pain': 39, 'diarrhoea': 40, 'mild_fever': 41, 'yellow_urine': 42,
    'yellowing_of_eyes': 43, 'acute_liver_failure': 44, 'fluid_overload': 45, 'swelling_of_stomach': 46,
    'swelled_lymph_nodes': 47, 'malaise': 48, 'blurred_and_distorted_vision': 49, 'phlegm': 50,
    'throat_irritation': 51, 'redness_of_eyes': 52, 'sinus_pressure': 53, 'runny_nose': 54,
    'congestion': 55, 'chest_pain': 56, 'weakness_in_limbs': 57, 'fast_heart_rate': 58,
    'pain_during_bowel_movements': 59, 'pain_in_anal_region': 60, 'bloody_stool': 61,
    'irritation_in_anus': 62, 'neck_pain': 63, 'dizziness': 64, 'cramps': 65, 'bruising': 66,
    'obesity': 67, 'swollen_legs': 68, 'swollen_blood_vessels': 69, 'puffy_face_and_eyes': 70,
    'enlarged_thyroid': 71, 'brittle_nails': 72, 'swollen_extremities': 73, 'excessive_hunger': 74,
    'extra_marital_contacts': 75, 'drying_and_tingling_lips': 76, 'slurred_speech': 77, 'knee_pain': 78,
    'hip_joint_pain': 79, 'muscle_weakness': 80, 'stiff_neck': 81, 'swelling_joints': 82,
    'movement_stiffness': 83, 'spinning_movements': 84, 'loss_of_balance': 85, 'unsteadiness': 86,
    'weakness_of_one_body_side': 87, 'loss_of_smell': 88, 'bladder_discomfort': 89,
    'foul_smell_of_urine': 90, 'continuous_feel_of_urine': 91, 'passage_of_gases': 92,
    'internal_itching': 93, 'toxic_look_(typhos)': 94, 'depression': 95, 'irritability': 96,
    'muscle_pain': 97, 'altered_sensorium': 98, 'red_spots_over_body': 99, 'belly_pain': 100,
    'abnormal_menstruation': 101, 'dischromic_patches': 102, 'watering_from_eyes': 103,
    'increased_appetite': 104, 'polyuria': 105, 'family_history': 106, 'mucoid_sputum': 107,
    'rusty_sputum': 108, 'lack_of_concentration': 109, 'visual_disturbances': 110,
    'receiving_blood_transfusion': 111, 'receiving_unsterile_injections': 112, 'coma': 113,
    'stomach_bleeding': 114, 'distention_of_abdomen': 115, 'history_of_alcohol_consumption': 116,
    'fluid_overload.1': 117, 'blood_in_sputum': 118, 'prominent_veins_on_calf': 119, 'palpitations': 120,
    'painful_walking': 121, 'pus_filled_pimples': 122, 'blackheads': 123, 'scurring': 124,
    'skin_peeling': 125, 'silver_like_dusting': 126, 'small_dents_in_nails': 127,
    'inflammatory_nails': 128, 'blister': 129, 'red_sore_around_nose': 130, 'yellow_crust_ooze': 131
}

invalid_words = {'in', 'on', 'the', 'for', 'hi', 'hello', 'were', 'where', 'of', 'half', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z'}

def get_closest_symptom(user_symptom):
    closest_match = process.extractOne(user_symptom, symptoms_dict.keys(), score_cutoff=70)
    return closest_match[0] if closest_match else None

def is_common_cold_related(symptoms):
    common_cold_related_symptoms = {'cold', 'shivering', 'fever', 'cough', 'sneezing', 'runny_nose', 'throat_irritation'}
    return any(symptom in common_cold_related_symptoms for symptom in symptoms)

def is_gerd_related(symptoms):
    gerd_related_symptoms = {'vomiting', 'dehydration', 'acidity', 'stomach_pain', 'indigestion', 'abdominal_pain', 'chest_pain'}
    return any(symptom in gerd_related_symptoms for symptom in symptoms)

def is_allergy(symptoms):
    allergy_symptoms = {'itching', 'skin_rash', 'eczema', 'redness_of_skin', 'swelling'}
    return any(symptom in allergy_symptoms for symptom in symptoms)

def is_diabetes(symptoms):
    diabetes_symptoms = {'increased_thirst', 'frequent_urination', 'extreme_hunger',
                         'slow_healing_sores', 'frequent_infections', 'blurred_vision'}
    return any(symptom in diabetes_symptoms for symptom in symptoms)

def is_jaundice(symptoms):
    jaundice_symptoms = {'yellowish_skin', 'yellowing_of_eyes', 'dark_urine', 'pale_stools'}
    return all(symptom in symptoms for symptom in jaundice_symptoms)


def is_nausea_related(symptoms):
    gerd_related_symptoms = {'vomiting', 'dehydration', 'acidity', 'stomach_pain', 'indigestion', 'abdominal_pain',
                             'chest_pain', 'nausea'}
    gastroenteritis_symptoms = {'vomiting', 'dehydration', 'stomach_pain', 'diarrhoea', 'fever', 'nausea'}

    if 'nausea' in symptoms:
        if any(symptom in gerd_related_symptoms for symptom in symptoms):
            return disease_list[16]  # GERD
        if any(symptom in gastroenteritis_symptoms for symptom in symptoms):
            return disease_list[17]  # Gastroenteritis
    return None

def get_predicted_value(patient_symptoms):
    # Define specific symptoms related to migraine
    migraine_symptoms = {'pulsating_headache', 'sensitivity_to_light', 'lightheadedness'}

    # Check if all migraine symptoms are present
    if all(symptom in patient_symptoms for symptom in migraine_symptoms):
        return disease_list[30]  # Migraine

    # Check if at least 3 typhoid symptoms are present
    typhoid_symptoms = {'loss_of_appetite', 'fever', 'constipation', 'vomiting',
                        'enlarged_liver', 'enlarged_spleen', 'stomach_pain'}
    matched_typhoid_symptoms = [symptom for symptom in patient_symptoms if symptom in typhoid_symptoms]
    if len(matched_typhoid_symptoms) >= 3:
        return disease_list[37]  # Typhoid

    # Check for diabetes symptoms
    diabetes_symptoms = {'increased_thirst', 'frequent_urination', 'extreme_hunger',
                         'slow_healing_sores', 'frequent_infections', 'blurred_vision'}
    if any(symptom in diabetes_symptoms for symptom in patient_symptoms):
        return disease_list[12]  # Diabetes

    # Check for specific conditions based on chest pain
    gastric_symptoms = {'vomiting', 'dehydration', 'acidity', 'stomach_pain', 'indigestion', 'abdominal_pain', 'chest_pain'}
    heart_related_symptoms = {'chest_pain', 'breathlessness', 'dizziness', 'nausea'}

    if 'chest_pain' in patient_symptoms and len(patient_symptoms) == 1:
        return disease_list[16]

    elif 'chest_pain' in patient_symptoms:
        if all(symptom in patient_symptoms for symptom in gastric_symptoms):
            return disease_list[16]  # GERD

        if any(symptom in heart_related_symptoms for symptom in patient_symptoms):
            return disease_list[18]  # Heart Attack

    nausea_related_disease = is_nausea_related(patient_symptoms)
    if nausea_related_disease:
        return nausea_related_disease

    # Check for common cold related symptoms
    if is_common_cold_related(patient_symptoms):
        return disease_list[10]  # Common Cold
    if is_gerd_related(patient_symptoms):
        return disease_list[16]
    if is_allergy(patient_symptoms):
        return disease_list[4]
    if is_jaundice(patient_symptoms):
        return disease_list[28]  # Jaundice


    # Check for heart attack symptoms (chest pain)


    # Default prediction using the machine learning model
    input_vector = np.zeros(len(symptoms_dict))
    valid_symptoms = False

    for item in patient_symptoms:
        matched_symptom = get_closest_symptom(item)
        if matched_symptom:
            input_vector[symptoms_dict[matched_symptom]] = 1
            valid_symptoms = True

    if not valid_symptoms:
        return None

    return disease_list[svc.predict([input_vector])[0]]





@app.route('/index')
def index():
    return render_template("index.html", symptoms=list(symptoms_dict.keys()))

@app.route('/predict', methods=['POST', 'GET'])
def predict():
    if request.method == "POST":
        symptoms = request.form.get('symptoms')

        if not symptoms:
            return render_template('index.html', error_message="Please enter symptoms!")

        user_symp = [s.strip() for s in symptoms.split(',')]
        user_symp = [sym.strip("[]' ") for sym in user_symp]
        user_symp = [sym.lower().replace(' ', '_') for sym in user_symp]

        user_symp = [sym for sym in user_symp if sym not in invalid_words]

        if len(user_symp) == 1 and (user_symp[0] == 'sweating' or user_symp[0] == 'sweat'):
            return render_template('index.html', error_message="Please provide more symptoms!")

        if not user_symp:
            return render_template('index.html', error_message="Please provide valid symptoms!")

        predicted_disease = get_predicted_value(user_symp)

        if not predicted_disease:
            return render_template('index.html', error_message="Please provide correct symptoms!")

        desc, prec, med, die, wrkout = helper(predicted_disease)

<<<<<<< HEAD
        # Handle empty lists and non-list content
        prec = prec if prec else ["No precautions available."]
        med = med if med else ["No medications available."]
        die = die if die else ["No diets available."]
        wrkout = wrkout if wrkout else ["No workouts available."]
        # Ensure valid list format for precautions, medications, diets, and workouts
        def parse_list(item):
            if isinstance(item, str):
                try:
                    return ast.literal_eval(item)
                except (ValueError, SyntaxError):
                    return [item]
            return item
=======
 
        if type(med) == list:
            med = ast.literal_eval(med[0] ) 

        if type(die) == list:
            die = ast.literal_eval( die[0] )        

        if not prec:
            return render_template('index.html', error_message="No precautions found for the predicted disease.")
        if not med:
            return render_template('index.html', error_message="No medications found for the predicted disease.")
        if not die:
            return render_template('index.html', error_message="No diets found for the predicted disease.")
        if not wrkout:
            return render_template('index.html', error_message="No workouts found for the predicted disease.")
>>>>>>> 0304195eb6a3ca19a7fb81d2edff2baa8ae0a57a

        prec = [parse_list(p) for p in prec]
        med = [parse_list(m) for m in med]
        die = [parse_list(d) for d in die]
        wrkout = [parse_list(w) for w in wrkout]

        my_prec = [i for sublist in prec for i in sublist]
        my_wo = [i for sublist in wrkout for i in sublist]
        my_med = [m for sublist in med for m in sublist]
        my_diet = [i for sublist in die for i in sublist]

        return render_template('index.html', predicted_disease=predicted_disease, description=desc, precausion=my_prec,
                               medication=my_med, diets=my_diet, workout=my_wo)

    return render_template('index.html')

@app.route('/contact')
def contact():
    return render_template("Contact.html")

@app.route('/about')
def about():
    return render_template("About.html")

if __name__ == "__main__":
    app.run(debug=True)
