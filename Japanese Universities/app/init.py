from flask import Flask, render_template, request, session, redirect, url_for
import pandas as pd
import joblib

app = Flask(__name__)
app.secret_key = 'your_secret_key'

# Load the saved ensemble model
model = joblib.load('ensemble_model.joblib')

# Label encoding mapping
type_mapping = {'private': 2, 'public': 1, 'national': 0}
has_mapping = {True: 1, False: 0}
difficulty_mapping = {0: 'A', 1: 'D', 2: 'F', 3: 'C', 4: 'B', 5: 'E', 6: 'S'}

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Get form data
        data = {
            'type': type_mapping[request.form['type']],
            'latitude': float(request.form['latitude']),
            'longitude': float(request.form['longitude']),
            'faculty_count': int(request.form['faculty_count']),
            'department_count': int(request.form['department_count']),
            'has_grad': has_mapping[request.form['has_grad'] == 'True'],
            'has_remote': has_mapping[request.form['has_remote'] == 'True'],
            'review_rating': float(request.form['review_rating']),
            'review_count': int(request.form['review_count']),
            'difficulty_SD': float(request.form['difficulty_SD']),
            'founding_year': int(request.form['founding_year']),
            'founding_month': int(request.form['founding_month']),
        }
        
        input_data = pd.DataFrame([data])
        
        # Make prediction
        prediction = model.predict(input_data)[0]
        predicted_label = difficulty_mapping[prediction]
        
        # Store the prediction in the session
        session['prediction'] = predicted_label
        
        # Redirect to the same page to avoid form resubmission
        return redirect(url_for('index'))
    
    prediction = session.pop('prediction', None)
    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
