from flask import Flask, render_template, request
import joblib, pandas as pd, numpy as np

app = Flask(__name__)
# Loading model and preprocessor
model = joblib.load('xgbr2_model.pkl')
preprocessor = joblib.load('preprocessor.pkl')

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    if request.method == 'POST':
        # Collecting input
        raw_data = {
            'Item_Identifier': request.form['Item_Identifier'][:2],
            'Item_Weight': float(request.form['Item_Weight']),
            'Item_Fat_Content': request.form['Item_Fat_Content'].strip().lower(),
            'Item_Visibility': float(request.form['Item_Visibility']),
            'Item_Type': request.form['Item_Type'],
            'Item_MRP': float(request.form['Item_MRP']),
            'Outlet_Identifier': request.form['Outlet_Identifier'],
            'Outlet_Establishment_Year': int(request.form['Outlet_Establishment_Year']),
            'Outlet_Size': request.form['Outlet_Size'],
            'Outlet_Location_Type': request.form['Outlet_Location_Type'],
            'Outlet_Type': request.form['Outlet_Type'],
            'Item_Outlet_Sales': 0 
            }
        # Item_Fat_Content data made consistent
        if raw_data['Item_Fat_Content'] in ['low fat', 'lowfat', 'Low Fat']:
            raw_data['Item_Fat_Content'] = 'LF'
        elif raw_data['Item_Fat_Content'] in ['reg', 'Regular']:
            raw_data['Item_Fat_Content'] = 'RF'
        else:
            raw_data['Item_Fat_Content'] = raw_data['Item_Fat_Content'].upper()
        
        #converting to df
        full_df = pd.DataFrame([raw_data])
        #preprocessnig - split - predict
        transformed = preprocessor.transform(full_df)
        transformed_input = transformed[:, :-1] 
        prediction = model.predict(transformed_input)[0]

        return render_template('form.html', prediction=round(prediction, 2), data=raw_data)

    return render_template('form.html', prediction='None', data=None)

if __name__ == '__main__':
    app.run()