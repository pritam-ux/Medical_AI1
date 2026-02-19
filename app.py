
import os
import pickle
import numpy as np
try:
    import shap
except ImportError:
    shap = None
import matplotlib.pyplot as plt
from io import BytesIO
import base64
from flask import Flask, render_template, request, redirect, url_for, flash
from flask_login import LoginManager, login_user, login_required, logout_user, current_user
from models import db, User

# Initialize App
app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-change-this-in-production'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Initialize Extensions
db.init_app(app)
login_manager = LoginManager()
login_manager.login_view = 'login'
login_manager.init_app(app)

# Load Model
model_path = os.path.join(os.path.dirname(__file__), "model.pkl")
if os.path.exists(model_path):
    model = pickle.load(open(model_path, "rb"))
    # Initialize SHAP explainer (try-except block for safety)
    try:
        if shap:
            explainer = shap.Explainer(model)
        else:
            explainer = None
    except Exception as e:
        print(f"Error initializing SHAP explainer: {e}")
        explainer = None
else:
    model = None
    print("Warning: model.pkl not found!")

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# Create DB
with app.app_context():
    db.create_all()

# --- Routes ---

@app.route('/')
@login_required
def home():
    return render_template('home.html')

@app.route('/about')
@login_required
def about():
    return render_template('about.html')

@app.route('/services')
@login_required
def services():
    return render_template('services.html')

@app.route('/contact', methods=['GET', 'POST'])
@login_required
def contact():
    if request.method == 'POST':
        flash('Message sent! We will get back to you shortly.', 'success')
        return redirect(url_for('contact'))
    return render_template('contact.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        user = User.query.filter_by(username=username).first()
        if user and user.check_password(password):
            login_user(user)
            return redirect(url_for('dashboard'))
        else:
            flash('Invalid username or password.', 'error')
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        if User.query.filter_by(username=username).first():
            flash('Username already exists.', 'error')
        else:
            new_user = User(username=username)
            new_user.set_password(password)
            db.session.add(new_user)
            db.session.commit()
            flash('Account created! Please log in.', 'success')
            return redirect(url_for('login'))
    return render_template('register.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('home'))

@app.route('/dashboard')
@login_required
def dashboard():
    return render_template('dashboard.html', user=current_user)

@app.route('/predict', methods=['POST'])
@login_required
def predict():
    if not model:
        flash("Model not loaded.", "error")
        return redirect(url_for('dashboard'))

    try:
        # Extract features
        features = [
            float(request.form.get('pregnancies')),
            float(request.form.get('glucose')),
            float(request.form.get('bp')),
            float(request.form.get('skin')),
            float(request.form.get('insulin')),
            float(request.form.get('bmi')),
            float(request.form.get('dpf')),
            float(request.form.get('age'))
        ]
        
        input_data = np.array([features])
        
        # Prediction
        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0][1]
        
        # Risk Logic
        risk_level = "LOW RISK"
        risk_color = "success"
        if probability >= 0.6:
            risk_level = "HIGH RISK"
            risk_color = "danger"
        elif probability >= 0.3:
            risk_level = "MODERATE RISK"
            risk_color = "warning"

        # SHAP Plot
        plot_url = None
        if explainer:
            shap_values = explainer(input_data)
            # Feature names for the plot
            shap_values.feature_names = ["Pregnancies", "Glucose", "Blood Pressure", "Skin Thickness", "Insulin", "BMI", "Diabetes Pedigree", "Age"]
            
            plt.figure()
            # We visualize the first (and only) instance
            # waterwall plot returns a matplotlib figure usually, but here we might need to be careful with handling
            # simpler approach: creating a summary plot or bar plot for just this instance if possible, 
            # but waterfall is best for single instance.
            # However, shap.plots.waterfall is tricky to save to BytesIO directly as it draws on current figure.
            
            try:
                fig, ax = plt.subplots()
                # waterfall draws on the current figure usually
                shap.plots.waterfall(shap_values[0][:, 1], show=False) 
                
                buf = BytesIO()
                plt.savefig(buf, format='png', bbox_inches='tight')
                plt.close(fig)
                buf.seek(0)
                plot_url = base64.b64encode(buf.getvalue()).decode('utf8')
            except Exception as e:
                print(f"SHAP plotting error: {e}")

        return render_template('result.html', 
                               risk_level=risk_level, 
                               risk_color=risk_color, 
                               probability=round(probability*100, 2),
                               plot_url=plot_url)

    except Exception as e:
        flash(f"Error during prediction: {e}", "error")
        return redirect(url_for('dashboard'))

if __name__ == '__main__':
    app.run(debug=True)
