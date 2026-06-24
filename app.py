"""
Detection of Fake Profiles and Botnets in Indian Social Networks
Using Graph Neural Networks — Flask Web Application
"""

import os, json, threading, time
from flask import (Flask, render_template, request, redirect,
                   url_for, session, jsonify, flash)
import pandas as pd
import numpy as np

from gnn_model import GNNPipeline

# ──────────────────────────────────────────────────────────────────────────────
app = Flask(__name__)
app.secret_key = 'gnn_fake_detection_2025_secret'
BASE       = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE, 'uploads')
MODEL_DIR  = os.path.join(BASE, 'models')
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(MODEL_DIR,  exist_ok=True)

# ──────────────────────────── In-memory state ─────────────────────────────────
ADMIN_CREDS    = {'admin': 'admin123'}
USERS_DB       = {}
PREDICTIONS    = []
DATASET_INFO   = {}
METRICS        = {}
PIPELINE       = None
TRAIN_PROGRESS = []
TRAINING_DONE  = False

# ── Auto-train on startup ─────────────────────────────────────────────────────
def _auto_train():
    global PIPELINE, METRICS, TRAINING_DONE, DATASET_INFO
    _model_path  = os.path.join(MODEL_DIR,  'pipeline.pkl')
    dataset_path = os.path.join(UPLOAD_DIR, 'Dataset.csv')

    # 1) Try loading saved model first
    if os.path.exists(_model_path):
        try:
            pipe = GNNPipeline.load(_model_path)
            if pipe.is_trained:
                PIPELINE      = pipe
                METRICS       = pipe.metrics
                TRAINING_DONE = True
                if os.path.exists(dataset_path):
                    df = pd.read_csv(dataset_path)
                    _set_dataset_info(df)
                print("[STARTUP] Loaded existing trained model.")
                return
        except Exception as e:
            print(f"[STARTUP] Load failed ({e}), retraining...")

    # 2) Train from the bundled dataset
    if not os.path.exists(dataset_path):
        print("[STARTUP] No dataset found – skipping auto-train.")
        return

    try:
        print("[STARTUP] Training GNN models on Dataset.csv ...")
        df = pd.read_csv(dataset_path)
        _set_dataset_info(df)
        pipe    = GNNPipeline()
        metrics = pipe.fit(df, max_nodes=400)
        pipe.save(_model_path)
        PIPELINE      = pipe
        METRICS       = metrics
        TRAINING_DONE = True
        print("[STARTUP] Training complete.")
        for name, m in metrics.items():
            print(f"  {name}: acc={m['accuracy']}%  f1={m['f1']}%")
    except Exception as e:
        print(f"[STARTUP] Training failed: {e}")


def _set_dataset_info(df):
    global DATASET_INFO
    DATASET_INFO = {
        'total':      len(df),
        'features':   len(df.columns),
        'n_fake':     int((df.get('label', pd.Series()) == 'Fake').sum()),
        'n_genuine':  int((df.get('label', pd.Series()) == 'Genuine').sum()),
        'train_size': int(len(df) * 0.9),
        'test_size':  int(len(df) * 0.1),
        'preview':    df.head(10).to_dict(orient='records'),
        'columns':    list(df.columns),
    }


_startup_thread = threading.Thread(target=_auto_train, daemon=True)
_startup_thread.start()

# ──────────────────────────────────── Routes ──────────────────────────────────

@app.route('/')
def home():
    return render_template('home.html')


# ── Admin ─────────────────────────────────────────────────────────────────────

@app.route('/admin_login', methods=['GET', 'POST'])
def admin_login():
    if request.method == 'POST':
        u = request.form.get('username', '')
        p = request.form.get('password', '')
        if ADMIN_CREDS.get(u) == p:
            session['admin'] = True
            session['admin_user'] = u
            return redirect(url_for('admin_dashboard'))
        flash('Invalid admin credentials', 'error')
    return render_template('admin_login.html')


@app.route('/admin_dashboard')
def admin_dashboard():
    if not session.get('admin'):
        return redirect(url_for('admin_login'))
    return render_template('admin_dashboard.html',
                           dataset_info=DATASET_INFO,
                           metrics=METRICS,
                           training_done=TRAINING_DONE)


@app.route('/load_dataset', methods=['GET', 'POST'])
def load_dataset():
    if not session.get('admin'):
        return redirect(url_for('admin_login'))
    global DATASET_INFO, PIPELINE, METRICS, TRAINING_DONE, TRAIN_PROGRESS

    if request.method == 'POST':
        f = request.files.get('dataset')
        if not f or not f.filename.endswith('.csv'):
            flash('Please upload a valid CSV file', 'error')
            return redirect(url_for('load_dataset'))
        path = os.path.join(UPLOAD_DIR, 'Dataset.csv')
        f.save(path)
        df = pd.read_csv(path)
        _set_dataset_info(df)
        PIPELINE      = None
        METRICS       = {}
        TRAINING_DONE = False
        TRAIN_PROGRESS = []
        flash('Dataset loaded successfully!', 'success')
        return redirect(url_for('admin_dashboard'))

    existing_path = os.path.join(UPLOAD_DIR, 'Dataset.csv')
    preview = None
    if os.path.exists(existing_path):
        df = pd.read_csv(existing_path)
        preview = df.head(5).to_dict(orient='records')
    return render_template('load_dataset.html', preview=preview)


@app.route('/run_gnn', methods=['GET', 'POST'])
def run_gnn():
    if not session.get('admin'):
        return redirect(url_for('admin_login'))
    return render_template('run_gnn.html',
                           metrics=METRICS,
                           training_done=TRAINING_DONE,
                           dataset_info=DATASET_INFO)


@app.route('/api/start_training', methods=['POST'])
def start_training():
    global PIPELINE, METRICS, TRAINING_DONE, TRAIN_PROGRESS
    path = os.path.join(UPLOAD_DIR, 'Dataset.csv')
    if not os.path.exists(path):
        return jsonify({'error': 'Dataset not loaded'}), 400

    TRAIN_PROGRESS = []
    TRAINING_DONE  = False
    METRICS        = {}

    def train():
        global PIPELINE, METRICS, TRAINING_DONE, TRAIN_PROGRESS
        try:
            df = pd.read_csv(path)
            _set_dataset_info(df)
            pipe = GNNPipeline()
            def cb(epoch, loss):
                TRAIN_PROGRESS.append({'epoch': epoch, 'loss': round(float(loss), 4)})
            metrics = pipe.fit(df, progress_cb=cb, max_nodes=400)
            pipe.save(os.path.join(MODEL_DIR, 'pipeline.pkl'))
            PIPELINE      = pipe
            METRICS       = metrics
            TRAINING_DONE = True
            TRAIN_PROGRESS.append({'done': True})
        except Exception as e:
            TRAIN_PROGRESS.append({'error': str(e)})

    t = threading.Thread(target=train, daemon=True)
    t.start()
    return jsonify({'status': 'started'})


@app.route('/api/training_progress')
def training_progress():
    return jsonify({'progress': TRAIN_PROGRESS, 'done': TRAINING_DONE, 'metrics': METRICS})


@app.route('/api/model_status')
def model_status():
    """Lets the browser poll whether the auto-training has finished."""
    return jsonify({
        'ready':       bool(PIPELINE and PIPELINE.is_trained),
        'still_loading': _startup_thread.is_alive(),
    })


@app.route('/view_fake_profiles')
def view_fake_profiles():
    if not session.get('admin'):
        return redirect(url_for('admin_login'))
    return render_template('view_fake_profiles.html', predictions=PREDICTIONS)


# ── User ──────────────────────────────────────────────────────────────────────

@app.route('/user_signup', methods=['GET', 'POST'])
def user_signup():
    if request.method == 'POST':
        u = request.form.get('username', '').strip()
        p = request.form.get('password', '').strip()
        c = request.form.get('contact',  '').strip()
        e = request.form.get('email',    '').strip()
        a = request.form.get('address',  '').strip()
        if not u or not p:
            flash('Username and password are required', 'error')
            return render_template('user_signup.html')
        if u in USERS_DB:
            flash('Username already exists', 'error')
            return render_template('user_signup.html')
        USERS_DB[u] = {'password': p, 'contact': c, 'email': e, 'address': a}
        flash('Signup successful! Please login.', 'success')
        return redirect(url_for('user_login'))
    return render_template('user_signup.html')


@app.route('/user_login', methods=['GET', 'POST'])
def user_login():
    if request.method == 'POST':
        u = request.form.get('username', '')
        p = request.form.get('password', '')
        if USERS_DB.get(u, {}).get('password') == p:
            session['user'] = u
            return redirect(url_for('user_dashboard'))
        flash('Invalid credentials', 'error')
    return render_template('user_login.html')


@app.route('/user_dashboard')
def user_dashboard():
    if not session.get('user'):
        return redirect(url_for('user_login'))
    return render_template('user_dashboard.html', username=session['user'])


@app.route('/fake_profile_detection', methods=['GET', 'POST'])
def fake_profile_detection():
    if not session.get('user'):
        return redirect(url_for('user_login'))

    # Block and wait if startup training is still running (max 90 s)
    if not (PIPELINE and PIPELINE.is_trained) and _startup_thread.is_alive():
        _startup_thread.join(timeout=90)

    result    = None
    form_data = {}

    if request.method == 'POST':
        form_data = {
            'account_age':      request.form.get('account_age',      0),
            'gender':           request.form.get('gender',           'Male'),
            'user_age':         request.form.get('user_age',         25),
            'link_description': request.form.get('link_description', 'No'),
            'status_count':     request.form.get('status_count',     0),
            'friend_count':     request.form.get('friend_count',     0),
            'internet':         request.form.get('internet',         0),
            'got_task':         request.form.get('got_task',         'No'),
            'changed_wifi':     request.form.get('changed_wifi',     'No'),
        }

        if PIPELINE and PIPELINE.is_trained:
            try:
                result = PIPELINE.predict_single(form_data)
                rec = {**form_data, **result,
                       'submitted_by': session['user'],
                       'timestamp':    time.strftime('%Y-%m-%d %H:%M:%S')}
                PREDICTIONS.append(rec)
            except Exception as e:
                result = {'error': str(e)}
        else:
            result = {'error': 'Model is still initialising — please wait a moment and refresh, then try again.'}

    return render_template('fake_profile_detection.html',
                           result=result,
                           form_data=form_data,
                           model_ready=bool(PIPELINE and PIPELINE.is_trained),
                           training_in_progress=_startup_thread.is_alive())


@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('home'))


# ──────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000, use_reloader=False)
