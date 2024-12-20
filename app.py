from flask import Flask, render_template, request, redirect, url_for, session
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,confusion_matrix
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import classification_report
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from dotenv import load_dotenv


load_dotenv()
SECRET_KEY = os.getenv("SECRET_KEY")


app = Flask(__name__)
app.secret_key = SECRET_KEY  
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['STATIC_FOLDER'] = 'static'

if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

if not os.path.exists(app.config['STATIC_FOLDER']):
    os.makedirs(app.config['STATIC_FOLDER'])

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return 'No file part in the request'
    
    file = request.files['file']
    
    if file.filename == '':
        return 'No selected file'
    
    if file:
        filename = file.filename
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        
        df = pd.read_csv(file_path)
        
        session['file_path'] = file_path
        session['columns'] = df.columns.tolist()
        session['shape'] = tuple(map(int, df.shape))  
        session['head'] = df.head().to_html(classes='data', header=True, index=False)
        session['duplicates'] = int(df.duplicated().sum())  
        session['isnull'] = int(df.isnull().sum().sum())  
        session['isna'] = int(df.isna().sum().sum())  
        
        return redirect(url_for('dataset_info'))

@app.route('/dataset_info')
def dataset_info():
    
    file_path = session.get('file_path')
    columns = session.get('columns')
    shape = session.get('shape')
    head = session.get('head')
    duplicates = session.get('duplicates')
    isnull = session.get('isnull')
    isna = session.get('isna')
    
    if not file_path:
        return redirect(url_for('index'))
    
    return render_template('dataset_info.html', 
                           file_path=file_path, 
                           columns=columns, 
                           shape=shape, 
                           head=head,
                           duplicates=duplicates,
                           isnull=isnull,
                           isna=isna)

@app.route('/select_columns_to_drop', methods=['GET', 'POST'])
def select_columns_to_drop():
    file_path = session.get('file_path')
    
    if not file_path or not os.path.exists(file_path):
        return redirect(url_for('dataset_info'))
    
    df = pd.read_csv(file_path)
        
    if request.method == 'POST':
        columns_to_drop = request.form.getlist('drop_columns')
        target_column = request.form.get('target_column')
        
        
        if columns_to_drop:
            df = df.drop(columns=columns_to_drop)
            df.to_csv(file_path, index=False)
        
        session['target_column'] = target_column
        
        session['columns'] = df.columns.tolist()
        session['shape'] = tuple(map(int, df.shape))  
        session['head'] = df.head().to_html(classes='data', header=True, index=False)
        
        return redirect(url_for('select_columns_to_encode'))
    
    return render_template('select_columns_to_drop.html', columns=df.columns)

@app.route('/drop_columns', methods=['POST'])
def drop_columns():
    selected_columns = request.form.getlist('columns_to_drop')
    file_path = session.get('file_path')
    
    if not file_path or not os.path.exists(file_path):
        return redirect(url_for('index'))
    
    df = pd.read_csv(file_path)
    df.drop(columns=selected_columns, inplace=True)
    df.to_csv(file_path, index=False)
    
    session['columns'] = df.columns.tolist()
    session['shape'] = tuple(map(int, df.shape))  
    session['head'] = df.head().to_html(classes='data', header=True, index=False)
    
    return redirect(url_for('select_columns_to_encode'))

@app.route('/select_columns_to_encode')
def select_columns_to_encode():
    columns = session.get('columns')
    
    if not columns:
        return redirect(url_for('dataset_info'))
    
    return render_template('select_columns_to_encode.html', columns=columns)

@app.route('/apply_encoding', methods=['POST'])
def apply_encoding():
    file_path = session.get('file_path')
    columns = session.get('columns')

    if not file_path or not os.path.exists(file_path):
        return redirect(url_for('index'))
    
    df = pd.read_csv(file_path)
    
    ordinal_columns = request.form.getlist('ordinal_columns')
    onehot_columns = request.form.getlist('onehot_columns')
    
    if ordinal_columns:
        encoder = OrdinalEncoder()
        df[ordinal_columns] = encoder.fit_transform(df[ordinal_columns])
    
    if onehot_columns:
        df = pd.get_dummies(df, columns=onehot_columns,drop_first=True)
    
    df.to_csv(file_path, index=False)
    session['columns'] = df.columns.tolist()
    session['shape'] = tuple(map(int, df.shape))  
    session['head'] = df.head().to_html(classes='data', header=True, index=False)
    
    return redirect(url_for('show_correlation_heatmap'))

@app.route('/show_correlation_heatmap')
def show_correlation_heatmap():
    file_path = session.get('file_path')
    target_column = session.get('target_column')
    
    if not file_path or not os.path.exists(file_path):
        return redirect(url_for('index'))
    
    df = pd.read_csv(file_path)

    corr_matrix = df.corr()
    fig = px.imshow(corr_matrix, 
                text_auto=True, 
                color_continuous_scale='RdBu', 
                title="Correlation Matrix",
                aspect='auto')
    heatmap_html = fig.to_html(full_html=False)
    
    if target_column in df.columns:
        corr_target = abs(corr_matrix[target_column]).sort_values(ascending=False)
    else:
        corr_target = pd.Series(dtype=float)  
    
    corr_target_html = corr_target.to_frame(name='Correlation with ' + target_column).reset_index().to_html(classes='data', header=True, index=False)
    
    top_features = corr_target.drop(target_column).head(3).index.tolist()
    session['top_features'] = top_features
    
    return render_template('show_correlation_heatmap.html', 
                           heatmap_html=heatmap_html,
                           corr_target_html=corr_target_html,
                           df_html=df.head().to_html(classes='data', header=True, index=False))

@app.route('/show_box_plots', methods=['GET', 'POST'])
def show_box_plots():
    file_path = session.get('file_path')
    target_column = session.get('target_column')
    
    if not file_path or not os.path.exists(file_path):
        return redirect(url_for('index'))
    
    df = pd.read_csv(file_path)
    
    corr_matrix = df.corr()
    if target_column in df.columns:
        corr_target = abs(corr_matrix[target_column]).sort_values(ascending=False)
        top_features = corr_target.index[1:4]  
        
        fig = make_subplots(rows=1, cols=len(top_features), subplot_titles=top_features)
        for i, feature in enumerate(top_features):
            fig.add_trace(
            go.Box(y=df[feature], name=feature),
            row=1, col=i+1
        )
        fig.update_layout(title="Box Plots of Top Features")
        boxplot_html = fig.to_html(full_html=False)

        
        df_html = df.head().to_html(classes='data', header=True, index=False)
        
        return render_template('show_box_plots.html', 
                               boxplot_html=boxplot_html,
                               df_html=df_html,
                               top_features=top_features)

@app.route('/remove_outliers', methods=['POST'])
def remove_outliers():
    file_path = session.get('file_path')
    target_column = session.get('target_column')
    
    if not file_path or not os.path.exists(file_path):
        return redirect(url_for('index'))
    
    df = pd.read_csv(file_path)
    
    columns_to_check = request.form.getlist('columns_to_remove')
    for column in columns_to_check:
        if column in df.columns:
            
            Q1 = df[column].quantile(0.25)
            Q3 = df[column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            
            df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    
    df.to_csv(file_path, index=False)
    session['columns'] = df.columns.tolist()
    session['shape'] = tuple(map(int, df.shape))  
    session['head'] = df.head().to_html(classes='data', header=True, index=False)
    
    return redirect(url_for('show_box_plots'))

@app.route('/train_model', methods=['POST','GET'])
def train_model():
    if request.method=='POST':
        file_path = session.get('file_path')
        target_column = session.get('target_column')
    
        if not file_path or not os.path.exists(file_path):
            return redirect(url_for('index'))
    
        df = pd.read_csv(file_path)
        X = df.drop(columns=[target_column])
        y = df[target_column]

        test_size=0.2
        random_state=42
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

        model=LogisticRegression()
        model.fit(X_train,y_train)
        y_pred=model.predict(X_test)

        accuracy=accuracy_score(y_test,y_pred)
        matrix=confusion_matrix(y_test,y_pred)
        report = classification_report(y_test, y_pred, target_names=model.classes_, output_dict=True)

        fig = px.imshow(matrix, 
                text_auto=True, 
                x=model.classes_, 
                y=model.classes_, 
                color_continuous_scale='Blues',
                title="Confusion Matrix")
        confusion_matrix_html = fig.to_html(full_html=False)


        report_html = pd.DataFrame(report).transpose().to_html(classes='data', header=True, index=True)
        return render_template('model_result.html',
                               test_size=test_size,
                               random_state=random_state,
                               accuracy=accuracy,
                               confusion_matrix_html=confusion_matrix_html,
                               report_html=report_html)



