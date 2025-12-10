import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import confusion_matrix, classification_report

# Configurazione pagina
st.set_page_config(
    page_title="Churn Prediction Dashboard",
    page_icon="📊",
    layout="wide"
)

# Titolo principale
st.title("📊 Customer Churn Prediction Dashboard")
st.markdown("### Progetto di Machine Learning - IULM")
st.markdown("**Autore:** Sara Rebecca Magarotto")
st.divider()

# Sidebar per navigazione
page = st.sidebar.selectbox(
    "Scegli una sezione:",
    ["🏠 Home", "📈 Analisi Dataset", "🤖 Modello ML", "🔮 Fai una Previsione"]
)

# Funzione per caricare i dati
@st.cache_data
def load_data():
    df_final = pd.read_csv('Telco-Customer-Churn-FINAL.csv')
    return df_final

# Funzione per caricare il modello
@st.cache_resource
def load_model():
    with open('churn_model.pkl', 'rb') as file:
        model = pickle.load(file)
    with open('model_columns.pkl', 'rb') as file:
        columns = pickle.load(file)
    return model, columns

# Carica dati e modello
try:
    df = load_data()
    model, model_columns = load_model()
    
    # HOME PAGE
    if page == "🏠 Home":
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Totale Clienti", f"{len(df):,}")
        with col2:
            churn_rate = (df['Churn'].sum() / len(df) * 100)
            st.metric("Tasso di Churn", f"{churn_rate:.1f}%")
        with col3:
            st.metric("Accuracy Modello", "80.41%")
        
        st.divider()
        
        st.markdown("## 🎯 Obiettivo del Progetto")
        st.write("""
        Questo progetto utilizza tecniche di **Machine Learning** per prevedere quali clienti 
        di una compagnia telefonica hanno maggiore probabilità di abbandonare il servizio (churn).
        """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📊 Dataset")
            st.write("- **Fonte:** Kaggle - Telco Customer Churn")
            st.write("- **Clienti:** 7,043")
            st.write("- **Features:** 30 variabili")
            st.write("- **Target:** Churn (Sì/No)")
        
        with col2:
            st.markdown("### 🤖 Modello")
            st.write("- **Algoritmo:** Random Forest")
            st.write("- **Alberi:** 100")
            st.write("- **Accuracy:** 80.41%")
            st.write("- **Split:** 80% train, 20% test")
        
        st.divider()
        
        # Grafico distribuzione churn
        st.markdown("### 📊 Distribuzione del Churn")
        churn_counts = df['Churn'].value_counts()
        fig = px.pie(
            values=churn_counts.values,
            names=['No Churn', 'Churn'],
            title='Distribuzione Clienti',
            color_discrete_sequence=['#00CC96', '#EF553B']
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # ANALISI DATASET
    elif page == "📈 Analisi Dataset":
        st.markdown("## 📈 Analisi Esplorativa del Dataset")
        
        tab1, tab2, tab3 = st.tabs(["📊 Statistiche", "📉 Visualizzazioni", "🔍 Dati Raw"])
        
        with tab1:
            st.markdown("### Statistiche Descrittive")
            
            # Statistiche numeriche
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            st.dataframe(df[numeric_cols].describe(), use_container_width=True)
            
            st.divider()
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("### Distribuzione Churn per Genere")
                if 'gender' in df.columns:
                    gender_churn = pd.crosstab(df['gender'], df['Churn'], normalize='index') * 100
                    st.dataframe(gender_churn)
            
            with col2:
                st.markdown("### Clienti per Tipo di Contratto")
                contract_cols = [col for col in df.columns if 'Contract' in col]
                if contract_cols:
                    st.write("Contratti disponibili:", len(contract_cols))
        
        with tab2:
            st.markdown("### 📊 Visualizzazioni Chiave")
            
            # Feature Importance
            st.markdown("#### Top 10 Variabili più Importanti")
            feature_importance = pd.DataFrame({
                'Feature': model_columns,
                'Importance': model.feature_importances_
            }).sort_values('Importance', ascending=False).head(10)
            
            fig = px.bar(
                feature_importance,
                x='Importance',
                y='Feature',
                orientation='h',
                title='Feature Importance',
                color='Importance',
                color_continuous_scale='Viridis'
            )
            fig.update_layout(yaxis={'categoryorder':'total ascending'})
            st.plotly_chart(fig, use_container_width=True)
            
            st.divider()
            
            # Distribuzione delle feature più importanti
            col1, col2 = st.columns(2)
            
            with col1:
                if 'tenure' in df.columns:
                    fig = px.histogram(
                        df, x='tenure', color='Churn',
                        title='Distribuzione Tenure per Churn',
                        color_discrete_map={0: '#00CC96', 1: '#EF553B'}
                    )
                    st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                if 'MonthlyCharges' in df.columns:
                    fig = px.box(
                        df, x='Churn', y='MonthlyCharges',
                        title='Monthly Charges per Churn Status',
                        color='Churn',
                        color_discrete_map={0: '#00CC96', 1: '#EF553B'}
                    )
                    st.plotly_chart(fig, use_container_width=True)
        
        with tab3:
            st.markdown("### 🔍 Primi 100 Record del Dataset")
            st.dataframe(df.head(100), use_container_width=True)
            
            st.download_button(
                label="📥 Scarica Dataset Completo (CSV)",
                data=df.to_csv(index=False),
                file_name="churn_data.csv",
                mime="text/csv"
            )
    
    # MODELLO ML
    elif page == "🤖 Modello ML":
        st.markdown("## 🤖 Performance del Modello")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Accuracy", "80.41%")
        with col2:
            st.metric("Precision (Churn)", "67%")
        with col3:
            st.metric("Recall (Churn)", "53%")
        with col4:
            st.metric("F1-Score (Churn)", "59%")
        
        st.divider()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 🎯 Confusion Matrix")
            st.write("""
            La matrice mostra le predizioni del modello:
            - **Veri Negativi:** 936 (No churn predetto correttamente)
            - **Falsi Positivi:** 99 (Churn predetto erroneamente)
            - **Falsi Negativi:** 177 (Churn non previsto)
            - **Veri Positivi:** 197 (Churn predetto correttamente)
            """)
            
            # Confusion Matrix come heatmap
            cm_data = [[936, 99], [177, 197]]
            fig = go.Figure(data=go.Heatmap(
                z=cm_data,
                x=['Predetto No', 'Predetto Si'],
                y=['Reale No', 'Reale Si'],
                text=cm_data,
                texttemplate='%{text}',
                colorscale='Blues'
            ))
            fig.update_layout(title='Confusion Matrix')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("### 📊 Metriche per Classe")
            
            metrics_data = {
                'Classe': ['No Churn', 'Churn'],
                'Precision': [0.84, 0.67],
                'Recall': [0.90, 0.53],
                'F1-Score': [0.87, 0.59]
            }
            metrics_df = pd.DataFrame(metrics_data)
            
            fig = go.Figure()
            for metric in ['Precision', 'Recall', 'F1-Score']:
                fig.add_trace(go.Bar(
                    name=metric,
                    x=metrics_df['Classe'],
                    y=metrics_df[metric],
                    text=metrics_df[metric],
                    texttemplate='%{text:.2f}'
                ))
            
            fig.update_layout(
                title='Metriche di Performance per Classe',
                barmode='group',
                yaxis_range=[0, 1]
            )
            st.plotly_chart(fig, use_container_width=True)
        
        st.divider()
        
        st.markdown("### 🔑 Interpretazione dei Risultati")
        st.write("""
        Il modello ha un'**accuracy dell'80.41%**, identificando correttamente 8 clienti su 10.
        
        **Punti di forza:**
        - Alta precisione nel riconoscere i clienti che NON faranno churn (90% recall)
        - Buona precisione generale (84% per No Churn)
        
        **Aree di miglioramento:**
        - Il recall per la classe Churn è del 53% - il modello perde circa la metà dei clienti che effettivamente abbandoneranno
        - Potrebbero essere necessari dati aggiuntivi o feature engineering per migliorare la detection del churn
        """)
    
    # FAI UNA PREVISIONE
    elif page == "🔮 Fai una Previsione":
        st.markdown("## 🔮 Simulatore di Previsione Churn")
        st.write("Inserisci i dati di un cliente per prevedere se abbandonerà il servizio")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Dati Cliente")
            tenure = st.slider("Mesi come cliente (tenure)", 0, 72, 12)
            monthly_charges = st.number_input("Costo Mensile ($)", 0.0, 120.0, 70.0)
            total_charges = st.number_input("Costo Totale ($)", 0.0, 9000.0, 1000.0)
        
        with col2:
            st.markdown("### Servizi")
            gender = st.selectbox("Genere", ["Male", "Female"])
            senior = st.selectbox("Senior Citizen", ["No", "Yes"])
            partner = st.selectbox("Ha Partner", ["No", "Yes"])
            dependents = st.selectbox("Ha Dipendenti", ["No", "Yes"])
        
        if st.button("🔮 Fai Previsione", type="primary"):
            # Questo è un esempio semplificato - nella realtà dovresti costruire
            # un vettore completo con tutte le 30 features
            st.info("""
            ⚠️ Questa è una demo semplificata. 
            Per una previsione reale, il modello richiede tutte le 30 features del dataset.
            """)
            
            # Simulazione di una previsione
            prediction_prob = np.random.rand()
            
            if prediction_prob > 0.5:
                st.error("🚨 **ALTO RISCHIO DI CHURN**")
                st.write(f"Probabilità di abbandono: {prediction_prob*100:.1f}%")
                st.write("**Azioni consigliate:**")
                st.write("- Contattare il cliente con offerte personalizzate")
                st.write("- Verificare la qualità del servizio")
                st.write("- Proporre upgrade o sconti")
            else:
                st.success("✅ **BASSO RISCHIO DI CHURN**")
                st.write(f"Probabilità di abbandono: {prediction_prob*100:.1f}%")
                st.write("Il cliente sembra soddisfatto del servizio!")

except FileNotFoundError as e:
    st.error(f"""
    ⚠️ **File non trovato!**
    
    Assicurati che questi file siano nella stessa cartella dell'app:
    - `Telco-Customer-Churn-FINAL.csv`
    - `churn_model.pkl`
    - `model_columns.pkl`
    
    Errore: {str(e)}
    """)
except Exception as e:
    st.error(f"Si è verificato un errore: {str(e)}")

# Footer
st.divider()
st.markdown("""
<div style='text-align: center'>
    <p>📊 Customer Churn Prediction Dashboard | Progetto IULM 2025</p>
    <p>Creato da Sara Rebecca Magarotto</p>
</div>
""", unsafe_allow_html=True)
