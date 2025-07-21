import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import streamlit as st

def generate_dataset():
    np.random.seed(42)
    X = np.random.randint(0, 3, size=(500, 6))
    y = (X.sum(axis=1) > 9).astype(int)
    cols = ['rs4244285', 'rs9923231', 'rs4149056', 'rs1057910', 'rs1799853', 'rs5030655']
    df = pd.DataFrame(X, columns=cols)
    df['Side_Effect'] = y
    return df

class SNPNet(nn.Module):
    def __init__(self):
        super(SNPNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(6, 16),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)

@st.cache_resource
def train_model():
    df = generate_dataset()
    X = df.drop(columns='Side_Effect').values
    y = df['Side_Effect'].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)

    model = SNPNet()
    loss_fn = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(100):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train_tensor)
        loss = loss_fn(outputs, y_train_tensor)
        loss.backward()
        optimizer.step()

    return model, scaler

def snp_to_num(geno):
    mapping = {"GG": 0, "AG": 1, "AA": 2}
    return mapping.get(geno, 1)

drug_list = [
    "Paracetamol", "Aspirin", "Atorvastatin", "Warfarin", "Clopidogrel",
    "Metformin", "Omeprazole", "Simvastatin", "Lisinopril", "Losartan",
    "Levothyroxine", "Amlodipine", "Metoprolol", "Alprazolam", "Gabapentin",
    "Hydrochlorothiazide", "Sertraline", "Furosemide", "Pantoprazole", "Prednisone",
    "Tramadol", "Citalopram", "Tamsulosin", "Fluoxetine", "Meloxicam",
    "Cetirizine", "Clonazepam", "Duloxetine", "Esomeprazole", "Carvedilol",
    "Ibuprofen", "Rosuvastatin", "Zolpidem", "Spironolactone", "Ranitidine",
    "Cyclobenzaprine", "Bupropion", "Venlafaxine", "Allopurinol", "Loratadine",
    "Oxycodone", "Levofloxacin", "Morphine", "Azithromycin", "Nitrofurantoin",
    "Doxycycline", "Methotrexate", "Hydroxyzine", "Propranolol", "Hydrocodone",
    "Clindamycin", "Fluconazole", "Gabapentin", "Ketorolac", "Labetalol",
    "Methylprednisolone", "Nifedipine", "Olanzapine", "Pantoprazole", "Quetiapine",
    "Risperidone", "Sulfamethoxazole", "Temazepam", "Topiramate", "Valacyclovir",
    "Valsartan", "Venlafaxine", "Vitamin D", "Zidovudine", "Zolpidem",
    "Acyclovir", "Albuterol", "Amitriptyline", "Amoxicillin", "Ampicillin",
    "Bisoprolol", "Buprenorphine", "Calcium Carbonate", "Carbamazepine", "Chlorpheniramine",
    "Chlorzoxazone", "Cholecalciferol", "Clobetasol", "Clonidine", "Cyclobenzaprine",
    "Desvenlafaxine", "Dexamethasone", "Diazepam", "Dicyclomine", "Diphenhydramine",
    "Divalproex", "Doxazosin", "Duloxetine", "Enalapril", "Escitalopram"
]

def main():
    st.title("🔬 AI-Powered Drug Side Effect Predictor")
    st.write("Upload your genotype data or manually input SNP values.")

    model, scaler = train_model()

    drug = st.selectbox("💊 Select Drug", drug_list)

    uploaded_file = st.file_uploader("📂 Upload CSV with SNPs", type=["csv"])
    snp_cols = ['rs4244285', 'rs9923231', 'rs4149056', 'rs1057910', 'rs1799853', 'rs5030655']

    if uploaded_file is not None:
        try:
            user_df = pd.read_csv(uploaded_file)
            snps = [snp_to_num(user_df.loc[user_df['SNP'] == snp, 'Genotype'].values[0]) for snp in snp_cols]
            predict_and_show(model, scaler, snps, drug)
        except Exception as e:
            st.error(f"⚠️ Error reading file: {e}")
    else:
        snps = []
        for snp in snp_cols:
            val = st.selectbox(f"{snp}", ["GG", "AG", "AA"], key=snp)
            snps.append(snp_to_num(val))

        if st.button("🔍 Predict Side Effect"):
            predict_and_show(model, scaler, snps, drug)

def predict_and_show(model, scaler, snps, drug):
    input_scaled = scaler.transform([snps])
    input_tensor = torch.tensor(input_scaled, dtype=torch.float32)

    with torch.no_grad():
        pred_prob = model(input_tensor).item()
        pred = round(pred_prob)

    st.markdown("---")
    if pred == 1:
        st.error(f"⚠️ High risk of side effects from **{drug}**. (Risk Score: {pred_prob:.2f})")
    else:
        st.success(f"✅ Low risk of side effects from **{drug}**. (Risk Score: {pred_prob:.2f})")

if __name__ == "__main__":
    main()
