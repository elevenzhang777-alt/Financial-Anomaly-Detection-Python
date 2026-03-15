import pandas as pd
from sklearn.ensemble import IsolationForest
# Load financial datasets
def run_audit():
    try:
        gl=pd.read_csv('General_ledger.csv')
        bank=pd.read_csv('Bank_Statement.csv')
        print("Data loaded successfully, Starting automated audit...")
    except:
        print("Data files not found. Please ensure CSVs are in the directory")
        return

    # 1.Automated Reconciliation Logic
    # we use an 'outer merge' to identify items present in one list but not the other
    recon=pd.merge(gl,bank,how='outer',on=['Vendor','Amount'],indicator=True)
    #Identify 'Outstanding Items' (In ledger but not in bank)
    outstanding=recon[recon['_merge']=='left_only']

    # 2.AI Anomaly Detection(Machine learning)
    # Using Isolation Forest to flag transactions that deviate from normal spending behavior
    model=IsolationForest(contamination=0.02,random_state=42)
    gl['Is_Anomaly']=model.fit_predict(gl[['Amount']])
    anomalies=gl[gl['Is_Anomaly']==-1]

    # 3. Audit Summary Report
    print(f"\n--- Audit Summary Report---")
    print(f"Total Transactions Processed:{len(gl)}")
    print(f'Outstanding Reconciliation Items:{len(outstanding)}')
    print(f'AI-Flagged High Risk Anomalies:{len(anomalies)}')
print(f"--------------------------------")


if __name__ == '__main__':
    run_audit()
