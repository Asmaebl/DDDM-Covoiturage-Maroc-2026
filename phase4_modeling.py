"""
Modélisation Prédictive & Interprétabilité
"""
import warnings; warnings.filterwarnings("ignore")
import os, pickle
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from math import radians, sin, cos, sqrt, atan2
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, KFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, roc_auc_score, f1_score, precision_score, recall_score,
    classification_report, confusion_matrix, ConfusionMatrixDisplay, roc_curve)
from sklearn.pipeline import Pipeline
import xgboost as xgb
import shap

SEED = 42
ASSETS = "assets"
MODELS = "models"
os.makedirs(ASSETS, exist_ok=True)
os.makedirs(MODELS, exist_ok=True)

PALETTE = {"primary":"#6C63FF","secondary":"#00C49F","accent":"#FF6B6B",
           "warn":"#FFB347","dark":"#1A1A2E","light":"#F8F9FA"}
sns.set_style("whitegrid")
plt.rcParams.update({"font.family":"DejaVu Sans","figure.dpi":130,
                     "axes.titlesize":13,"axes.labelsize":11})

# ══════════════════════════════════════════════
# 4.1 FEATURE ENGINEERING
# ══════════════════════════════════════════════
print("\n" + "="*60)
print("PHASE 4.1 — FEATURE ENGINEERING")
print("="*60)

# ── Uber ──
df_uber = pd.read_csv("uber_final.csv")

def haversine(lat1,lon1,lat2,lon2):
    R=6371.0
    dlat=radians(lat2-lat1); dlon=radians(lon2-lon1)
    a=sin(dlat/2)**2+cos(radians(lat1))*cos(radians(lat2))*sin(dlon/2)**2
    return R*2*atan2(sqrt(a),sqrt(1-a))

df_uber = df_uber.copy()
df_uber["distance_km"] = df_uber.apply(
    lambda r: haversine(r.pickup_latitude,r.pickup_longitude,
                        r.dropoff_latitude,r.dropoff_longitude), axis=1)
df_uber["is_peak"]    = df_uber["hour"].apply(lambda h: 1 if h in range(7,9) or h in range(17,20) else 0)
df_uber["is_weekend"] = df_uber["day_of_week"].apply(lambda d: 1 if d in ["Saturday","Sunday"] else 0)
df_uber = df_uber[(df_uber["distance_km"]>0.1)&(df_uber["distance_km"]<60)].reset_index(drop=True)
print(f"[uber_final] {df_uber.shape} | nouvelles colonnes: distance_km, is_peak, is_weekend")

# ── Rides ──
df_r = pd.read_csv("rides_merged_clustered.csv")
df_r = df_r.copy()

# Extract hour from ride_time "HH:MM"
df_r["ride_hour_calc"] = df_r["ride_time"].apply(lambda t: int(str(t).split(":")[0]) if pd.notna(t) else 12)
df_r["is_peak"]  = df_r["ride_hour_calc"].apply(lambda h: 1 if h in range(7,9) or h in range(17,20) else 0)
df_r["is_night"] = df_r["ride_hour_calc"].apply(lambda h: 1 if h>=22 or h<=5 else 0)

le_pay = LabelEncoder()
df_r["payment_encoded"] = le_pay.fit_transform(df_r["payment_method"].fillna("cash"))
df_r["gender_binary"]   = df_r["gender"].apply(lambda g: 1 if g=="Female" else 0)
df_r["fare_per_mile"]   = df_r["fare_amount"] / df_r["ride_distance_miles"].replace(0, np.nan)
df_r["fare_per_mile"]   = df_r["fare_per_mile"].fillna(df_r["fare_per_mile"].median())
df_r["speed_mph"]       = df_r["ride_distance_miles"] / (df_r["ride_duration_minutes"]/60).replace(0,np.nan)
df_r["speed_mph"]       = df_r["speed_mph"].fillna(df_r["speed_mph"].median())

# Labels
df_r["churn"] = (df_r["rating"] <= 2).astype(int)
fare_med = df_r["fare_amount"].median()
df_r["prefers_fixed_price"] = (
    (df_r["fare_amount"] < fare_med) &
    (df_r["rating"] >= 3) &
    (df_r["cluster"].isin([0,1,4]))
).astype(int)

print(f"[rides_merged] {df_r.shape} | churn: {df_r['churn'].sum()} | prefers_fixed: {df_r['prefers_fixed_price'].sum()}")

# ══════════════════════════════════════════════
# 4.2 MODÈLE 1 — RÉGRESSION LINÉAIRE
# ══════════════════════════════════════════════
print("\n"+"="*60+"\nPHASE 4.2 — MODÈLE 1 : RÉGRESSION LINÉAIRE\n"+"="*60)

FEAT_LR = ["distance_km","passenger_count","hour","is_peak","is_weekend","month"]
df_lr = df_uber[FEAT_LR+["fare_amount"]].dropna()
X_lr = df_lr[FEAT_LR].values; y_lr = df_lr["fare_amount"].values

X_tr,X_te,y_tr,y_te = train_test_split(X_lr,y_lr,test_size=0.2,random_state=SEED)
sc_lr = StandardScaler()
X_tr_s = sc_lr.fit_transform(X_tr); X_te_s = sc_lr.transform(X_te)

m_lr = LinearRegression()
m_lr.fit(X_tr_s,y_tr)

cv_lr = cross_val_score(Pipeline([("sc",StandardScaler()),("lr",LinearRegression())]),
    X_lr,y_lr,cv=KFold(5,shuffle=True,random_state=SEED),scoring="r2")
y_pred_lr = m_lr.predict(X_te_s)
rmse_lr = mean_squared_error(y_te,y_pred_lr)**0.5
mae_lr  = mean_absolute_error(y_te,y_pred_lr)
r2_lr   = r2_score(y_te,y_pred_lr)

print(f"  RMSE={rmse_lr:.4f}$ | MAE={mae_lr:.4f}$ | R²={r2_lr:.4f} | CV R²={cv_lr.mean():.4f}±{cv_lr.std():.4f}")
coef_lr = pd.Series(m_lr.coef_,index=FEAT_LR).sort_values(key=abs,ascending=False)
print("  Coefficients:"); print(coef_lr.to_string())

pickle.dump({"model":m_lr,"scaler":sc_lr,"features":FEAT_LR},open("models/model_lr.pkl","wb"))

# ══════════════════════════════════════════════
# 4.3 MODÈLE 2 — XGBOOST
# ══════════════════════════════════════════════
print("\n"+"="*60+"\nPHASE 4.3 — MODÈLE 2 : XGBOOST\n"+"="*60)

FEAT_XGB = ["fare_amount","ride_distance_miles","ride_duration_minutes",
            "ride_hour_calc","is_peak","payment_encoded","gender_binary",
            "fare_per_mile","speed_mph","cluster"]
df_xgb = df_r[FEAT_XGB+["prefers_fixed_price"]].dropna()
X_xgb = df_xgb[FEAT_XGB].values; y_xgb = df_xgb["prefers_fixed_price"].values

X_tr2,X_te2,y_tr2,y_te2 = train_test_split(X_xgb,y_xgb,test_size=0.2,random_state=SEED,stratify=y_xgb)
spw = (y_xgb==0).sum()/(y_xgb==1).sum()

m_xgb = xgb.XGBClassifier(n_estimators=300,max_depth=5,learning_rate=0.05,
    subsample=0.8,colsample_bytree=0.8,scale_pos_weight=spw,
    eval_metric="logloss",random_state=SEED,verbosity=0)
m_xgb.fit(X_tr2,y_tr2,eval_set=[(X_te2,y_te2)],verbose=False)

cv_xgb = cross_val_score(
    xgb.XGBClassifier(n_estimators=200,max_depth=5,learning_rate=0.1,
        scale_pos_weight=spw,eval_metric="logloss",random_state=SEED,verbosity=0),
    X_xgb,y_xgb,
    cv=StratifiedKFold(5,shuffle=True,random_state=SEED),scoring="roc_auc")

y_pred2 = m_xgb.predict(X_te2); y_prob2 = m_xgb.predict_proba(X_te2)[:,1]
acc_xgb=accuracy_score(y_te2,y_pred2); auc_xgb=roc_auc_score(y_te2,y_prob2)
f1_xgb=f1_score(y_te2,y_pred2); prec_xgb=precision_score(y_te2,y_pred2)
rec_xgb=recall_score(y_te2,y_pred2)

print(f"  ACC={acc_xgb:.4f} | AUC={auc_xgb:.4f} | F1={f1_xgb:.4f} | CV AUC={cv_xgb.mean():.4f}±{cv_xgb.std():.4f}")
print(classification_report(y_te2,y_pred2,target_names=["Pro inDrive","Pro Prix Fixe"]))
pickle.dump({"model":m_xgb,"features":FEAT_XGB},open("models/model_xgb.pkl","wb"))

# ══════════════════════════════════════════════
# 4.4 MODÈLE 3 — RÉGRESSION LOGISTIQUE (churn)
# ══════════════════════════════════════════════
print("\n"+"="*60+"\nPHASE 4.4 — MODÈLE 3 : LOGISTIC REGRESSION (CHURN)\n"+"="*60)

FEAT_LOG = ["fare_amount","ride_distance_miles","ride_duration_minutes",
            "ride_hour_calc","is_peak","payment_encoded","gender_binary",
            "fare_per_mile","speed_mph","cluster"]
df_log = df_r[FEAT_LOG+["churn"]].dropna()
X_log = df_log[FEAT_LOG].values; y_log = df_log["churn"].values

X_tr3,X_te3,y_tr3,y_te3 = train_test_split(X_log,y_log,test_size=0.2,random_state=SEED,stratify=y_log)
sc_log = StandardScaler()
X_tr3_s = sc_log.fit_transform(X_tr3); X_te3_s = sc_log.transform(X_te3)

m_log = LogisticRegression(class_weight="balanced",max_iter=1000,random_state=SEED,C=0.5)
m_log.fit(X_tr3_s,y_tr3)

cv_log = cross_val_score(
    Pipeline([("sc",StandardScaler()),("lr",LogisticRegression(class_weight="balanced",max_iter=1000,C=0.5,random_state=SEED))]),
    X_log,y_log,
    cv=StratifiedKFold(5,shuffle=True,random_state=SEED),scoring="roc_auc")

y_pred3 = m_log.predict(X_te3_s); y_prob3 = m_log.predict_proba(X_te3_s)[:,1]
acc_log=accuracy_score(y_te3,y_pred3); auc_log=roc_auc_score(y_te3,y_prob3)
f1_log=f1_score(y_te3,y_pred3); prec_log=precision_score(y_te3,y_pred3)
rec_log=recall_score(y_te3,y_pred3)

print(f"  ACC={acc_log:.4f} | AUC={auc_log:.4f} | F1={f1_log:.4f} | CV AUC={cv_log.mean():.4f}±{cv_log.std():.4f}")
print(classification_report(y_te3,y_pred3,target_names=["Rétenu","Churné"]))
pickle.dump({"model":m_log,"scaler":sc_log,"features":FEAT_LOG},open("models/model_log.pkl","wb"))

# ══════════════════════════════════════════════
# 4.5 SHAP VALUES
# ══════════════════════════════════════════════
print("\n"+"="*60+"\nPHASE 4.5 — SHAP VALUES\n"+"="*60)

# SHAP XGB
explainer_xgb = shap.TreeExplainer(m_xgb)
X_te2_df = pd.DataFrame(X_te2,columns=FEAT_XGB)
shap_xgb  = explainer_xgb(X_te2_df)

fig, axes = plt.subplots(1,2,figsize=(18,7))
ax=axes[0]
shap_mean = np.abs(shap_xgb.values).mean(axis=0)
fi = pd.Series(shap_mean,index=FEAT_XGB).sort_values(ascending=True)
bars = ax.barh(fi.index,fi.values,color=PALETTE["primary"],alpha=0.85)
ax.bar_label(bars,fmt="{:.3f}",padding=3,fontsize=9)
ax.set_xlabel("Importance SHAP moyenne"); ax.set_xlim(0,fi.max()*1.25)
ax.set_title("SHAP Global — XGBoost\nAdoption Prix Fixe vs inDrive",fontweight="bold")

ax2=axes[1]
probs2 = m_xgb.predict_proba(X_te2)[:,1]
cases_idx = [np.argsort(probs2)[-1], np.argsort(np.abs(probs2-0.5))[0], np.argsort(probs2)[0]]
cases_lbl = [f"Cas 1 — Pro Prix Fixe (p={probs2[cases_idx[0]]:.2f})",
             f"Cas 2 — Neutre (p={probs2[cases_idx[1]]:.2f})",
             f"Cas 3 — Pro inDrive (p={probs2[cases_idx[2]]:.2f})"]
case_df = pd.DataFrame([shap_xgb.values[i] for i in cases_idx],
                        columns=FEAT_XGB, index=cases_lbl)
im = ax2.imshow(case_df.values,cmap="RdYlGn",aspect="auto",vmin=-0.3,vmax=0.3)
ax2.set_xticks(range(len(FEAT_XGB))); ax2.set_xticklabels(FEAT_XGB,rotation=45,ha="right",fontsize=8)
ax2.set_yticks(range(3)); ax2.set_yticklabels(cases_lbl,fontsize=9)
ax2.set_title("SHAP Local — 3 Cas Individuels",fontweight="bold")
for i in range(3):
    for j in range(len(FEAT_XGB)):
        ax2.text(j,i,f"{case_df.iloc[i,j]:.2f}",ha="center",va="center",fontsize=7,fontweight="bold")
plt.colorbar(im,ax=ax2,label="Valeur SHAP")
plt.tight_layout()
plt.savefig(f"{ASSETS}/phase4_01_shap_xgboost.png",bbox_inches="tight"); plt.close()
print(f"  Saved: {ASSETS}/phase4_01_shap_xgboost.png")

# SHAP Logistic
explainer_log2 = shap.LinearExplainer(m_log, sc_log.transform(X_tr3))
X_te3_df = pd.DataFrame(X_te3_s,columns=FEAT_LOG)
shap_log = explainer_log2(X_te3_df)
fig,ax = plt.subplots(figsize=(10,6))
shap_mean_log = np.abs(shap_log.values).mean(axis=0)
fi2 = pd.Series(shap_mean_log,index=FEAT_LOG).sort_values(ascending=True)
bars2 = ax.barh(fi2.index,fi2.values,color=PALETTE["accent"],alpha=0.85)
ax.bar_label(bars2,fmt="{:.4f}",padding=3,fontsize=9)
ax.set_title("SHAP Global — Régression Logistique\nPrédiction Churn",fontweight="bold")
plt.tight_layout()
plt.savefig(f"{ASSETS}/phase4_02_shap_logistic.png",bbox_inches="tight"); plt.close()
print(f"  Saved: {ASSETS}/phase4_02_shap_logistic.png")

# ══════════════════════════════════════════════
# 4.6 VISUALISATIONS MODÈLES
# ══════════════════════════════════════════════
print("\n  Génération visualisations...")

fig,axes = plt.subplots(2,3,figsize=(20,12))
fig.suptitle("Évaluation comparative — 3 Modèles DDDM Maroc",fontsize=16,fontweight="bold",y=1.01)

# Plot 1: LR Predicted vs Actual
ax=axes[0,0]
idx_s = np.random.RandomState(SEED).choice(len(y_te),min(3000,len(y_te)),replace=False)
ax.scatter(y_te[idx_s],y_pred_lr[idx_s],alpha=0.3,s=10,color=PALETTE["primary"])
lims=[min(y_te.min(),y_pred_lr.min()),max(y_te.max(),y_pred_lr.max())]
ax.plot(lims,lims,"r--",lw=2,label="Parfait")
ax.set_xlabel("Prix réel ($)"); ax.set_ylabel("Prix prédit ($)")
ax.set_title(f"Régression Linéaire\nRMSE={rmse_lr:.2f}$ | R²={r2_lr:.3f}",fontweight="bold")
ax.legend(fontsize=9)

# Plot 2: LR Résidus
ax=axes[0,1]
res_lr = y_te - y_pred_lr
ax.hist(res_lr,bins=60,color=PALETTE["primary"],alpha=0.7,edgecolor="white")
ax.axvline(0,color="red",lw=2,linestyle="--")
ax.set_xlabel("Résidu (réel−prédit)"); ax.set_ylabel("Fréquence")
ax.set_title("Distribution des résidus\nRégression Linéaire",fontweight="bold")
ax.text(0.98,0.95,f"µ={res_lr.mean():.3f}\nσ={res_lr.std():.3f}",
        transform=ax.transAxes,ha="right",va="top",fontsize=9,
        bbox=dict(boxstyle="round",facecolor="white",alpha=0.8))

# Plot 3: CV comparison
ax=axes[0,2]
cv_labels=["LR (R²)","XGBoost (AUC)","LogReg (AUC)"]
cv_means=[cv_lr.mean(),cv_xgb.mean(),cv_log.mean()]
cv_stds=[cv_lr.std(),cv_xgb.std(),cv_log.std()]
bars3=ax.bar(cv_labels,cv_means,yerr=cv_stds,capsize=6,
             color=[PALETTE["primary"],PALETTE["secondary"],PALETTE["accent"]],
             alpha=0.85,edgecolor="white",linewidth=1.5)
ax.bar_label(bars3,fmt="{:.3f}",padding=4,fontsize=11,fontweight="bold")
ax.set_ylim(0,1.15); ax.set_ylabel("Score CV (k=5)")
ax.set_title("Comparaison CV — 3 Modèles",fontweight="bold")
ax.axhline(0.7,color="gray",ls="--",lw=1,alpha=0.5)

# Plot 4: XGB confusion matrix
ax=axes[1,0]
cm2=confusion_matrix(y_te2,y_pred2)
ConfusionMatrixDisplay(cm2,display_labels=["Pro inDrive","Pro Prix Fixe"]).plot(ax=ax,cmap="Blues",colorbar=False)
ax.set_title(f"XGBoost — Matrice Confusion\nAUC={auc_xgb:.3f} | F1={f1_xgb:.3f}",fontweight="bold")

# Plot 5: LogReg confusion matrix
ax=axes[1,1]
cm3=confusion_matrix(y_te3,y_pred3)
ConfusionMatrixDisplay(cm3,display_labels=["Rétenu","Churné"]).plot(ax=ax,cmap="Reds",colorbar=False)
ax.set_title(f"LogReg — Matrice Confusion\nAUC={auc_log:.3f} | F1={f1_log:.3f}",fontweight="bold")

# Plot 6: XGB feature importance
ax=axes[1,2]
fi_xgb=pd.Series(m_xgb.feature_importances_,index=FEAT_XGB).sort_values(ascending=True)
b6=ax.barh(fi_xgb.index,fi_xgb.values,color=PALETTE["secondary"],alpha=0.85)
ax.bar_label(b6,fmt="{:.3f}",padding=2,fontsize=8)
ax.set_xlabel("Importance (gain)"); ax.set_title("XGBoost Feature Importance",fontweight="bold")

plt.tight_layout()
plt.savefig(f"{ASSETS}/phase4_03_model_evaluation.png",bbox_inches="tight"); plt.close()
print(f"  Saved: {ASSETS}/phase4_03_model_evaluation.png")

# ROC curves
fig,ax = plt.subplots(figsize=(8,6))
fpr2,tpr2,_ = roc_curve(y_te2,y_prob2)
fpr3,tpr3,_ = roc_curve(y_te3,y_prob3)
ax.plot(fpr2,tpr2,color=PALETTE["secondary"],lw=2.5,label=f"XGBoost Adoption (AUC={auc_xgb:.3f})")
ax.plot(fpr3,tpr3,color=PALETTE["accent"],lw=2.5,label=f"LogReg Churn (AUC={auc_log:.3f})")
ax.plot([0,1],[0,1],"k--",lw=1.5,alpha=0.5,label="Aléatoire")
ax.fill_between(fpr2,tpr2,alpha=0.07,color=PALETTE["secondary"])
ax.fill_between(fpr3,tpr3,alpha=0.07,color=PALETTE["accent"])
ax.set_xlabel("FPR"); ax.set_ylabel("TPR"); ax.set_xlim(0,1); ax.set_ylim(0,1.02)
ax.set_title("Courbes ROC — XGBoost vs Régression Logistique",fontweight="bold")
ax.legend(loc="lower right",fontsize=10)
plt.tight_layout()
plt.savefig(f"{ASSETS}/phase4_04_roc_curves.png",bbox_inches="tight"); plt.close()
print(f"  Saved: {ASSETS}/phase4_04_roc_curves.png")

# Churn & preference by cluster
churn_cl = df_r.groupby("cluster")["churn"].agg(["mean","sum","count"]).reset_index()
churn_cl.columns=["cluster","churn_rate","churn_count","total"]
pref_cl  = df_r.groupby("cluster")["prefers_fixed_price"].mean().reset_index()

fig,axes=plt.subplots(1,2,figsize=(14,5))
colors_cl=["#6C63FF","#00C49F","#FF6B6B","#FFB347","#9B59B6","#3498DB","#E74C3C","#1ABC9C"]
ax=axes[0]
b7=ax.bar(churn_cl["cluster"].astype(str),churn_cl["churn_rate"]*100,
          color=colors_cl[:len(churn_cl)],alpha=0.85,edgecolor="white")
ax.bar_label(b7,fmt="{:.1f}%",padding=3,fontsize=10,fontweight="bold")
ax.set_xlabel("Segment (Cluster)"); ax.set_ylabel("Taux churn (%)")
ax.set_title("Taux de Churn Prédit par Segment\n(LogReg)",fontweight="bold")
ax2=axes[1]
b8=ax2.bar(pref_cl["cluster"].astype(str),pref_cl["prefers_fixed_price"]*100,
           color=colors_cl[:len(pref_cl)],alpha=0.85,edgecolor="white")
ax2.bar_label(b8,fmt="{:.1f}%",padding=3,fontsize=10,fontweight="bold")
ax2.set_ylim(0,100); ax2.set_xlabel("Segment (Cluster)"); ax2.set_ylabel("% préférant prix fixe")
ax2.set_title("Préférence Prix Fixe par Segment",fontweight="bold")
plt.tight_layout()
plt.savefig(f"{ASSETS}/phase4_05_churn_preference_by_cluster.png",bbox_inches="tight"); plt.close()
print(f"  Saved: {ASSETS}/phase4_05_churn_preference_by_cluster.png")

# Save enriched dataframe
df_r.to_csv("rides_final_phase4.csv",index=False)
churn_cl.to_csv("models/churn_by_cluster.csv",index=False)
pref_cl.to_csv("models/pref_by_cluster.csv",index=False)

# Save metrics for dashboard
metrics = {
    "lr": {"RMSE":round(rmse_lr,4),"MAE":round(mae_lr,4),"R2":round(r2_lr,4),"CV_R2":round(cv_lr.mean(),4)},
    "xgb":{"ACC":round(acc_xgb,4),"AUC":round(auc_xgb,4),"F1":round(f1_xgb,4),"CV_AUC":round(cv_xgb.mean(),4)},
    "log":{"ACC":round(acc_log,4),"AUC":round(auc_log,4),"F1":round(f1_log,4),"CV_AUC":round(cv_log.mean(),4)},
}
import json
with open("models/metrics.json","w") as f: json.dump(metrics,f,indent=2)

print("\n"+"="*60)
print("✓ PHASE 4 TERMINÉE — tous les modèles et assets sauvegardés")
print("="*60)
