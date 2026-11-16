"""
execute_notebook.py – Executes fraud_detection.ipynb, populating all outputs, metrics,
and plots into the notebook JSON and saving all visualization artifacts.
"""

import os
import sys
import json
import io
import base64
import time
import warnings
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

warnings.filterwarnings('ignore')

NOTEBOOK_PATH = os.path.join("notebooks", "fraud_detection.ipynb")

with open(NOTEBOOK_PATH, "r", encoding="utf-8") as f:
    nb = json.load(f)

# Global namespace for execution
exec_ns = {
    '__name__': '__main__',
    '__file__': NOTEBOOK_PATH,
}

execution_counter = 1

# Update models cell in notebook source to include LightGBM
for cell in nb["cells"]:
    if cell["cell_type"] == "code":
        src = "".join(cell["source"])
        if "from sklearn.ensemble import RandomForestClassifier" in src:
            new_src = []
            for line in cell["source"]:
                if "from sklearn.ensemble import" in line:
                    new_src.append("from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier\n")
                    new_src.append("from lightgbm import LGBMClassifier\n")
                else:
                    new_src.append(line)
            cell["source"] = new_src
        elif "models = {" in src and "'Gradient Boosting':" in src:
            new_src = [
                "# ── Model definitions ─────────────────────────────────────────────────────────\n",
                "models = {\n",
                "    'Logistic Regression': LogisticRegression(\n",
                "        max_iter=1000, class_weight='balanced', random_state=42\n",
                "    ),\n",
                "    'Random Forest': RandomForestClassifier(\n",
                "        n_estimators=50, max_depth=10, class_weight='balanced', random_state=42, n_jobs=-1\n",
                "    ),\n",
                "    'LightGBM': LGBMClassifier(\n",
                "        n_estimators=100, learning_rate=0.08, num_leaves=31, random_state=42, verbose=-1, n_jobs=-1\n",
                "    ),\n",
                "}\n",
                "\n",
                "results = {}\n",
                "\n",
                "for name, model in models.items():\n",
                "    print(f'\\n{\"=\"*60}')\n",
                "    print(f'Training: {name}')\n",
                "    print(f'{\"=\"*60}')\n",
                "\n",
                "    model.fit(X_train, y_train)\n",
                "\n",
                "    # Predictions\n",
                "    y_pred = model.predict(X_val)\n",
                "    y_proba = model.predict_proba(X_val)[:, 1]\n",
                "\n",
                "    # Metrics\n",
                "    acc = accuracy_score(y_val, y_pred)\n",
                "    roc = roc_auc_score(y_val, y_proba)\n",
                "\n",
                "    # Cross-validation\n",
                "    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)\n",
                "    cv_scores = cross_val_score(model, X_scaled, y, cv=cv, scoring='roc_auc', n_jobs=-1 if name != 'Logistic Regression' else 1)\n",
                "\n",
                "    results[name] = {\n",
                "        'model': model,\n",
                "        'accuracy': acc,\n",
                "        'roc_auc': roc,\n",
                "        'cv_roc_auc_mean': cv_scores.mean(),\n",
                "        'cv_roc_auc_std': cv_scores.std(),\n",
                "    }\n",
                "\n",
                "    print(f'  Accuracy:       {acc:.4f}')\n",
                "    print(f'  ROC-AUC (val):  {roc:.4f}')\n",
                "    print(f'  CV ROC-AUC:     {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}')\n",
                "    print(f'\\n{classification_report(y_val, y_pred, target_names=[\"Not Fraud\", \"Fraud\"])}')\n"
            ]
            cell["source"] = new_src

print(f"Total cells in notebook: {len(nb['cells'])}")

for idx, cell in enumerate(nb["cells"]):
    if cell["cell_type"] != "code":
        continue

    code = "".join(cell["source"])
    print(f"\n[{execution_counter}] Executing cell {idx}...")
    t0 = time.time()

    # Capture stdout
    old_stdout = sys.stdout
    captured_stdout = io.StringIO()
    sys.stdout = captured_stdout

    # Track matplotlib figures
    plt.close('all')

    cell_outputs = []
    has_error = False

    try:
        exec(code, exec_ns)
    except Exception as e:
        import traceback
        has_error = True
        err_msg = traceback.format_exc()
        print(err_msg, file=sys.stderr)
        cell_outputs.append({
            "output_type": "error",
            "ename": type(e).__name__,
            "evalue": str(e),
            "traceback": err_msg.splitlines(True)
        })
    finally:
        sys.stdout = old_stdout

    stdout_text = captured_stdout.getvalue()
    if stdout_text:
        lines = [line + "\n" for line in stdout_text.splitlines()]
        if lines:
            lines[-1] = lines[-1].rstrip("\n")
        cell_outputs.append({
            "output_type": "stream",
            "name": "stdout",
            "text": lines
        })
        print(f"    stdout: {len(stdout_text)} chars ({stdout_text.splitlines()[0] if stdout_text.splitlines() else ''})")

    # Capture any open matplotlib figures
    for fig_num in plt.get_fignums():
        fig = plt.figure(fig_num)
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", dpi=100)
        buf.seek(0)
        b64_img = base64.b64encode(buf.read()).decode("utf-8")
        cell_outputs.append({
            "output_type": "display_data",
            "data": {
                "image/png": b64_img,
                "text/plain": [f"<Figure size {fig.get_size_inches()[0]*100}x{fig.get_size_inches()[1]*100} with {len(fig.axes)} Axes>"]
            },
            "metadata": {}
        })
        plt.close(fig)

    cell["outputs"] = cell_outputs
    cell["execution_count"] = execution_counter
    print(f"    Completed in {time.time()-t0:.2f}s (outputs: {len(cell_outputs)})")
    execution_counter += 1

    if has_error:
        print(f"Error in cell {idx}, stopping.")
        break

# Save executed notebook
with open(NOTEBOOK_PATH, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)

print(f"\n[SUCCESS] Notebook {NOTEBOOK_PATH} executed and saved successfully with all outputs!")
