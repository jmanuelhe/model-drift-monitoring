"""Dashboard HTML de monitoreo (Plotly vía CDN)."""
import plotly.graph_objects as go
from plotly.subplots import make_subplots

INK, CORAL, TEAL, SLATE, SAND = "#1B2430", "#E4572E", "#17A398", "#6B7A8F", "#F3EEE7"


def build(w, psi, s, path):
    fig = make_subplots(rows=3, cols=1, vertical_spacing=0.09, row_heights=[0.32, 0.28, 0.40],
                        subplot_titles=("ROC-AUC semanal", "Churn real vs. predicho (%)", "PSI por variable"))
    fig.add_trace(go.Scatter(x=w.week, y=w.auc_static, name="estático", line=dict(color=SLATE)), 1, 1)
    fig.add_trace(go.Scatter(x=w.week, y=w.auc_live, name="con reentrenamiento", line=dict(color=TEAL, width=3)), 1, 1)
    fig.add_trace(go.Scatter(x=w.week, y=w.churn_rate * 100, name="real", line=dict(color=INK)), 2, 1)
    fig.add_trace(go.Scatter(x=w.week, y=w.pred_rate_static * 100, name="pred. estático",
                             line=dict(color=SLATE, dash="dash")), 2, 1)
    fig.add_trace(go.Scatter(x=w.week, y=w.pred_rate_live * 100, name="pred. reentrenado",
                             line=dict(color=TEAL, dash="dash")), 2, 1)
    p = psi.pivot(index="feature", columns="week", values="psi")
    fig.add_trace(go.Heatmap(z=p.values, x=p.columns, y=p.index, colorscale="YlOrRd", zmin=0, zmax=0.4,
                             colorbar=dict(len=0.38, y=0.18, title="PSI"),
                             hovertemplate="%{y} · sem %{x}: PSI %{z:.3f}<extra></extra>"), 3, 1)
    for wk in w.loc[w.retrain, "week"]:
        fig.add_vline(x=wk + 0.5, line=dict(color=CORAL, dash="dot"), row=1, col=1)
    fig.update_layout(height=980, template="simple_white", font=dict(family="Inter, Arial", color=INK),
                      legend=dict(orientation="h", y=1.05))
    rows = "".join(f"<tr><td>Semana {r['week']}</td><td>{r['reason']}</td></tr>" for r in s["retrains"])
    alerts = w[(w.n_critical > 0) | (w.n_warning > 0)]
    alert_rows = "".join(f"<tr><td>{r.week}</td><td>{r.n_warning}</td><td>{r.n_critical}</td><td>{r.max_psi:.3f}</td></tr>"
                         for r in alerts.itertuples())
    html = f"""<!doctype html><html lang="es"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Monitoreo de modelo · Churn</title><script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>body{{font-family:Inter,Arial,sans-serif;background:{SAND};color:{INK};margin:0}}main{{max-width:1100px;margin:auto;padding:24px 16px}}
.ks{{display:flex;gap:12px;flex-wrap:wrap}}.k{{background:#fff;border-radius:10px;padding:14px 18px;flex:1;min-width:150px}}
.k b{{display:block;font-size:26px;color:{TEAL}}}.card{{background:#fff;border-radius:10px;padding:12px;margin-top:16px;overflow-x:auto}}
table{{border-collapse:collapse;width:100%}}td,th{{padding:6px 8px;border-bottom:1px solid #eee;text-align:left}}</style></head>
<body><main><h1>Monitoreo del modelo de churn · 16 semanas simuladas</h1>
<div class="ks"><div class="k"><b>{s['auc_static_weeks_9_16']:.3f}</b>AUC estático (sem 9–16)</div>
<div class="k"><b>{s['auc_live_weeks_9_16']:.3f}</b>AUC con reentrenamiento (sem 9–16)</div>
<div class="k"><b>{len(s['retrains'])}</b>reentrenamientos</div><div class="k"><b>sem {s['first_warning_week']}</b>primera alerta de drift</div></div>
<div class="card">{fig.to_html(full_html=False, include_plotlyjs=False)}</div>
<div class="card"><h2>Reentrenamientos</h2><table><tr><th>Cuándo</th><th>Motivo</th></tr>{rows}</table></div>
<div class="card"><h2>Alertas de drift</h2><table><tr><th>Semana</th><th>Variables en vigilancia</th><th>Variables críticas</th><th>PSI máximo</th></tr>{alert_rows}</table></div>
</main></body></html>"""
    path.write_text(html, encoding="utf-8")
