# meridian_helper.py
# Thin wrapper around Google Meridian. Safe to import even if Meridian is not installed.

def fit_meridian_and_summarize(df, schema=None, priors=None, time_params=None, holdout_frac=0.2, outdir="./reports"):
    try:
        import meridian
        from meridian import api as mdapi
        from meridian.analysis import analyzer
    except Exception as e:
        return {"ok": False, "reason": f"Meridian unavailable: {e}"}

    if schema is None:
        channels = [c.replace("_spend","") for c in df.columns if c.endswith("_spend")]
        schema = {
            "date": "date",
            "kpi": "subscriptions",
            "paid_media": [
                {"channel": ch, "spend": f"{ch}_spend", "impressions": f"{ch}_impressions" if f"{ch}_impressions" in df.columns else None}
                for ch in channels
            ],
            "controls": [],
            "time_granularity": "weekly"
        }
    if priors is None:
        priors = {"treatment_prior_type": "ROI", "custom_priors": None}
    if time_params is None:
        time_params = {
            "adstock": {"type":"geometric","decay_prior_mean":0.5,"decay_prior_sd":0.2,"max_lag":8},
            "saturation": {"type":"hill","half_saturation_prior_mean":0.5,"alpha_prior_mean":1.0},
        }

    try:
        model = mdapi.build_model(data=df, schema=schema, priors=priors, time_params=time_params, holdout=holdout_frac)
        results = mdapi.fit_model(model, draws=1500, chains=2, target_accept=0.9)
        mdapi.save_results(results, outdir)
        analyzer.generate_two_pager(outdir, out_dir=outdir)
        analyzer.run_budget_optimization(outdir, out_dir=outdir)
        return {"ok": True, "reports_dir": outdir}
    except Exception as e:
        return {"ok": False, "reason": f"Meridian error: {e}"}
