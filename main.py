    # ---------- calibration ----------
    from sklearn.model_selection import train_test_split
    from src.calibration.calibrate import ConformalCalibrator, PlattCalibrator, ece
    sc, lb = np.array(res["scores"]), np.array(res["labels"])

    conf = ConformalCalibrator().fit(val["scores"])
    thr05 = conf.threshold(0.05)
    print(f"conformal thr (alpha=0.05): {thr05:.3f} | "
          f"FPR on test normals: {(sc[lb == 0] > thr05).mean():.3f} | "
          f"TPR: {(sc[lb == 1] > thr05).mean():.3f}")

    cal_i, ev_i = train_test_split(np.arange(len(lb)), test_size=0.5, stratify=lb, random_state=0)
    platt = PlattCalibrator().fit(sc[cal_i], lb[cal_i])
    print("ECE (Platt, held-out half):", round(ece(platt.predict_proba(sc[ev_i]), lb[ev_i]), 4))

    # ---------- distribution shift ----------
    from src.shift.drift_detector import ShiftDetector
    from src.data.shift_simulator import shift_batch
    ref = torch.stack([full[i]["image"] for i in train_set.indices[:150]])
    det = ShiftDetector().fit(ref, pc)
    good = res["images"][torch.tensor(lb) == 0]
    print("clean       :", det.detect(good, pc)["summary"])
    for kind in ["brightness", "contrast", "blur", "noise"]:
        print(f"{kind:<12}:", det.detect(shift_batch(good, kind, 0.5), pc)["summary"])

    # ---------- false-positive management ----------
    from src.false_positive.fp_manager import FalsePositiveManager
    fp = FalsePositiveManager(conf)
    decisions = [fp.submit(f"test_{i}", s) for i, s in enumerate(sc)]
    for name in ["pass", "review", "defect"]:
        sel = np.array([d == name for d in decisions])
        print(f"{name:<7} normals={int((sel & (lb == 0)).sum()):>3} defects={int((sel & (lb == 1)).sum()):>3}")