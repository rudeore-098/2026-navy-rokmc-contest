"""앙상블 — 여러 실험의 OOF를 모아 가중치를 최적화하고 블렌딩한다.

OOF가 앙상블의 핵심 입력인 이유: 각 실험의 oof.npy는 '누수 없는' 검증 예측이라,
이걸로 가중치를 정하면 test에 일반화되는 비율을 찾을 수 있다.

사용법:
    python -m src.ensemble --task binary --exps exp_001 exp_002 exp_003
"""
import os
import argparse
import numpy as np

from src.utils.metrics import get_metric, metric_name


def optimize_weights(task, oofs, y_true, n_trials=300, seed=42):
    """Optuna로 OOF 점수를 최대화하는 가중치를 탐색. (가벼워서 많이 돌려도 됨)"""
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial):
        w = np.array([trial.suggest_float(f"w{i}", 0.0, 1.0) for i in range(len(oofs))])
        if w.sum() == 0:
            return -1e9
        w = w / w.sum()
        blend = sum(wi * o for wi, o in zip(w, oofs))
        return get_metric(task, y_true, blend)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    w = np.array([study.best_params[f"w{i}"] for i in range(len(oofs))])
    return w / w.sum(), study.best_value


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=["binary", "multiclass", "regression"])
    ap.add_argument("--exps", nargs="+", required=True, help="앙상블할 실험 폴더들")
    ap.add_argument("--n-trials", type=int, default=300)
    args = ap.parse_args()

    oofs, names = [], []
    y_true = None
    for exp in args.exps:
        d = os.path.join("experiments", exp)
        oof_path = os.path.join(d, "oof.npy")
        if not os.path.exists(oof_path):
            print(f"건너뜀: {oof_path} 없음"); continue
        oofs.append(np.load(oof_path)); names.append(exp)
        if y_true is None:
            y_true = np.load(os.path.join(d, "y_true.npy"))

    if len(oofs) < 2:
        print("앙상블하려면 OOF가 2개 이상 필요합니다."); return

    # 단순 평균 vs 가중 평균 비교
    simple = np.mean(oofs, axis=0)
    simple_score = get_metric(args.task, y_true, simple)

    w, weighted_score = optimize_weights(args.task, oofs, y_true, args.n_trials)

    print(f"=== 앙상블 결과 ({metric_name(args.task)}) ===")
    for n, wi in zip(names, w):
        print(f"  {n}: weight={wi:.3f}")
    print(f"단순 평균: {simple_score:.5f}")
    print(f"가중 평균: {weighted_score:.5f}")

    if weighted_score >= simple_score:
        print(f"\n👉 가중 평균 채택 (+{weighted_score - simple_score:.5f})")
        print(f"   infer에 넘길 가중치: {','.join(f'{x:.3f}' for x in w)}")
    else:
        print("\n👉 단순 평균 채택 (과적합 방지로 더 안전)")


if __name__ == "__main__":
    main()
