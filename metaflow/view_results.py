"""
Xem kết quả tất cả runs — Metaflow Client API

Chạy:
  python3 view_results.py
"""
import os
os.environ["METAFLOW_DEFAULT_DATASTORE"] = "local"
os.environ["METAFLOW_DEFAULT_METADATA"] = "local"

from metaflow import Flow
import pandas as pd


def view_uc1():
    """Xem kết quả UC1 MNIST"""
    print("\n" + "=" * 60)
    print("  UC1 MNIST — Metaflow Results")
    print("=" * 60)

    try:
        flow = Flow('MNISTFlow')
        results = []
        for run in flow.runs():
            try:
                results.append({
                    "run_id": run.id,
                    "model": run.data.model_name,
                    "lr": run.data.lr,
                    "batch_size": run.data.batch_size,
                    "accuracy": round(run.data.accuracy, 4),
                    "train_time_s": round(run.data.train_time, 1),
                    "eval_time_s": round(run.data.eval_time, 1),
                    "pipeline_time_s": round(run.data.pipeline_time, 1),
                })
            except:
                pass

        if results:
            df = pd.DataFrame(results)
            print(df.to_string(index=False))
            df.to_csv("metaflow_uc1_results.csv", index=False)
            print("\n Saved: metaflow_uc1_results.csv")

            for model in df['model'].unique():
                subset = df[df['model'] == model]
                print(f"\n  {model} (n={len(subset)}): "
                      f"acc_TB={subset['accuracy'].mean():.4f}, "
                      f"TC7_TB={subset['pipeline_time_s'].mean():.1f}s")
        else:
            print("  Chưa có runs. Chạy train_uc1_metaflow.py trước.")
    except Exception as e:
        print(f"  Lỗi: {e}")
        print("  Chưa có flow MNISTFlow. Chạy UC1 trước.")


def view_uc2():
    """Xem kết quả UC2 PhoBERT"""
    print("\n" + "=" * 60)
    print("  UC2 PhoBERT — Metaflow Results")
    print("=" * 60)

    try:
        flow = Flow('PhoBERTSentimentFlow')
        results = []
        for run in flow.runs():
            try:
                results.append({
                    "run_id": run.id,
                    "lr": run.data.lr,
                    "batch_size": run.data.batch_size,
                    "accuracy": round(run.data.accuracy, 4),
                    "f1_macro": round(run.data.f1_macro, 4),
                    "train_time_s": round(run.data.train_time, 1),
                    "eval_time_s": round(run.data.eval_time, 1),
                    "pipeline_time_s": round(run.data.pipeline_time, 1),
                })
            except:
                pass

        if results:
            df = pd.DataFrame(results)
            print(df.to_string(index=False))
            df.to_csv("metaflow_uc2_results.csv", index=False)
            print("\n Saved: metaflow_uc2_results.csv")

            print(f"\n  TB (n={len(df)}): "
                  f"acc={df['accuracy'].mean():.4f}, "
                  f"f1_macro={df['f1_macro'].mean():.4f}, "
                  f"TC7={df['pipeline_time_s'].mean():.1f}s")
        else:
            print("  Chưa có runs. Chạy train_uc2_metaflow.py trước.")
    except Exception as e:
        print(f"  Lỗi: {e}")
        print("  Chưa có flow PhoBERTSentimentFlow. Chạy UC2 trước.")


if __name__ == '__main__':
    view_uc1()
    view_uc2()
