import pandas as pd


def export_point_table(point_tables: list, csv_path: str, excel_path: str):
    if len(point_tables) == 0:
        return

    df = pd.concat(point_tables, ignore_index=True)
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    df.to_excel(excel_path, index=False)

    print("Saved labeled point table:")
    print(f"  {csv_path}")
    print(f"  {excel_path}")

def export_tp_matches(rows: list, csv_path: str):
    if len(rows) == 0:
        print("No TP matches to export.")
        return

    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    print("Saved TP match table:")
    print(f"  {csv_path}")

def export_tp_matches_excel(rows: list, path: str):
    if len(rows) == 0:
        return

    df = pd.DataFrame(rows)
    df.to_excel(path, index=False)

    print("Saved TP match Excel:")
    print(path)
