import numpy as np
import datatable as dt
from datetime import datetime
from datatable import f, join, sort
import sys
import os
from pathlib import Path
import argparse

def main():
    parser = argparse.ArgumentParser(description="Format raw AML Kaggle CSV → canonical transactions CSV")
    parser.add_argument("inpath", help="Path to raw CSV (e.g., data/raw/HI-Small_Trans.csv)")
    parser.add_argument("--outdir", default="data/interim", help="Output directory (default: data/interim)")
    args = parser.parse_args()

    inPath = args.inpath
    outDir = Path(args.outdir)
    outDir.mkdir(parents=True, exist_ok=True)

    # Output filename: <input_stem>_formatted.csv
    base = Path(inPath).stem  # e.g., HI-Small_Trans
    outPath = outDir / f"{base}_formatted.csv"

    # loading raw data
    raw = dt.fread(inPath, columns=dt.str32)

    # creating dictionaries for categorical encoding
    currency = {}
    paymentFormat = {}
    account = {}

    def get_dict_val(name, collection):
        if name in collection:
            val = collection[name]
        else:
            val = len(collection)
            collection[name] = val
        return val

    header = ("EdgeID,from_id,to_id,Timestamp,"
              "Amount Sent,Sent Currency,Amount Received,Received Currency,"
              "Payment Format,Is Laundering\n")

    firstTs = -1

    with open(outPath, "w") as writer:
        writer.write(header)
        for i in range(raw.nrows):
            # Expecting format YYYY/MM/DD HH:MM
            dt_obj = datetime.strptime(raw[i, "Timestamp"], "%Y/%m/%d %H:%M")
            ts = dt_obj.timestamp()

            if firstTs == -1:
                startTime = datetime(dt_obj.year, dt_obj.month, dt_obj.day)
                firstTs = startTime.timestamp() - 10
            ts = ts - firstTs

            cur_recv = get_dict_val(raw[i, "Receiving Currency"], currency)
            cur_pay = get_dict_val(raw[i, "Payment Currency"], currency)
            fmt = get_dict_val(raw[i, "Payment Format"], paymentFormat)

            # Build account IDs (bank + account number columns)
            fromAccIdStr = raw[i, "From Bank"] + raw[i, 2]   # 3rd column = sender acct id in original CSVs
            toAccIdStr   = raw[i, "To Bank"]   + raw[i, 4]   # 5th column = receiver acct id

            fromId = get_dict_val(fromAccIdStr, account)
            toId   = get_dict_val(toAccIdStr, account)

            amountReceivedOrig = float(raw[i, "Amount Received"])
            amountPaidOrig     = float(raw[i, "Amount Paid"])
            isl = int(raw[i, "Is Laundering"])

            line = "%d,%d,%d,%d,%f,%d,%f,%d,%d,%d\n" % (
                i, fromId, toId, ts,
                amountPaidOrig, cur_pay,
                amountReceivedOrig, cur_recv,
                fmt, isl
            )
            writer.write(line)

    # Sort by Timestamp (4th column, index 3) and overwrite
    formatted = dt.fread(str(outPath))
    formatted = formatted[:, :, sort(3)]
    formatted.to_csv(str(outPath))

    print(f"Wrote: {outPath}  (rows={formatted.nrows})")

if __name__ == "__main__":
    if len(sys.argv) == 1:
        print("Usage: python src/data_prep/format_data.py data/raw/HI-Small_Trans.csv [--outdir data/interim]")
        sys.exit(1)
    main()
