import argparse
import json
import pandas as pd # type: ignore
from src.objective_extractor import ObjectiveExtractor

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--document', type=str, help='Document (raw text) to extract the objective from.', required=True)
    parser.add_argument('--token_starts', type=str, help='Token starts.', required=False, default=[0, 1000, 2000, 3000, 4000, 5000])
    #token_starts=[0, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]
    args = parser.parse_args()

    df = pd.DataFrame([{"text": args.document}])
    
    extractor = ObjectiveExtractor(do_train=False, trained_promt="/data/source/ObjectiveExtractor-saved.json")
    df = extractor.predict(df, col_extract="text", token_starts=args.token_starts, checkpoint_path="checkpoint.pkl")
    
    # delete checkpoint file
    import os
    path_remove = "checkpoint.pkl"
    os.remove(path_remove)

    print(json.dumps({
        "objective": df.to_dict(orient="records")[0]["objective"],
        "in_text_score": df.to_dict(orient="records")[0]["in_text_score"]
    }))
    
    return


if __name__ == "__main__":
    main()