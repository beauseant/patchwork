import os
import pathlib
import shutil

from matplotlib import pyplot as plt
import numpy as np
from src.TopicModeling.solr_backend_utils.utils import create_trainconfig
from src.Utils.utils import set_logger, train_test_split
import argparse
import yaml
from pathlib import Path
import pandas as pd
from gensim import corpora
import sys
import time
from src.TopicModeling import (
    BERTopicModel,
    GensimLDAModel,
    MalletLDAModel,
    NMFModel,
    TomotopyCTModel,
    TomotopyLDAModel,
)

########################
# AUXILIARY FUNCTIONS  #
########################
def create_model(model_name, **kwargs):
    # Map model names to corresponding classes
    model_mapping = {
        'BERTtopic': BERTopicModel,
        'Gensim': GensimLDAModel,
        'Mallet': MalletLDAModel,
        'NMF': NMFModel,
        'TomotopyCTModel': TomotopyCTModel,
        'TomotopyLDAModel': TomotopyLDAModel,
    }

    # Retrieve the class based on the model name
    model_class = model_mapping.get(model_name)

    # Check if the model name is valid
    if model_class is None:
        raise ValueError(f"Invalid model name: {model_name}")

    # Create an instance of the model class
    model_instance = model_class(**kwargs)

    return model_instance


if __name__ == "__main__":

    # Set logger
    logger = set_logger(console_log=True, file_log=True)

    # Parse args
    parser = argparse.ArgumentParser(
        description="Train options for topic modeling")
    parser.add_argument(
        "--options",
        default="config/options_env.yaml",
        help="Path to options YAML file"
    )
    parser.add_argument(
        "--trainer",
        default="Mallet",
        help="Trainer to use for topic modeling"
    )
    parser.add_argument(
        "--test_split",
        default=0.0,
        type=float,
        help="Size of test split for training the model. If set to 0.0, the model will be trained on the entire dataset and no test set will be created."
    )
    parser.add_argument(
        "--data_path",
        default="/export/usuarios_ml4ds/lbartolome/NextProcurement/NP-Text_Object/data/train_data/training_dfs_per_cpv",
        help="Path to data"
    )
    parser.add_argument(
        "--num_topics",
        default="5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,25,30,35,40,45,50",
        help="Number of topics to train the model on"
    )
    parser.add_argument(
        "--path_stops",
        default="/export/usuarios_ml4ds/lbartolome/NextProcurement/NP-Search-Tool/sample_data/stopwords",
        help="Path to stopwords file"
    )
    parser.add_argument(
        "--path_eq",
        default="/export/usuarios_ml4ds/lbartolome/NextProcurement/NP-Search-Tool/sample_data/equivalences/manual_equivalences.txt",
        help="Path to equivalence dictionary file"
    )

    parser.add_argument(
        "--extra",
        type=bool,
        default=False,
        help="Extra parameter for the model"
    )
    parser.add_argument(
        "--path_data",
        default="/export/usuarios_ml4ds/lbartolome/NextProcurement/NP-Text_Object/data/train_data/processed_objectives_with_lemmas_emb_label.parquet",
        help="Path to data"
    )
    
    parser.add_argument(
        "--final_models",
        default="/export/usuarios_ml4ds/lbartolome/NextProcurement/NP-Search-Tool/sample_data/zaragoza_models_more_stops",
        help="Path to final models"
    )
    
    parser.add_argument(
        "--corpus_seggregate",
        default="zaragoza",
    )
    
    args = parser.parse_args()

    with open(args.options, "r") as f:
        options = dict(yaml.safe_load(f))

    # Set logger
    dir_logger = Path(options.get("dir_logger", "app.log"))
    console_log = options.get("console_log", True)
    file_log = options.get("file_log", True)
    logger = set_logger(
        console_log=console_log,
        file_log=file_log,
        file_loc=dir_logger
    )

    ##########
    # Config #
    ##########
    # Number of topics for training the model and word min len
    num_topics = options.get('training_params', {}).get('num_topics', '50')
    try:
        num_topics = [int(k) for k in args.num_topics.split(",")]
    except:
        num_topics = [int(num_topics)]
    word_min_len = options.get('training_params', {}).get('word_min_len', 4)
    min_lemmas = 3#options.get('training_params', {}).get('min_lemmas', 5)
    
    # File directories
    dir_output_models = Path(args.final_models)
    dir_mallet = Path(options.get("dir_mallet"))
    
    ##########
    # Train  #
    ##########
    # Get training parameters
    tr_params = options.get('training_params', {}).get(
        f"{args.trainer}_params", {})
    if not bool(tr_params):
        logger.error(
            "--- Training parameters not found in options.yaml. Quitting script....")
        sys.exit(1)
    model_init_params = {
        "word_min_len": word_min_len,
        "logger": logger,
    }
    if args.trainer == "Mallet":
        model_init_params["mallet_path"] = dir_mallet

    
    #####################
    # Load stops /eqs #
    #####################
    print(f"Loading stopwords")
    start_time = time.time()
    stopwords = set()
    for file in os.listdir(args.path_stops):
        if args.corpus_seggregate == "cpv":
            condition = file.endswith('.txt')
        else:
            condition = file.endswith('.txt') and not "stops_cpv_final" in file
        if condition:
            with open(os.path.join(args.path_stops, file), 'r', encoding='utf-8') as f:
                stopwords.update(f.read().splitlines())            
    print(f"-- -- {len(stopwords)} stopwords load in {time.time() - start_time:.2f} seconds")

    print(f"Loading equivalences")
    start_time = time.time()
    equivalents = {}
    with open(args.path_eq, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            words = line.split(':')
            if len(words) < 2:
                print(f"Skipped or incomplete line: '{line}'")
                continue
            equivalents[words[0]] = words[1]
    print(f"-- -- {len(equivalents)} equivalence loaded in: {time.time() - start_time:.2f} seconds")

    ##########################################
    # Carry out specific preprocessing steps #
    ##########################################
    def tkz_clean_str(rawtext, stopwords, equivalents):
        if not rawtext:
            return ''
        
        # Lowercasing and tokenization
        cleantext = rawtext.lower().split()

        # Remove stopwords and apply equivalences in one pass
        cleantext = [equivalents.get(word, word) for word in cleantext if word not in stopwords]

        # Second stopword removal (in case equivalences introduced new stopwords)
        cleantext = [word for word in cleantext if word not in stopwords]
        
        return ' '.join(cleantext)

    def preprocBOW(data_col, min_lemas=15, no_below=10, no_above=0.6, keep_n=100000):

        # filter out documents (rows) with less than minimum number of lemmas
        data_col = data_col[data_col.apply(lambda x: len(x.split())) >= min_lemas]
        
        final_tokens = [doc.split() for doc in data_col.values.tolist()]
        gensimDict = corpora.Dictionary(final_tokens)

        # Remove words that appear in less than no_below documents, or in more than no_above, and keep at most keep_n most frequent terms
        gensimDict.filter_extremes(no_below=no_below, no_above=no_above, keep_n=keep_n)
        
        # Remove words not in dictionary, and return a string
        vocabulary = set([gensimDict[idx] for idx in range(len(gensimDict))])
        
        return vocabulary
    
    if args.corpus_seggregate == "cpv":    
        cpv_files = os.listdir(args.data_path)
        # the files are defined as {id}_{cpv}.parquet
        # sort the files by id from 0 to n
        cpv_files = sorted(cpv_files, key=lambda x: int(x.split("_")[0]))
        cpv_files = cpv_files[:-1]
        print(f"CPV files: {cpv_files}")
        # extract the cpv from the file name
        cpvs = [cpv.split(".")[0] for cpv in cpv_files]
        cpvs = cpvs[:-1]
        print(f"CPVs: {cpvs}")
        for cpv_data, cpv in zip(cpv_files, cpvs):
            
            df = pd.read_parquet(os.path.join(args.data_path, cpv_data))
            # drop duplicates based on place_id
            df = df.drop_duplicates(subset=['place_id'])
            data_path = Path(cpv_data).name
            print(f"Processing data for CPV: {data_path}")
            print(f"Data shape: {df.shape}")
            print(f"Average lemmas per document (before processing): {df['lemmas'].apply(lambda x: len(x.split())).mean()}")
            
            # Clean text
            start_time = time.time()
            df['lemmas'] = df['lemmas'].apply(lambda x: tkz_clean_str(x, stopwords, equivalents))
            print(f"-- -- Text cleaned in {time.time() - start_time:.2f} seconds")
            # create vocabulary
            start_time = time.time()
            vocabulary = preprocBOW(df['lemmas'], min_lemas=min_lemmas)
            print(f"-- -- Vocabulary created in {time.time() - start_time:.2f} seconds")
            # remove words that are not in the vocabulary
            df['lemmas'] = df['lemmas'].apply(lambda x: ' '.join([word for word in x.split() if word in vocabulary]))
            # remove rows with less than min_lemmas lemmas
            df = df[df['lemmas'].apply(lambda x: len(x.split())) >= min_lemmas]
            print(f"Data shape after filtering: {df.shape}")
            print(f"Average lemmas per document (after processing): {df['lemmas'].apply(lambda x: len(x.split())).mean()}")
            
            # Train models
            cohrs = []
            for k in num_topics:
                logger.info(f"{'='*10} Training model with {k} topics. {'='*10}")

                texts_train, texts_test = train_test_split(df, args.test_split)
                logger.info("Data loaded.")
                logger.info(
                    f"Train: {len(texts_train)} docs. Test: {len(texts_test)}.")

                # Update model parameters            
                model_path = dir_output_models / cpv
                model_path.mkdir(parents=True, exist_ok=True)
                model_path = model_path / f"{k}_topics"
                model_init_params["model_dir"] = model_path

                # Create model
                model = create_model(args.trainer, **model_init_params)
                
                start = time.time()

                path_data = args.path_data if args.extra else None
                
                #import pdb; pdb.set_trace()
                # Train model
                model.train(
                    texts_train.lemmas,
                    texts_train.place_id,
                    num_topics=k,
                    **tr_params,
                    texts_test=texts_test.lemmas,
                    ids_test=texts_test.place_id, 
                    extra=args.extra,
                    path_data=path_data
                )

                model.save_model(
                    path=model_init_params["model_dir"] /
                    "model_data" / "model.pickle"
                )
                
                print("-- -- Model trained in:", time.time() - start)

                print(f"-- -- Model saved in {model_init_params['model_dir']}")
                
                create_trainconfig(
                    modeldir=model_init_params["model_dir"],
                    model_name=pathlib.Path(model_init_params["model_dir"]).name,
                    model_desc=pathlib.Path(model_init_params["model_dir"]).name,
                    trainer=args.trainer,
                    TrDtSet=data_path,
                    TMparam=tr_params
                )
                
                cohr = np.load(model_init_params["model_dir"] / "model_data" / "TMmodel/topic_coherence.npy").mean()
                cohrs.append({
                    "num_topics": k,
                    "coherence": cohr
                })
                
            df_cohrs = pd.DataFrame(cohrs)
            # Plot coherence graph
            plt.figure(figsize=(10, 6))
            plt.plot(df_cohrs['num_topics'], df_cohrs['coherence'], marker='o', linestyle='-')
            plt.xlabel("Number of Topics")
            plt.ylabel("Coherence Score")
            plt.title("Coherence Score vs. Number of Topics")
            plt.grid(True)
            plot_path = dir_output_models / cpv / "coherence_plot.png"        
            plt.savefig(plot_path)
            plt.show()
            
            # find the first local maximum in the first half of the graph and the second maximum in the second half
            half = len(df_cohrs) // 2
            df_cohrs_first_half = df_cohrs.iloc[:half]
            df_cohrs_second_half = df_cohrs.iloc[half:]
            
            max_cohr_idx = df_cohrs_first_half['coherence'].idxmax()
            max_cohr = df_cohrs_first_half.loc[max_cohr_idx]
            print(f"First best coherence: {max_cohr['coherence']} with {max_cohr['num_topics']} topics")
            # copy the best model to the best_models directory
            final_model_path = Path(args.final_models) / cpv / f"{int(max_cohr['num_topics'])}_topics"
            final_model_path.mkdir(parents=True, exist_ok=True)
            # copy model_path of best model to final_model_path
            best_model_path = dir_output_models / cpv / f"{int(max_cohr['num_topics'])}_topics"
            shutil.copytree(best_model_path, final_model_path, dirs_exist_ok=True)
        
            # second maximum
            max_cohr_idx = df_cohrs_second_half['coherence'].idxmax()
            max_cohr = df_cohrs_second_half.loc[max_cohr_idx]
            print(f"Second best coherence: {max_cohr['coherence']} with {max_cohr['num_topics']} topics")
            # copy the best model to the best_models directory
            final_model_path = Path(args.final_models) / cpv / f"{int(max_cohr['num_topics'])}_topics"
            final_model_path.mkdir(parents=True, exist_ok=True)
            # copy model_path of best model to final_model_path
            best_model_path = dir_output_models / cpv / f"{int(max_cohr['num_topics'])}_topics"
            shutil.copytree(best_model_path, final_model_path, dirs_exist_ok=True)
    else:
        
        df = pd.read_parquet(args.data_path)
        
        # drop duplicates based on place_id
        df = df.drop_duplicates(subset=['place_id'])
        print(f"Data shape: {df.shape}")
        print(f"Average lemmas per document (before processing): {df['lemmas'].apply(lambda x: len(x.split())).mean()}")
        
        # Clean text
        start_time = time.time()
        df['lemmas'] = df['lemmas'].apply(lambda x: tkz_clean_str(x, stopwords, equivalents))
        print(f"-- -- Text cleaned in {time.time() - start_time:.2f} seconds")
        # create vocabulary
        start_time = time.time()
        vocabulary = preprocBOW(df['lemmas'], min_lemas=min_lemmas)
        print(f"-- -- Vocabulary created in {time.time() - start_time:.2f} seconds")
        # remove words that are not in the vocabulary
        df['lemmas'] = df['lemmas'].apply(lambda x: ' '.join([word for word in x.split() if word in vocabulary]))
        # remove rows with less than min_lemmas lemmas
        df = df[df['lemmas'].apply(lambda x: len(x.split())) >= min_lemmas]
        print(f"Data shape after filtering: {df.shape}")
        print(f"Average lemmas per document (after processing): {df['lemmas'].apply(lambda x: len(x.split())).mean()}")
        
        # check that there is no empty lemmas
        print(f"Empty lemmas: {df['lemmas'].apply(lambda x: len(x)).sum() == 0}")
        
        # Train models
        cohrs = []
        for k in num_topics:
            logger.info(f"{'='*10} Training model with {k} topics. {'='*10}")

            texts_train, texts_test = train_test_split(df, args.test_split)
            logger.info("Data loaded.")
            logger.info(
                f"Train: {len(texts_train)} docs. Test: {len(texts_test)}.")

            # Update model parameters            
            model_path = dir_output_models
            model_path.mkdir(parents=True, exist_ok=True)
            model_path = model_path / f"{k}_topics"
            model_init_params["model_dir"] = model_path

            # Create model
            model = create_model(args.trainer, **model_init_params)
            
            start = time.time()

            path_data = args.path_data if args.extra else None
                
            # Train model
            model.train(
                texts_train.lemmas,
                texts_train.place_id,
                num_topics=k,
                **tr_params,
                texts_test=texts_test.lemmas,
                ids_test=texts_test.place_id, 
                extra=args.extra,
                path_data=path_data
            )

            model.save_model(
                path=model_init_params["model_dir"] /
                "model_data" / "model.pickle"
            )
            
            print("-- -- Model trained in:", time.time() - start)

            print(f"-- -- Model saved in {model_init_params['model_dir']}")
            
            create_trainconfig(
                modeldir=model_init_params["model_dir"],
                model_name=pathlib.Path(model_init_params["model_dir"]).name,
                model_desc=pathlib.Path(model_init_params["model_dir"]).name,
                trainer=args.trainer,
                TrDtSet=args.data_path,
                TMparam=tr_params
            )
            
            cohr = np.load(model_init_params["model_dir"] / "model_data" / "TMmodel/topic_coherence.npy").mean()
            cohrs.append({
                "num_topics": k,
                "coherence": cohr
            })
            
        df_cohrs = pd.DataFrame(cohrs)
        # Plot coherence graph
        plt.figure(figsize=(10, 6))
        plt.plot(df_cohrs['num_topics'], df_cohrs['coherence'], marker='o', linestyle='-')
        plt.xlabel("Number of Topics")
        plt.ylabel("Coherence Score")
        plt.title("Coherence Score vs. Number of Topics")
        plt.grid(True)
        plot_path = dir_output_models / "coherence_plot.png"        
        plt.savefig(plot_path)
        plt.show()