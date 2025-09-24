import json
import datetime as DT
import pathlib


def create_logical_corpus(
    path_datasets: pathlib.Path,
    dtsName: str,
    dtsDesc: str,
    Dtset_loc: str,
    Dtset_source: str,
    Dtset_idfld: str,
    Dtset_lemmas_fld: str,
    Preproc : dict,
    privacy: str = "public",
    Dtset_filter: str = ""
):
    """
    Creates a logical corpus (subset of a raw corpus used for training a topic model, stored as a JSON file named after the entity). Logical corpora are defined according to the specifications provided for the Interactive Model Trainer (IMT). 

    Example:

    "name": "Cordis",
    "description": "Cordis training dataset for TM",
    "valid_for": "TM",
    "visibility": "private",
    "Dtsets": [
        {
            "parquet":"/data/source/CORDIS.parquet",
            "source": "CORDIS",
            "idfld": "id",
            "lemmasfld": [
                "rawtext"
            ],
            "filter": ""
        }
    ],
    "creation_date": "2022-07-12 15:15:24.070089"

    Assumtions
    ----------
    Even though Dtsets is structured as a list, in this context, we have assumed that the logical corpus consists of only one raw corpus. However, if outsiders, insiders, and minors are modeled as a single dataset, we will consider the joint combination of the three as the data source, rather than treating them as separate individual files.

    Parameters
    ----------
    path_datasets: pathlib.Path
        Path where the logical corpora are saved.
    dtsName: str
        Name of the logical corpus.
    dtsDesc: str
        Description of the logical corpus.
    privacy: str
        Privacy level of the logical corpus, defaults to 'public'.
    Dtset_loc: str
        Path to the raw corpus (parquet file).
    Dtset_source: str
        Source of the logical corpus.
    Dtset_idfld: str
        Field in the datafame that serves as identifier.
    Dtset_lemmas_fld: str
        Field in the dataframe that serves as lemmas.
    Preproc : dict
        Preprocessing information.
    Dtset_filter: str
        Specific filtering options for the creation of the dataset, defaults to "".

    Returns
    -------
    str: Path to the just created logical corpus.
    """

    TM_Dtset = [
        {
            'parquet': Dtset_loc,
            'source': Dtset_source,
            'idfld': Dtset_idfld,
            'lemmasfld': Dtset_lemmas_fld,
            'filter': Dtset_filter
        }
    ]

    Dtset = {
        'name': dtsName,
        'description': dtsDesc,
        'valid_for': "TM",
        'visibility': privacy,
        'Dtsets': TM_Dtset,
        'Preproc': Preproc,
        'creation_date': DT.datetime.now()
    }

    path_Dtset = path_datasets / (Dtset['name'] + '.json')
    path_Dtset.parent.mkdir(parents=True, exist_ok=True)
    with path_Dtset.open('w', encoding='utf-8') as fout:
        json.dump(Dtset, fout, ensure_ascii=False,
                  indent=2, default=str)

    return path_Dtset.as_posix()


def create_trainconfig(
    modeldir: str,
    model_name: str,
    model_desc: str,
    trainer: str,
    TrDtSet: str,
    visibility: str = 'public',
    **kwargs

):
    """
    Creates a JSON file storing all the information describing a trained topic model. 

    Example:

    {
        "name": "Mallet-25",
        "description": "Mallet-25",
        "visibility": "Private",
        "trainer": "mallet",
        "TestSet": "/data/source/datasets/Cordis.json",
        "Preproc": {
            "min_lemas": 15,
            "no_below": 10,
            "no_above": 0.6,
            "keep_n": 500000,
            "stopwords": [],
            "equivalences": []
        },
        "TMparam": {
            "mallet_path": "/Users/topicmodeler/mallet-2.0.8/bin/mallet",
            "ntopics": 25,
            "alpha": 5.0,
            "optimize_interval": 10,
            "num_threads": 4,
            "num_iterations": 1000,
            "doc_topic_thr": 0.0,
            "thetas_thr": 0.003,
            "token_regexp": "[\\p{L}\\p{N}][\\p{L}\\p{N}\\p{P}]*\\p{L}"
        },
        "creation_date": "2022-07-20 10:48:11.125048",
    }

    Parameters
    ----------
    modeldir: str 
        Directory where the model will be stored.
    model_name: str
        Name of the model.
    model_desc: str
        Description of the model.
    trainer: str
        Trainer used for the model.
    TrDtSet: str
        Data set used for training.
    visibility: str
        Visibility of the model, defaults to 'public'.
    kwargs: dict
        Additional parameters for preprocessing and topic model.
    """

    configFile = pathlib.Path(modeldir) / 'trainconfig.json'
    
    train_config = {
        "name": model_name,
        "description": model_desc,
        "visibility": visibility,
        "creator": "NP",
        "trainer": trainer,
        "TrDtSet": TrDtSet,
        "Preproc": kwargs.get('Preproc', {}),
        "TMparam": kwargs.get('TMparam', {}),
        "creation_date": str(DT.datetime.now()),
    }
    
    configFile.parent.mkdir(parents=True, exist_ok=True)
    with configFile.open('w', encoding='utf-8') as outfile:
        json.dump(train_config, outfile,
                  ensure_ascii=False, indent=2, default=str)

    pass
