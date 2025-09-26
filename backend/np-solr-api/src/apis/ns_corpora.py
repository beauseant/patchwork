"""
This script defines a Flask RESTful namespace for managing corpora stored in Solr as collections. 

Author: Lorena Calvo-Bartolomé
Date: 27/03/2023
"""
from flask_restx import Namespace, Resource, fields, reqparse
from src.core.clients.np_solr_client import NPSolrClient

# ======================================================
# Define namespace for managing corpora
# ======================================================
api = Namespace(
    'Corpora', description='Corpora-related operations (i.e., index/delete corpora))')

# ======================================================
# Namespace variables
# ======================================================
# Create Solr client
sc = NPSolrClient(api.logger)

# Define parser to take inputs from user
parser = reqparse.RequestParser()
parser.add_argument(
    'corpus_name', help="Name of the corpus to index/delete. For example, if the corpus we want to index is stored in a file name 'corpus.parquet', the corpus_name should be 'corpus' (without the extension quotes). Do not use quotes in the name of the corpus.")

parser2 = reqparse.RequestParser()
parser2.add_argument(
    'corpus_col', help="Name of the corpus collection to list its models")

parser3 = reqparse.RequestParser()
parser3.add_argument(
    'corpus_col', help="Name of the corpus collection to list its MetadataDisplayed fields")

parser4 = reqparse.RequestParser()
parser4.add_argument(
    'corpus_col', help="Name of the corpus collection to list its SearcheableField fields")

parser5 = reqparse.RequestParser()
parser5.add_argument(
    'corpus_col', help="Name of the corpus collection")
parser5.add_argument(
    'searchable_fields', help="Fields to be added as searchable fields of the corpus collection, separated by commas")

parser6 = reqparse.RequestParser()
parser6.add_argument(
    'corpus_col', help="Name of the corpus collection")
parser6.add_argument(
    'searchable_fields', help="Fields to be added as searchable fields of the corpus collection, separated by commas")


@api.route('/indexCorpus/')
class IndexCorpus(Resource):
    @api.doc(parser=parser)
    def post(self):
        args = parser.parse_args()
        corpus_name = args['corpus_name']
        try:
            sc.index_corpus(corpus_name)
            return '', 200
        except Exception as e:
            return str(e), 500

@api.route('/deleteCorpus/')
class DeleteCorpus(Resource):
    @api.doc(parser=parser)
    def post(self):
        args = parser.parse_args()
        corpus_name = args['corpus_name']
        try:
            sc.delete_corpus(corpus_name)
            return '', 200
        except Exception as e:
            return str(e), 500

@api.route('/listAllCorpus/')
class listAllCorpus(Resource):
    def get(self):
        try:
            corpus_lst, code = sc.list_corpus_collections()
            return corpus_lst, code
        except Exception as e:
            return str(e), 500

@api.route('/listCorpusModels/')
class listCorpusModels(Resource):
    @api.doc(parser=parser2)
    def get(self):
        args = parser2.parse_args()
        corpus_col = args['corpus_col']
        try:
            corpus_lst, code = sc.get_corpus_models(corpus_col=corpus_col)
            return corpus_lst, code
        except Exception as e:
            return str(e), 500
        
@api.route('/listCorpusMetadataDisplayed')
class listCorpusMetadataDisplayed(Resource):
    @api.doc(parser=parser3)
    def get(self):
        args = parser3.parse_args()
        corpus_col = args['corpus_col']
        try:
            SearcheableField_lst, code = sc.get_corpus_MetadataDisplayed(corpus_col=corpus_col)
            return SearcheableField_lst, code
        except Exception as e:
            return str(e), 500
        
@api.route('/listCorpusSearcheableFields/')
class listCorpusSearcheableFields(Resource):
    @api.doc(parser=parser4)
    def get(self):
        args = parser4.parse_args()
        corpus_col = args['corpus_col']
        try:
            SearcheableField_lst, code = sc.get_corpus_SearcheableField(corpus_col=corpus_col)
            return SearcheableField_lst, code
        except Exception as e:
            return str(e), 500
        
@api.route('/addSearcheableFields/')
class addSearcheableFields(Resource):
    @api.doc(parser=parser5)
    def post(self):
        args = parser5.parse_args()
        corpus_col = args['corpus_col']
        search_fields = args['searchable_fields']
        try:
            sc.modify_corpus_SearcheableFields(
                SearcheableFields=search_fields,
                corpus_col=corpus_col,
                action="add")
            return '', 200
        except Exception as e:
            return str(e), 500
        
@api.route('/deleteSearcheableFields/')
class deleteSearcheableFields(Resource):
    @api.doc(parser=parser6)
    def post(self):
        args = parser6.parse_args()
        corpus_col = args['corpus_col']
        search_fields = args['searchable_fields']
        try:
            sc.modify_corpus_SearcheableFields(
                SearcheableFields=search_fields,
                corpus_col=corpus_col,
                action="remove")
            return '', 200
        except Exception as e:
            return str(e), 500
        