"""
This script defines a Flask RESTful namespace for managing Solr queries.

Author: Lorena Calvo-Bartolomé
Date: 13/04/2023
"""

from flask_restx import Namespace, Resource, reqparse
from src.core.clients.np_solr_client import NPSolrClient

# ======================================================
# Define namespace for managing queries
# ======================================================
api = Namespace(
    'Queries', description='Specfic Solr queries.')

# ======================================================
# Namespace variables
# ======================================================
# Create Solr client
sc = NPSolrClient(api.logger)

# Define parsers to take inputs from user
q1_parser = reqparse.RequestParser()
q1_parser.add_argument(
    'corpus_collection', help='Name of the corpus collection', required=True)
q1_parser.add_argument(
    'doc_id', help='ID of the document whose doc-topic distribution associated to a specific model is to be retrieved', required=True)
q1_parser.add_argument(
    'model_name', help='Name of the model reponsible for the creation of the doc-topic distribution to be retrieved', required=True)

q2_parser = reqparse.RequestParser()
q2_parser.add_argument(
    'corpus_collection', help='Name of the corpus collection', required=True)

q3_parser = reqparse.RequestParser()
q3_parser.add_argument(
    'collection', help='Name of the collection', required=True)

q5_parser = reqparse.RequestParser()
q5_parser.add_argument(
    'corpus_collection', help='Name of the corpus collection', required=True)
q5_parser.add_argument(
    'model_name', help='Name of the model reponsible for the creation of the doc-topic distribution', required=True)
q5_parser.add_argument(
    'doc_id', help="ID of the document whose similarity is going to be checked against all other documents in 'corpus_collection'", required=True)
q5_parser.add_argument(
    'start', help='Specifies an offset (by default, 0) into the responses at which Solr should begin displaying content', required=False)
q5_parser.add_argument(
    'rows', help='Controls how many rows of responses are displayed at a time (default value: maximum number of docs in the collection)', required=False)

q6_parser = reqparse.RequestParser()
q6_parser.add_argument(
    'corpus_collection', help='Name of the corpus collection', required=True)
q6_parser.add_argument(
    'doc_id', help="ID of the document whose metadata is going to be retrieved'", required=True)

q7_parser = reqparse.RequestParser()
q7_parser.add_argument(
    'corpus_collection', help='Name of the corpus collection', required=True)
q7_parser.add_argument(
    'string', help="String to be search in the SearcheableField field'", required=True)
q7_parser.add_argument(
    'start', help='Specifies an offset (by default, 0) into the responses at which Solr should begin displaying content', required=False)
q7_parser.add_argument(
    'rows', help='Controls how many rows of responses are displayed at a time (default value: maximum number of docs in the collection)', required=False)

q8_parser = reqparse.RequestParser()
q8_parser.add_argument(
    'model_collection', help='Name of the model collection', required=True)
q8_parser.add_argument(
    'start', help='Specifies an offset (by default, 0) into the responses at which Solr should begin displaying content', required=False)
q8_parser.add_argument(
    'rows', help='Controls how many rows of responses are displayed at a time (default value: maximum number of docs in the collection)', required=False)

q9_parser = reqparse.RequestParser()
q9_parser.add_argument(
    'corpus_collection', help='Name of the corpus collection', required=True)
q9_parser.add_argument(
    'model_name', help='Name of the model reponsible for the creation of the doc-topic distribution', required=True)
q9_parser.add_argument(
    'topic_id', help="ID of the topic whose top documents according to 'model_name' are being searched", required=True)
q9_parser.add_argument(
    'start', help='Specifies an offset (by default, 0) into the responses at which Solr should begin displaying content', required=False)
q9_parser.add_argument(
    'rows', help='Controls how many rows of responses are displayed at a time (default value: maximum number of docs in the collection)', required=False)

q10_parser = reqparse.RequestParser()
q10_parser.add_argument(
    'model_collection', help='Name of the model collection', required=True)
q10_parser.add_argument(
    'start', help='Specifies an offset (by default, 0) into the responses at which Solr should begin displaying content', required=False)
q10_parser.add_argument(
    'rows', help='Controls how many rows of responses are displayed at a time (default value: maximum number of docs in the collection)', required=False)

q14_parser = reqparse.RequestParser()
q14_parser.add_argument(
    'corpus_collection', help='Name of the corpus collection', required=True)
q14_parser.add_argument(
    'model_name', help='Name of the model responsible for the creation of the doc-topic distribution', required=True)
q14_parser.add_argument(
    'text_to_infer', help="Text to be inferred", required=True)
q14_parser.add_argument(
    'start', help='Specifies an offset (by default, 0) into the responses at which Solr should begin displaying content', required=False)
q14_parser.add_argument(
    'rows', help='Controls how many rows of responses are displayed at a time (default value: maximum number of docs in the collection)', required=False)

q20_parser = reqparse.RequestParser()
q20_parser.add_argument(
    'corpus_collection', help='Name of the corpus collection', required=True)
q20_parser.add_argument(
    'model_collection', help='Name of the model collection', required=True)
q20_parser.add_argument(
    'word', help="Word to search for documents that are similar to it.", required=True)
q20_parser.add_argument(
    'start', help='Specifies an offset (by default, 0) into the responses at which Solr should begin displaying content', required=False)
q20_parser.add_argument(
    'rows', help='Controls how many rows of responses are displayed at a time (default value: maximum number of docs in the collection)', required=False)

q21_parser = reqparse.RequestParser()
q21_parser.add_argument(
    'corpus_collection', help='Name of the corpus collection', required=True)
q21_parser.add_argument(
    'free_text', help="Document (free text) to search for documents that are similar to it.", required=True)
q21_parser.add_argument(
    "keyword", 
    help="An optional keyword used for filtering documents. If provided, the search will return documents that contain this keyword in addition to semantic similarity.", 
    required=False)
q21_parser.add_argument(
    'start', help='Specifies an offset (by default, 0) into the responses at which Solr should begin displaying content', required=False)
q21_parser.add_argument(
    'rows', help='Controls how many rows of responses are displayed at a time (default value: maximum number of docs in the collection)', required=False)

q22_parser = reqparse.RequestParser()
q22_parser.add_argument(
    'text_to_infer',
    help='Input text to be inferred. Separate multiple texts with commas.',
    required=True
)
q22_parser.add_argument(
    'cpv',
    help='The first two digits of the CPV code corresponding to the text.',
    required=False
)
q22_parser.add_argument(
    'granularity',
    help='Specifies the level of topic detail: "large" (more topics) or "small" (fewer topics). Default is "large".',
    default="large",
    required=False
)

q22_parser.add_argument(
    'model_name',
    help="Model to be used for the inference in case 'cpv' and 'granularity' are not provided.",
    required=False
)

@api.route('/getThetasDocById/')
class getThetasDocById(Resource):
    @api.doc(parser=q1_parser)
    def get(self):
        args = q1_parser.parse_args()
        corpus_collection = args['corpus_collection']
        doc_id = args['doc_id']
        model_name = args['model_name']

        try:
            return sc.do_Q1(corpus_col=corpus_collection,
                            doc_id=doc_id,
                            model_name=model_name)
        except Exception as e:
            return str(e), 500


@api.route('/getCorpusMetadataFields/')
class getCorpusMetadataFields(Resource):
    @api.doc(parser=q2_parser)
    def get(self):
        args = q2_parser.parse_args()
        corpus_collection = args['corpus_collection']
        try:
            return sc.do_Q2(corpus_col=corpus_collection)
        except Exception as e:
            return str(e), 500


@api.route('/getNrDocsColl/')
class getNrDocsColl(Resource):
    @api.doc(parser=q3_parser)
    def get(self):
        args = q3_parser.parse_args()
        collection = args['collection']
        try:
            return sc.do_Q3(col=collection)
        except Exception as e:
            return str(e), 500


@api.route('/getDocsWithHighSimWithDocByid/')
class getDocsWithHighSimWithDocByid(Resource):
    @api.doc(parser=q5_parser)
    def get(self):
        args = q5_parser.parse_args()
        corpus_collection = args['corpus_collection']
        model_name = args['model_name']
        doc_id = args['doc_id']
        start = args['start']
        rows = args['rows']

        try:
            return sc.do_Q5(corpus_col=corpus_collection,
                            model_name=model_name,
                            doc_id=doc_id,
                            start=start,
                            rows=rows)
        except Exception as e:
            return str(e), 500


@api.route('/getMetadataDocById/')
class getMetadataDocById(Resource):
    @api.doc(parser=q6_parser)
    def get(self):
        args = q6_parser.parse_args()
        corpus_collection = args['corpus_collection']
        doc_id = args['doc_id']

        try:
            return sc.do_Q6(corpus_col=corpus_collection,
                            doc_id=doc_id)
        except Exception as e:
            return str(e), 500


@api.route('/getDocsWithString/')
class getDocsWithString(Resource):
    @api.doc(parser=q7_parser)
    def get(self):
        args = q7_parser.parse_args()
        corpus_collection = args['corpus_collection']
        string = args['string']
        start = args['start']
        rows = args['rows']

        try:
            return sc.do_Q7(corpus_col=corpus_collection,
                            string=string,
                            start=start,
                            rows=rows)
        except Exception as e:
            return str(e), 500


@api.route('/getTopicsLabels/')
class getTopicsLabels(Resource):
    @api.doc(parser=q8_parser)
    def get(self):
        args = q8_parser.parse_args()
        model_collection = args['model_collection']
        start = args['start']
        rows = args['rows']

        try:
            return sc.do_Q8(model_col=model_collection,
                            start=start,
                            rows=rows)
        except Exception as e:
            return str(e), 500


@api.route('/getTopicTopDocs/')
class getTopicTopDocs(Resource):
    @api.doc(parser=q9_parser)
    def get(self):
        args = q9_parser.parse_args()
        corpus_collection = args['corpus_collection']
        model_name = args['model_name']
        topic_id = args['topic_id']
        start = args['start']
        rows = args['rows']

        try:
            return sc.do_Q9(corpus_col=corpus_collection,
                            model_name=model_name,
                            topic_id=topic_id,
                            start=start,
                            rows=rows)
        except Exception as e:
            return str(e), 500


@api.route('/getModelInfo/')
class getModelInfo(Resource):
    @api.doc(parser=q10_parser)
    def get(self):
        args = q10_parser.parse_args()
        model_collection = args['model_collection']
        start = args['start']
        rows = args['rows']
        try:
            return sc.do_Q10(model_col=model_collection,
                             start=start,
                             rows=rows,
                             only_id=False)
        except Exception as e:
            return str(e), 500


@api.route('/getDocsSimilarToFreeTextTM/')
class getDocsSimilarToFreeTextTM(Resource):
    @api.doc(parser=q14_parser)
    def get(self):
        args = q14_parser.parse_args()
        corpus_collection = args['corpus_collection']
        model_name = args['model_name']
        text_to_infer = args['text_to_infer']
        start = args['start']
        rows = args['rows']

        try:
            return sc.do_Q14(corpus_col=corpus_collection,
                             model_name=model_name,
                             text_to_infer=text_to_infer,
                             start=start,
                             rows=rows)
        except Exception as e:
            return str(e), 500


@api.route('/getDocsRelatedToWord/')
class getDocsRelatedToWord(Resource):
    @api.doc(parser=q20_parser)
    def get(self):
        args = q20_parser.parse_args()
        corpus_collection = args['corpus_collection']
        model_collection = args['model_collection']
        search_word = args['word']
        start = args['start']
        rows = args['rows']
        try:
            return sc.do_Q20(
                corpus_col=corpus_collection,
                model_name=model_collection,
                search_word=search_word,
                embedding_model="word2vec",
                start=start,
                rows=rows
            )
        except Exception as e:
            return str(e), 500


@api.route('/getDocsSimilarToFreeTextEmb/')
class getDocsSimilarToFreeTextEmb(Resource):
    @api.doc(parser=q21_parser)
    def get(self):
        args = q21_parser.parse_args()
        corpus_collection = args['corpus_collection']
        doc = args['free_text']
        start = args['start']
        rows = args['rows']
        try:
            return sc.do_Q21(
                corpus_col=corpus_collection,
                search_doc=doc,
                embedding_model="bert",
                start=start,
                rows=rows
            )
        except Exception as e:
            return str(e), 500
        
@api.route('/inferTopicInformation/')
class inferTopicInformation(Resource):
    @api.doc(parser=q22_parser)
    def get(self):
        args = q22_parser.parse_args()
        
        if args['cpv'] is None and args['model_name'] is None:
            return "CPV code or model name not provided.", 500
        
        if args['cpv'] is not None:
            model_name = args['cpv'] + "_" + args['granularity']
        else:
            model_name = args['model_name']
            
        text_to_infer = args['text_to_infer']

        try:
            return sc.do_Q22(model_name=model_name, text_to_infer=text_to_infer)
        except Exception as e:
            return str(e), 500
