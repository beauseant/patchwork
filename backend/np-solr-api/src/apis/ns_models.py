"""
This script defines a Flask RESTful namespace for managing models stored in Solr as collections. 

Author: Lorena Calvo-Bartolomé
Date: 27/03/2023
"""

from flask_restx import Namespace, Resource, reqparse
from src.core.clients.np_solr_client import NPSolrClient

# ======================================================
# Define namespace for managing models
# ======================================================
api = Namespace(
    'Models', description='Models-related operations (i.e., index/delete models))')

# ======================================================
# Namespace variables
# ======================================================
# Create Solr client
sc = NPSolrClient(api.logger)

# Define parser to take inputs from user
parser = reqparse.RequestParser()
parser.add_argument(
    'model_name', help="Name of the model to index/index. You should specify the name of the folder in which topic model information is stored.")

parser_add_rel = reqparse.RequestParser()
parser_add_rel.add_argument(
    'model_name', help="Name of the model to which the relevant topic will be added")
parser_add_rel.add_argument(
    'topic_id', help="Topic id of the relevant topic to be added")
parser_add_rel.add_argument(
    'user', help="User who is adding the relevant topic")

parser_del_rel = reqparse.RequestParser()
parser_del_rel.add_argument(
    'model_name', help="Name of the model to which the relevant topic will be removed")
parser_del_rel.add_argument(
    'topic_id', help="Topic id of the relevant topic to be removed")
parser_del_rel.add_argument(
    'user', help="User who is deleting the relevant topic")

@api.route('/indexModel/')
class IndexModel(Resource):
    @api.doc(parser=parser)
    def post(self):
        args = parser.parse_args()
        model_name = args['model_name']
        try:
            sc.index_model(model_name)
            return '', 200
        except Exception as e:
            return str(e), 500


@api.route('/deleteModel/')
class DeleteModel(Resource):
    @api.doc(parser=parser)
    def post(self):
        args = parser.parse_args()
        model_name = args['model_name']
        try:
            sc.delete_model(model_name)
            return '', 200
        except Exception as e:
            return str(e), 500

@api.route('/listAllModels/')
class ListAllModels(Resource):
    def get(self):
        try:
            models_lst, code = sc.list_model_collections()
            
            # models have the form "0_45_6_topics" -> "iter_cpv_topics"
            # make a list of dictionaries with key cpv and value granularity
            
            # sort elements in models_lst so all 0_45_XX appear together
            models_lst.sort(key=lambda x: (x.split('_')[1], x.split('_')[2]))
            models_lst_formatted = []
            for i in range(len(models_lst)):
                parts = models_lst[i].split('_')
                cpv = parts[1]
                granularity = parts[2]
                models_lst_formatted.append({'cpv': cpv, 'granularity': granularity})
                
            # group by cpv
            grouped_models = {}
            for model in models_lst_formatted:
                cpv = model['cpv']
                granularity = model['granularity']
                if cpv not in grouped_models:
                    grouped_models[cpv] = []
                grouped_models[cpv].append(granularity)
            
            # convert each granularity value to high-level:granularity or low-level:granularity
            for cpv in grouped_models:
                granularities = grouped_models[cpv]
                granularities = [int(g) for g in granularities]
                granularities_formatted = []
                
                if granularities[0] < granularities[1]:
                    granularities_formatted.append(
                       {"low":granularities[0]}
                    )
                    granularities_formatted.append(
                       {"high":granularities[1]}
                    )
                else:
                    granularities_formatted.append(
                       {"high":granularities[0]}
                    )
                    granularities_formatted.append(
                       {"low":granularities[1]}
                    )
                grouped_models[cpv] = granularities_formatted
                    
            return grouped_models, code
        except Exception as e:
            return str(e), 500