import sys
from awsglue.transforms import *
from awsglue.utils import getResolvedOptions
from pyspark.context import SparkContext
from awsglue.context import GlueContext
from awsglue.job import Job
import preprocess
from preprocess import Preprocessor
import os
path= "./preprocess_6.zip/glove_twitter_200d_clean.txt"

pre=Preprocessor(path)

## @params: [JOB_NAME]
args = getResolvedOptions(sys.argv, ['JOB_NAME'])

sc = SparkContext()
glueContext = GlueContext(sc) 
spark = glueContext.spark_session
job = Job(glueContext)
job.init(args['JOB_NAME'], args)
## @type: DataSource
## @args: [database = "hw4", table_name = "eval", transformation_ctx = "datasource0"]
## @return: datasource0
## @inputs: []
datasource0 = glueContext.create_dynamic_frame.from_catalog(database = "hw4", table_name = "dev", transformation_ctx = "datasource0")
## @type: ApplyMapping
## @args: [mapping = [("sentiment", "long", "sentiment", "long"), ("tweet", "string", "tweet", "string")], transformation_ctx = "applymapping1"]
## @return: applymapping1
## @inputs: [frame = datasource0]
applymapping1 = ApplyMapping.apply(frame = datasource0, mappings = [("sentiment", "long", "sentiment", "long"), ("tweet", "string", "tweet", "string")], transformation_ctx = "applymapping1")
## @type: Map
## @args: [f = <function>, transformation_ctx = "<transformation_ctx>"]
## @return: <output>
## @inputs: [frame = <frame>]
def map_function(dynamicRecord):
    dynamicRecord["features"]=pre.pipeline(dynamicRecord["tweet"])
    # dynamicRecord["features"]=dynamicRecord["tweet"]
    return dynamicRecord

process_tweet = Map.apply(frame = applymapping1, f = map_function, transformation_ctx = "process_tweets")
## @type: DataSink
## @args: [connection_type = "s3", connection_options = {"path": "s3://as5646"}, format = "json", transformation_ctx = "datasink2"]
## @return: datasink2
## @inputs: [frame = applymapping1]
datasink2 = glueContext.write_dynamic_frame.from_options(frame = process_tweet, connection_type = "s3", connection_options = {"path": "s3://as5646/"}, format = "json", transformation_ctx = "datasink2")
job.commit()