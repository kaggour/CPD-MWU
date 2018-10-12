#!/bin/bash

export LD_LIBRARY_PATH=/home/aggour/rpi/dissertation/python/lib
echo $LD_LIBRARY_PATH
export PATH=/home/aggour/rpi/dissertation/python/bin:$PATH
export SPARK_CONF_DIR=/home/aggour/rpi/dissertation/python/conf
echo $SPARK_CONF_DIR

spark-submit \
   --master yarn --deploy-mode client \
   --conf spark.network.timeout=10000000 \
   --conf spark.pyspark.python=/home/aggour/rpi/dissertation/python/bin/python \
   --conf spark.driver.extraLibraryPath=/home/aggour/rpi/dissertation/python/lib \
   --conf spark.driver.extraClassPath=/home/aggour/rpi/dissertation/python/bin \
   --conf spark.executor.extraLibraryPath=/home/aggour/rpi/dissertation/python/lib \
   --conf spark.executor.extraClassPath=/home/aggour/rpi/dissertation/python/bin \
   --py-files /home/aggour/rpi/dissertation/spark/lib.zip \
   --num-executors 1200 \
    /home/aggour/rpi/dissertation/spark/$@

