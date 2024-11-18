import pandas as pd
from pyspark.sql import SparkSession
from delta import configure_spark_with_delta_pip


def get_spark_session() -> SparkSession:
    """
    Create and return a Spark session configured with Delta Lake extensions.

    This function initializes a Spark session with the necessary configurations
    to support Delta Lake operations. It includes settings for the Delta core library
    and enables the Delta catalog.

    Returns:
    -------
        SparkSession: A Spark session with Delta Lake support enabled.
    """

    builder = (
        SparkSession.builder.appName("DeltaApp")
        .config("spark.jars.packages", "io.delta:delta-core_2.12:3.2.1")
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
        .config(
            "spark.sql.catalog.spark_catalog",
            "org.apache.spark.sql.delta.catalog.DeltaCatalog",
        )
    )

    spark = configure_spark_with_delta_pip(builder).getOrCreate()
    return spark


def write_to_delta(spark: SparkSession, df: pd.DataFrame, delta_path: str) -> None:
    """
    Write a Pandas DataFrame to a Delta Lake table.

    This function converts the given Pandas DataFrame to a Spark DataFrame and
    writes it to the specified Delta Lake path. The write operation will
    overwrite any existing data at the specified location.

    Returns:
    -------
        None: This function does not return any value.
    """

    spark_df = spark.createDataFrame(df)
    spark_df.write.format("delta").mode("overwrite").save(delta_path)


def read_from_delta(spark: SparkSession, delta_path: str):
    """
    Read data from a Delta Lake table into a Spark DataFrame.

    This function loads data from a specified Delta Lake path and returns it as
    a Spark DataFrame, allowing for further data processing or analysis.

    Returns:
    -------
        DataFrame: A Spark DataFrame containing the data read from the Delta Lake table.
    """

    return spark.read.format("delta").load(delta_path)
