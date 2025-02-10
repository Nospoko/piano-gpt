import os

import pandas as pd
import sqlalchemy as sa
from dotenv import load_dotenv

load_dotenv()
POSTGRES_PASSWORD = os.environ["POSTGRES_PASSWORD"]
POSTGRES_USER = os.environ["POSTGRES_USER"]
POSTGRES_HOST = os.environ["POSTGRES_HOST"]
POSTGRES_PORT = os.environ["POSTGRES_PORT"]
DB_DSN = (
    f"postgresql://{POSTGRES_HOST}:{POSTGRES_PORT}/midi_transformers?user={POSTGRES_USER}&password={POSTGRES_PASSWORD}"
)


class DatabaseConnection:
    def __init__(self):
        db_url = sa.engine.make_url(DB_DSN)
        # Pre-ping allows the connection to stay alive in long running sessions with low activity
        # https://docs.sqlalchemy.org/en/20/core/pooling.html#sqlalchemy.pool.Pool.params.pre_ping
        self.__engine = sa.create_engine(db_url, pool_pre_ping=True)

    def open(self):
        db_url = sa.engine.make_url(DB_DSN)
        if self.__engine is None:
            self.__engine = sa.create_engine(db_url, pool_pre_ping=True)

    def close(self):
        if self.__engine is not None:
            self.__engine.dispose()
            self.__engine = None

    @property
    def engine(self) -> sa.engine.Engine:
        if self.__engine is None:
            raise RuntimeError("Database connection is not open. Call 'open()' first.")
        return self.__engine

    def execute(self, query: str):
        with self.__engine.connect() as connection:
            result = connection.execute(sa.text(query))

        return result

    def read_sql(self, sql: str, **kwargs) -> pd.DataFrame:
        with self.__engine.connect() as connection:
            df = pd.read_sql(
                sql=sql,
                con=connection,
                **kwargs,
            )
        return df

    def to_sql(self, df: pd.DataFrame, table: str, **kwargs):
        with self.__engine.connect() as connection:
            df.to_sql(
                table,
                con=connection,
                **kwargs,
            )


database_cnx = DatabaseConnection()
